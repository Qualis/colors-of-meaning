import logging
from typing import Any, List
import numpy.typing as npt
import torch
import torch.nn as nn
from pathlib import Path

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.shared.determinism import seed_everything

logger = logging.getLogger(__name__)


def offdiagonal_entries(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.shape[0]
    keep = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return matrix[keep]


class LabProjectorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_2, 3),
        )

        self.l_activation = nn.Sigmoid()
        self.ab_activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x)

        lightness = self.l_activation(features[:, 0:1]) * 100.0
        a_val = self.ab_activation(features[:, 1:2]) * 127.5
        b_val = self.ab_activation(features[:, 2:3]) * 127.5

        return torch.cat([lightness, a_val, b_val], dim=1)


class PyTorchColorMapper(ColorMapper):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._generator = seed_everything(seed)
        self.network = LabProjectorNetwork(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self._epoch_checkpoints: List[Any] = []

    def embed_to_lab(self, embedding: npt.NDArray) -> LabColor:
        self.network.eval()
        with torch.no_grad():
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            lab_tensor = self.network(embedding_tensor)
            lab_values = lab_tensor.cpu().numpy()[0]

        return LabColor.from_unclamped(lab_values[0], lab_values[1], lab_values[2])

    def embed_batch_to_lab(self, embeddings: npt.NDArray) -> List[LabColor]:
        self.network.eval()
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            lab_tensor = self.network(embeddings_tensor)
            lab_values = lab_tensor.cpu().numpy()

        return [LabColor.from_unclamped(row[0], row[1], row[2]) for row in lab_values]

    def train(self, embeddings: npt.NDArray, epochs: int, learning_rate: float) -> None:
        self.network.train()

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

        batch_size = min(32, len(embeddings))
        num_batches = (len(embeddings) + batch_size - 1) // batch_size

        self._epoch_checkpoints = []
        for epoch in range(epochs):
            avg_loss = self._train_epoch(embeddings_tensor, optimizer, batch_size, num_batches)
            self._epoch_checkpoints.append(self._capture_state())

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, avg_loss)

    def _capture_state(self) -> dict:
        return {key: value.clone() for key, value in self.network.state_dict().items()}

    def epoch_checkpoints(self) -> List[Any]:
        return self._epoch_checkpoints

    def restore_checkpoint(self, checkpoint: Any) -> None:
        self.network.load_state_dict(checkpoint)

    def _train_epoch(
        self,
        embeddings_tensor: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        num_batches: int,
    ) -> float:
        total_loss = 0.0
        indices = torch.randperm(len(embeddings_tensor), generator=self._generator)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(embeddings_tensor))
            batch_embeddings = embeddings_tensor[indices[start_idx:end_idx]]

            total_loss += self._train_batch(batch_embeddings, optimizer)

        return total_loss / num_batches

    def _train_batch(self, batch_embeddings: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        optimizer.zero_grad()
        loss = self._structure_loss(batch_embeddings)
        loss.backward()
        optimizer.step()

        return loss.item()

    def _structure_loss(self, batch_embeddings: torch.Tensor) -> torch.Tensor:
        lab_output = self.network(batch_embeddings)
        teacher_similarity = self._teacher_similarity(batch_embeddings)
        student_similarity = self._student_similarity(lab_output)

        return self._similarity_discrepancy(student_similarity, teacher_similarity)

    def _teacher_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self._cosine_similarity_matrix(embeddings).detach()

    def _student_similarity(self, lab_output: torch.Tensor) -> torch.Tensor:
        centred_lab = lab_output - lab_output.mean(dim=0, keepdim=True)
        return self._cosine_similarity_matrix(centred_lab)

    @staticmethod
    def _cosine_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
        normalized = nn.functional.normalize(vectors, p=2, dim=1)
        return normalized @ normalized.t()

    @staticmethod
    def _similarity_discrepancy(
        student_similarity: torch.Tensor,
        teacher_similarity: torch.Tensor,
    ) -> torch.Tensor:
        student_offdiagonal = offdiagonal_entries(student_similarity)

        if student_offdiagonal.numel() == 0:
            return student_similarity.sum() * 0.0

        teacher_offdiagonal = offdiagonal_entries(teacher_similarity)
        return nn.functional.mse_loss(student_offdiagonal, teacher_offdiagonal)

    def save_weights(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.network.eval()
