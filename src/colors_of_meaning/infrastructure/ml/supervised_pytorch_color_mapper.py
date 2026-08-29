import logging
from typing import Any, List, Optional
import numpy.typing as npt
import torch
import torch.nn as nn
from pathlib import Path

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import LabProjectorNetwork
from colors_of_meaning.infrastructure.ml.structure_objectives import offdiagonal_entries
from colors_of_meaning.shared.determinism import seed_everything

logger = logging.getLogger(__name__)


class SupervisedPyTorchColorMapper(ColorMapper):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
        device: str = "cpu",
        num_classes: int = 4,
        classification_weight: float = 0.1,
        contrastive_margin: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.classification_weight = classification_weight
        self.contrastive_margin = contrastive_margin
        self._generator = seed_everything(seed)
        self.network = LabProjectorNetwork(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            dropout_rate=dropout_rate,
        ).to(self.device)
        self.classification_head = nn.Linear(3, num_classes).to(self.device)
        self._training_labels: Optional[torch.Tensor] = None
        self._epoch_checkpoints: List[Any] = []

    def set_training_labels(self, labels: npt.NDArray) -> None:
        self._training_labels = torch.tensor(labels, dtype=torch.long, device=self.device)

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
        if self._training_labels is None:
            raise ValueError("Training labels must be set before training. Call set_training_labels first.")

        self.network.train()
        self.classification_head.train()

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        all_params = list(self.network.parameters()) + list(self.classification_head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        batch_size = min(32, len(embeddings))
        num_batches = (len(embeddings) + batch_size - 1) // batch_size

        self._epoch_checkpoints = []
        best_state = self._run_training_loop(
            embeddings_tensor,
            optimizer,
            scheduler,
            batch_size,
            num_batches,
            epochs,
        )

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _capture_state(self) -> dict:
        return {key: value.clone() for key, value in self.network.state_dict().items()}

    def epoch_checkpoints(self) -> List[Any]:
        return self._epoch_checkpoints

    def restore_checkpoint(self, checkpoint: Any) -> None:
        self.network.load_state_dict(checkpoint)

    def _run_training_loop(
        self,
        embeddings_tensor: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        batch_size: int,
        num_batches: int,
        epochs: int,
    ) -> Optional[dict]:
        best_loss = float("inf")
        best_state: Optional[dict] = None

        for epoch in range(epochs):
            avg_loss = self._train_epoch(embeddings_tensor, optimizer, batch_size, num_batches)
            scheduler.step()

            best_loss, best_state = self._checkpoint_if_improved(avg_loss, best_loss, best_state)
            self._epoch_checkpoints.append(self._capture_state())

            if (epoch + 1) % 10 == 0:
                logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, avg_loss)

        return best_state

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
            batch_indices = indices[start_idx:end_idx]

            total_loss += self._train_batch(
                embeddings_tensor[batch_indices],
                self._training_labels[batch_indices],  # type: ignore[index]
                optimizer,
            )

        return total_loss / num_batches

    def _train_batch(
        self,
        batch_embeddings: torch.Tensor,
        batch_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()

        lab_output = self.network(batch_embeddings)
        loss = self._compute_combined_loss(lab_output, batch_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.network.parameters()) + list(self.classification_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        return loss.item()

    def _compute_combined_loss(
        self,
        lab_output: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        structure_loss = self._contrastive_loss(lab_output, labels)

        class_logits = self.classification_head(lab_output)
        classification_loss = nn.functional.cross_entropy(class_logits, labels)

        return structure_loss + self.classification_weight * classification_loss

    def _contrastive_loss(self, lab_output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalised_lab = self._normalise_lab(lab_output)
        distances = torch.cdist(normalised_lab, normalised_lab)

        same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
        attraction = distances**2
        repulsion = torch.clamp(self.contrastive_margin - distances, min=0.0) ** 2
        pair_loss = torch.where(same_label, attraction, repulsion)

        offdiagonal = offdiagonal_entries(pair_loss)
        if offdiagonal.numel() == 0:
            return lab_output.sum() * 0.0

        return offdiagonal.mean()

    @staticmethod
    def _normalise_lab(lab_output: torch.Tensor) -> torch.Tensor:
        lightness = lab_output[:, 0:1] / 100.0
        chroma = lab_output[:, 1:3] / 127.5
        return torch.cat([lightness, chroma], dim=1)

    def _checkpoint_if_improved(
        self,
        avg_loss: float,
        best_loss: float,
        best_state: Optional[dict],
    ) -> tuple:
        if avg_loss < best_loss:
            return avg_loss, {k: v.clone() for k, v in self.network.state_dict().items()}
        return best_loss, best_state

    def save_weights(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.network.eval()
