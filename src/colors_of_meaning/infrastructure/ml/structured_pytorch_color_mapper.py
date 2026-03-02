from typing import List
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.cluster import KMeans  # type: ignore[import-untyped]

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.ml.structured_lab_projector_network import (
    StructuredLabProjectorNetwork,
)


class StructuredPyTorchColorMapper(ColorMapper):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
        device: str = "cpu",
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        num_clusters: int = 16,
        max_chroma: float = 128.0,
    ) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_clusters = num_clusters
        self.max_chroma = max_chroma
        self.network = StructuredLabProjectorNetwork(
            input_dim=input_dim,
            hidden_dim_1=hidden_dim_1,
            hidden_dim_2=hidden_dim_2,
            dropout_rate=dropout_rate,
            max_chroma=max_chroma,
        ).to(self.device)

    def embed_to_lab(self, embedding: npt.NDArray) -> LabColor:
        self.network.eval()
        with torch.no_grad():
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            lab_tensor = self.network(embedding_tensor)
            lab_values = lab_tensor.cpu().numpy()[0]

        return LabColor(
            l=float(lab_values[0]),
            a=float(lab_values[1]),
            b=float(lab_values[2]),
        ).clamp()

    def embed_batch_to_lab(self, embeddings: npt.NDArray) -> List[LabColor]:
        self.network.eval()
        with torch.no_grad():
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            lab_tensor = self.network(embeddings_tensor)
            lab_values = lab_tensor.cpu().numpy()

        return [
            LabColor(
                l=float(row[0]),
                a=float(row[1]),
                b=float(row[2]),
            ).clamp()
            for row in lab_values
        ]

    def train(self, embeddings: npt.NDArray, epochs: int, learning_rate: float) -> None:
        self.network.train()

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        targets = self._prepare_targets(embeddings)

        optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        batch_size = min(32, len(embeddings))
        num_batches = (len(embeddings) + batch_size - 1) // batch_size

        best_state = self._run_training_loop(
            embeddings_tensor, targets, optimizer, scheduler, batch_size, num_batches, epochs
        )

        if best_state is not None:
            self.network.load_state_dict(best_state)

    def _run_training_loop(
        self,
        embeddings_tensor: torch.Tensor,
        targets: tuple,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        batch_size: int,
        num_batches: int,
        epochs: int,
    ) -> dict:
        best_loss = float("inf")
        best_state: dict = {}

        for epoch in range(epochs):
            avg_loss = self._train_epoch(embeddings_tensor, targets, optimizer, batch_size, num_batches)
            scheduler.step()

            best_loss, best_state = self._checkpoint_if_improved(avg_loss, best_loss, best_state)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        return best_state

    def _checkpoint_if_improved(self, avg_loss: float, best_loss: float, best_state: dict) -> tuple:
        if avg_loss < best_loss:
            return avg_loss, {k: v.clone() for k, v in self.network.state_dict().items()}
        return best_loss, best_state

    def _prepare_targets(self, embeddings: npt.NDArray) -> tuple:
        hue_targets, lightness_targets, chroma_targets = self._derive_targets(embeddings)
        return (
            hue_targets.to(self.device),
            lightness_targets.to(self.device),
            chroma_targets.to(self.device),
        )

    def _train_epoch(
        self,
        embeddings_tensor: torch.Tensor,
        targets: tuple,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        num_batches: int,
    ) -> float:
        hue_targets, lightness_targets, chroma_targets = targets
        total_loss = 0.0
        indices = torch.randperm(len(embeddings_tensor))

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(embeddings_tensor))
            batch_indices = indices[start_idx:end_idx]

            total_loss += self._train_batch(
                embeddings_tensor[batch_indices],
                hue_targets[batch_indices],
                lightness_targets[batch_indices],
                chroma_targets[batch_indices],
                optimizer,
            )

        return total_loss / num_batches

    def _train_batch(
        self,
        batch_embeddings: torch.Tensor,
        batch_hue: torch.Tensor,
        batch_lightness: torch.Tensor,
        batch_chroma: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        optimizer.zero_grad()

        pred_l, pred_hue, pred_chroma = self.network.forward_structured(batch_embeddings)

        loss = self._compute_loss(pred_l, pred_hue, pred_chroma, batch_lightness, batch_hue, batch_chroma)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()

    def save_weights(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        self.network.eval()

    def _compute_loss(
        self,
        pred_lightness: torch.Tensor,
        pred_hue: torch.Tensor,
        pred_chroma: torch.Tensor,
        target_lightness: torch.Tensor,
        target_hue: torch.Tensor,
        target_chroma: torch.Tensor,
    ) -> torch.Tensor:
        hue_loss = torch.mean(1.0 - torch.cos(pred_hue - target_hue))
        lightness_loss = nn.functional.mse_loss(pred_lightness, target_lightness)
        chroma_loss = nn.functional.mse_loss(pred_chroma, target_chroma)

        return self.alpha * hue_loss + self.beta * lightness_loss + self.gamma * chroma_loss

    def _derive_targets(self, embeddings: npt.NDArray) -> tuple:
        hue_targets = self._derive_hue_targets(embeddings)
        lightness_targets = self._derive_lightness_targets(embeddings)
        chroma_targets = self._derive_chroma_targets(embeddings)
        return hue_targets, lightness_targets, chroma_targets

    def _derive_hue_targets(self, embeddings: npt.NDArray) -> torch.Tensor:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normalized = embeddings / norms

        num_clusters = min(self.num_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normalized)

        hue_angles = np.array(
            [2.0 * np.pi * label / num_clusters for label in labels],
            dtype=np.float32,
        )

        return torch.tensor(hue_angles, dtype=torch.float32).unsqueeze(1)

    def _derive_lightness_targets(self, embeddings: npt.NDArray) -> torch.Tensor:
        mean_activations = np.mean(embeddings, axis=1)
        min_val = mean_activations.min()
        max_val = mean_activations.max()
        range_val = max_val - min_val

        if range_val < 1e-8:
            scaled = np.full_like(mean_activations, 50.0)
        else:
            scaled = ((mean_activations - min_val) / range_val) * 100.0

        return torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)

    def _derive_chroma_targets(self, embeddings: npt.NDArray) -> torch.Tensor:
        variances = np.var(embeddings, axis=1)
        min_val = variances.min()
        max_val = variances.max()
        range_val = max_val - min_val

        if range_val < 1e-8:
            scaled = np.full_like(variances, self.max_chroma / 2.0)
        else:
            scaled = ((variances - min_val) / range_val) * self.max_chroma

        return torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)
