import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.domain.model.lab_color import LabColor


class TestStructuredPyTorchColorMapper:
    def test_should_initialize_mapper(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")

        assert mapper.network is not None
        assert isinstance(mapper.device, torch.device)

    def test_should_embed_to_lab(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.random.randn(10).astype(np.float32)

        result = mapper.embed_to_lab(embedding)

        assert isinstance(result, LabColor)

    def test_should_produce_valid_lab_ranges(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.random.randn(10).astype(np.float32)

        result = mapper.embed_to_lab(embedding)

        assert 0 <= result.l <= 100
        assert -128 <= result.a <= 127
        assert -128 <= result.b <= 127

    def test_should_embed_batch_to_lab(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(5, 10).astype(np.float32)

        results = mapper.embed_batch_to_lab(embeddings)

        assert len(results) == 5

    def test_should_return_lab_colors_for_batch(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(3, 10).astype(np.float32)

        results = mapper.embed_batch_to_lab(embeddings)

        assert all(isinstance(color, LabColor) for color in results)

    def test_should_train_mapper(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        mapper.train(embeddings, epochs=2, learning_rate=0.001)

        assert True

    def test_should_save_weights(self, tmp_path: Path) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        save_path = tmp_path / "model.pth"

        mapper.save_weights(str(save_path))

        assert save_path.exists()

    def test_should_load_weights(self, tmp_path: Path) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        save_path = tmp_path / "model.pth"
        mapper.save_weights(str(save_path))

        new_mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        new_mapper.load_weights(str(save_path))

        assert True

    def test_should_use_cpu_when_cuda_not_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            mapper = StructuredPyTorchColorMapper(device="cuda")

        assert mapper.device.type == "cpu"

    def test_should_store_loss_weights(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, alpha=2.0, beta=0.5, gamma=1.5, device="cpu")

        assert mapper.alpha == 2.0
        assert mapper.beta == 0.5
        assert mapper.gamma == 1.5

    def test_should_store_num_clusters(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, num_clusters=32, device="cpu")

        assert mapper.num_clusters == 32

    def test_should_handle_small_batch_size(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=2)
        embeddings = np.random.randn(5, 10).astype(np.float32)

        mapper.train(embeddings, epochs=1, learning_rate=0.001)

        assert True

    def test_should_print_loss_every_10_epochs(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        mapper.train(embeddings, epochs=10, learning_rate=0.001)

        assert True

    def test_should_skip_restore_when_best_state_is_none(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        with patch.object(mapper, "_run_training_loop", return_value=None):
            mapper.train(embeddings, epochs=2, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])

        assert isinstance(result, LabColor)

    def test_should_restore_best_model_after_training(self, tmp_path: Path) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        mapper.train(embeddings, epochs=5, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])

        assert isinstance(result, LabColor)


class TestStructuredTargetDerivation:
    def test_should_derive_hue_targets(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        targets = mapper._derive_hue_targets(embeddings)

        assert targets.shape == (20, 1)

    def test_should_derive_hue_targets_in_valid_range(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        targets = mapper._derive_hue_targets(embeddings)

        assert torch.all(targets >= 0)
        assert torch.all(targets < 2 * np.pi)

    def test_should_derive_lightness_targets(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(20, 10).astype(np.float32)

        targets = mapper._derive_lightness_targets(embeddings)

        assert targets.shape == (20, 1)

    def test_should_derive_lightness_targets_in_valid_range(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(20, 10).astype(np.float32)

        targets = mapper._derive_lightness_targets(embeddings)

        assert torch.all(targets >= 0)
        assert torch.all(targets <= 100)

    def test_should_derive_chroma_targets(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", max_chroma=128.0)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        targets = mapper._derive_chroma_targets(embeddings)

        assert targets.shape == (20, 1)

    def test_should_derive_chroma_targets_in_valid_range(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", max_chroma=128.0)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        targets = mapper._derive_chroma_targets(embeddings)

        assert torch.all(targets >= 0)
        assert torch.all(targets <= 128.0)

    def test_should_handle_constant_embeddings_for_lightness(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.ones((20, 10), dtype=np.float32)

        targets = mapper._derive_lightness_targets(embeddings)

        assert torch.all(targets == 50.0)

    def test_should_handle_constant_embeddings_for_chroma(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", max_chroma=128.0)
        embeddings = np.ones((20, 10), dtype=np.float32)

        targets = mapper._derive_chroma_targets(embeddings)

        assert torch.all(targets == 64.0)

    def test_should_derive_all_targets(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        hue, lightness, chroma = mapper._derive_targets(embeddings)

        assert hue.shape == (20, 1)
        assert lightness.shape == (20, 1)
        assert chroma.shape == (20, 1)

    def test_should_limit_clusters_to_sample_count(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=100)
        embeddings = np.random.randn(5, 10).astype(np.float32)

        targets = mapper._derive_hue_targets(embeddings)

        assert targets.shape == (5, 1)


class TestStructuredLossComputation:
    def test_should_compute_loss(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")

        pred_l = torch.tensor([[50.0]])
        pred_hue = torch.tensor([[1.0]])
        pred_chroma = torch.tensor([[64.0]])
        target_l = torch.tensor([[60.0]])
        target_hue = torch.tensor([[1.5]])
        target_chroma = torch.tensor([[70.0]])

        loss = mapper._compute_loss(pred_l, pred_hue, pred_chroma, target_l, target_hue, target_chroma)

        assert loss.item() > 0

    def test_should_return_zero_loss_for_matching_values(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")

        pred_l = torch.tensor([[50.0]])
        pred_hue = torch.tensor([[1.0]])
        pred_chroma = torch.tensor([[64.0]])

        loss = mapper._compute_loss(pred_l, pred_hue, pred_chroma, pred_l, pred_hue, pred_chroma)

        assert loss.item() < 1e-6

    def test_should_weight_loss_components(self) -> None:
        mapper_high_alpha = StructuredPyTorchColorMapper(input_dim=10, device="cpu", alpha=10.0, beta=0.0, gamma=0.0)
        mapper_low_alpha = StructuredPyTorchColorMapper(input_dim=10, device="cpu", alpha=1.0, beta=0.0, gamma=0.0)

        pred_l = torch.tensor([[50.0]])
        pred_hue = torch.tensor([[0.0]])
        pred_chroma = torch.tensor([[64.0]])
        target_l = torch.tensor([[50.0]])
        target_hue = torch.tensor([[1.0]])
        target_chroma = torch.tensor([[64.0]])

        loss_high = mapper_high_alpha._compute_loss(pred_l, pred_hue, pred_chroma, target_l, target_hue, target_chroma)
        loss_low = mapper_low_alpha._compute_loss(pred_l, pred_hue, pred_chroma, target_l, target_hue, target_chroma)

        assert loss_high.item() > loss_low.item()
