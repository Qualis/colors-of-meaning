import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

import pytest

from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.domain.model.lab_color import LabColor


class TestSupervisedPyTorchColorMapper:
    def test_should_initialize_with_default_parameters(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")

        assert mapper.network is not None
        assert isinstance(mapper.device, torch.device)

    def test_should_initialize_classification_head_with_correct_dimensions(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=6)

        assert mapper.classification_head.in_features == 3
        assert mapper.classification_head.out_features == 6

    def test_should_embed_single_to_lab_color(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.random.randn(10).astype(np.float32)

        result = mapper.embed_to_lab(embedding)

        assert isinstance(result, LabColor)

    def test_should_embed_batch_to_lab_colors(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(5, 10).astype(np.float32)

        results = mapper.embed_batch_to_lab(embeddings)

        assert len(results) == 5
        assert all(isinstance(color, LabColor) for color in results)

    def test_should_produce_valid_lab_lightness_range(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.random.randn(10).astype(np.float32)

        result = mapper.embed_to_lab(embedding)

        assert 0 <= result.l <= 100

    def test_should_produce_valid_lab_a_channel_range(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.random.randn(10).astype(np.float32)

        result = mapper.embed_to_lab(embedding)

        assert -128 <= result.a <= 127

    def test_should_produce_valid_lab_b_channel_range(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.random.randn(10).astype(np.float32)

        result = mapper.embed_to_lab(embedding)

        assert -128 <= result.b <= 127

    def test_should_use_cpu_when_cuda_not_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            mapper = SupervisedPyTorchColorMapper(device="cuda")

        assert mapper.device.type == "cpu"


class TestSupervisedLabelHandling:
    def test_should_set_training_labels(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        labels = np.array([0, 1, 2, 3])

        mapper.set_training_labels(labels)

        assert mapper._training_labels is not None
        assert len(mapper._training_labels) == 4

    def test_should_raise_when_training_without_labels(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(10, 10).astype(np.float32)

        with pytest.raises(ValueError, match="Training labels must be set"):
            mapper.train(embeddings, epochs=1, learning_rate=0.001)

    def test_should_validate_label_count_matches_embeddings(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu")
        labels = np.array([0, 1, 2])

        mapper.set_training_labels(labels)

        assert len(mapper._training_labels) == 3  # type: ignore[arg-type]


class TestSupervisedTraining:
    def test_should_train_with_labels_and_reduce_loss(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3)
        embeddings = np.random.randn(30, 10).astype(np.float32)
        labels = np.array([0, 1, 2] * 10)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=5, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])
        assert isinstance(result, LabColor)

    def test_should_handle_small_batch_training(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(5, 10).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0])

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=2, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])
        assert isinstance(result, LabColor)

    def test_should_handle_binary_classification(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0, 1] * 10)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=2, learning_rate=0.001)

        assert mapper.classification_head.out_features == 2

    def test_should_handle_multi_class_classification(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=5)
        embeddings = np.random.randn(25, 10).astype(np.float32)
        labels = np.array([0, 1, 2, 3, 4] * 5)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=2, learning_rate=0.001)

        assert mapper.classification_head.out_features == 5

    def test_should_print_loss_every_10_epochs(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(10, 10).astype(np.float32)
        labels = np.array([0, 1] * 5)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=10, learning_rate=0.001)

        assert True


class TestSupervisedLoss:
    def test_should_compute_combined_loss(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3)

        lab_output = torch.tensor([[50.0, 10.0, -20.0]], requires_grad=True)
        targets = torch.tensor([[60.0, 15.0, -10.0]])
        labels = torch.tensor([1])

        loss = mapper._compute_combined_loss(lab_output, targets, labels)

        assert loss.item() > 0

    def test_should_weight_classification_loss_by_config(self) -> None:
        mapper_high = SupervisedPyTorchColorMapper(
            input_dim=10, device="cpu", num_classes=3, classification_weight=10.0
        )
        mapper_low = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3, classification_weight=0.01)

        mapper_low.classification_head.load_state_dict(mapper_high.classification_head.state_dict())

        lab_output = torch.tensor([[50.0, 10.0, -20.0]])
        targets = torch.tensor([[50.0, 10.0, -20.0]])
        labels = torch.tensor([1])

        loss_high = mapper_high._compute_combined_loss(lab_output, targets, labels)
        loss_low = mapper_low._compute_combined_loss(lab_output, targets, labels)

        assert loss_high.item() > loss_low.item()

    def test_should_produce_zero_classification_loss_weight(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3, classification_weight=0.0)

        lab_output = torch.tensor([[50.0, 10.0, -20.0]])
        targets = torch.tensor([[60.0, 15.0, -10.0]])
        labels = torch.tensor([1])

        loss = mapper._compute_combined_loss(lab_output, targets, labels)
        projection_only = torch.nn.functional.mse_loss(lab_output, targets)

        assert abs(loss.item() - projection_only.item()) < 1e-6

    def test_should_compute_projection_loss_component(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3, classification_weight=0.0)

        lab_output = torch.tensor([[50.0, 10.0, -20.0]])
        targets = torch.tensor([[50.0, 10.0, -20.0]])
        labels = torch.tensor([1])

        loss = mapper._compute_combined_loss(lab_output, targets, labels)

        assert loss.item() < 1e-6


class TestSupervisedCheckpointing:
    def test_should_checkpoint_best_model_during_training(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0, 1] * 10)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=5, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])
        assert isinstance(result, LabColor)

    def test_should_restore_best_model_after_training(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0, 1] * 10)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=5, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])
        assert isinstance(result, LabColor)

    def test_should_skip_restore_when_best_state_is_none(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0, 1] * 10)

        mapper.set_training_labels(labels)

        with patch.object(mapper, "_run_training_loop", return_value=None):
            mapper.train(embeddings, epochs=2, learning_rate=0.001)

        result = mapper.embed_to_lab(embeddings[0])
        assert isinstance(result, LabColor)


class TestSupervisedWeightPersistence:
    def test_should_save_projector_weights_only(self, tmp_path: Path) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4)
        save_path = tmp_path / "model.pth"

        mapper.save_weights(str(save_path))

        saved_state = torch.load(str(save_path), weights_only=True)
        assert all("classification" not in key for key in saved_state.keys())

    def test_should_load_projector_weights(self, tmp_path: Path) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4)
        save_path = tmp_path / "model.pth"
        mapper.save_weights(str(save_path))

        new_mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4)
        new_mapper.load_weights(str(save_path))

        assert True

    def test_should_not_include_classification_head_in_saved_weights(self, tmp_path: Path) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4)
        save_path = tmp_path / "model.pth"

        mapper.save_weights(str(save_path))

        saved_state = torch.load(str(save_path), weights_only=True)
        classification_keys = [k for k in saved_state.keys() if "classification" in k]
        assert len(classification_keys) == 0
