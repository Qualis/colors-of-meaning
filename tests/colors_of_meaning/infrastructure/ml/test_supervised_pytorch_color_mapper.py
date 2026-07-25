import logging

import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch

import pytest

from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.domain.model.lab_color import LabColor


def _combined_loss_on(mapper: SupervisedPyTorchColorMapper, embeddings: np.ndarray, labels: np.ndarray) -> float:
    label_tensor = torch.tensor(labels, dtype=torch.long)
    mapper.network.eval()
    with torch.no_grad():
        lab_output = mapper.network(torch.tensor(embeddings, dtype=torch.float32))
        return mapper._compute_combined_loss(lab_output, label_tensor).item()


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

    def test_should_initialize_with_contrastive_margin(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", contrastive_margin=2.5)

        assert mapper.contrastive_margin == 2.5

    def test_should_remove_generate_targets_from_supervised_mapper_when_refactored(self) -> None:
        assert not hasattr(SupervisedPyTorchColorMapper, "_generate_targets")


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
    def test_should_reduce_combined_loss_when_trained_on_seeded_batch(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3, seed=0)
        embeddings = np.random.randn(30, 10).astype(np.float32)
        labels = np.array([0, 1, 2] * 10)
        mapper.set_training_labels(labels)
        loss_before = _combined_loss_on(mapper, embeddings, labels)

        mapper.train(embeddings, epochs=50, learning_rate=0.01)

        loss_after = _combined_loss_on(mapper, embeddings, labels)
        assert loss_after < loss_before, f"training did not reduce combined loss: {loss_after} !< {loss_before}"

    def test_should_train_supervised_with_contrastive_objective_when_labels_set(self) -> None:
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

    def test_should_log_loss_when_epoch_count_reaches_ten(self, caplog: pytest.LogCaptureFixture) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(10, 10).astype(np.float32)
        labels = np.array([0, 1] * 5)

        mapper.set_training_labels(labels)
        with caplog.at_level(logging.INFO):
            mapper.train(embeddings, epochs=10, learning_rate=0.001)

        assert "Epoch [10/10]" in caplog.text

    def test_should_reproduce_lab_output_when_same_seed(self) -> None:
        embedding = np.arange(10, dtype=np.float32)

        first = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", seed=77).embed_to_lab(embedding)
        second = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", seed=77).embed_to_lab(embedding)

        assert first.to_tuple() == second.to_tuple()

    def test_should_capture_one_checkpoint_per_epoch(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0, 1] * 10)

        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=4, learning_rate=0.001)

        assert len(mapper.epoch_checkpoints()) == 4

    def test_should_restore_checkpoint_weights_deterministically(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2, seed=9)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0, 1] * 10)
        mapper.set_training_labels(labels)
        mapper.train(embeddings, epochs=3, learning_rate=0.01)
        embedding = np.arange(10, dtype=np.float32)

        mapper.restore_checkpoint(mapper.epoch_checkpoints()[0])
        first = mapper.embed_to_lab(embedding)
        mapper.restore_checkpoint(mapper.epoch_checkpoints()[0])
        second = mapper.embed_to_lab(embedding)

        assert first.to_tuple() == second.to_tuple()


class TestSupervisedLoss:
    def test_should_compute_combined_loss(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3)

        lab_output = torch.tensor([[50.0, 10.0, -20.0], [20.0, -5.0, 15.0]], requires_grad=True)
        labels = torch.tensor([1, 0])

        loss = mapper._compute_combined_loss(lab_output, labels)

        assert loss.item() > 0

    def test_should_weight_classification_loss_by_config(self) -> None:
        mapper_high = SupervisedPyTorchColorMapper(
            input_dim=10, device="cpu", num_classes=3, classification_weight=10.0
        )
        mapper_low = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3, classification_weight=0.01)

        mapper_low.classification_head.load_state_dict(mapper_high.classification_head.state_dict())

        lab_output = torch.tensor([[50.0, 10.0, -20.0], [20.0, -5.0, 15.0]])
        labels = torch.tensor([1, 0])

        loss_high = mapper_high._compute_combined_loss(lab_output, labels)
        loss_low = mapper_low._compute_combined_loss(lab_output, labels)

        assert loss_high.item() > loss_low.item()

    def test_should_produce_zero_classification_loss_weight(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3, classification_weight=0.0)

        lab_output = torch.tensor([[50.0, 10.0, -20.0], [20.0, -5.0, 15.0]])
        labels = torch.tensor([1, 0])

        loss = mapper._compute_combined_loss(lab_output, labels)
        contrastive_only = mapper._contrastive_loss(lab_output, labels)

        assert abs(loss.item() - contrastive_only.item()) < 1e-6

    def test_should_penalise_same_class_separation_when_contrastive(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        labels = torch.tensor([0, 0])
        colocated = torch.tensor([[50.0, 10.0, -20.0], [50.0, 10.0, -20.0]])
        separated = torch.tensor([[0.0, -127.5, -127.5], [100.0, 127.5, 127.5]])

        colocated_loss = mapper._contrastive_loss(colocated, labels)
        separated_loss = mapper._contrastive_loss(separated, labels)

        assert separated_loss.item() > colocated_loss.item()

    def test_should_repel_different_class_within_margin_when_contrastive(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        labels = torch.tensor([0, 1])
        colocated = torch.tensor([[50.0, 10.0, -20.0], [50.0, 10.0, -20.0]])

        loss = mapper._contrastive_loss(colocated, labels)

        assert loss.item() > 0.0

    def test_should_return_zero_contrastive_loss_when_single_sample(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        labels = torch.tensor([0])
        lab_output = torch.tensor([[50.0, 10.0, -20.0]])

        loss = mapper._contrastive_loss(lab_output, labels)

        assert loss.item() == 0.0

    def test_should_keep_classification_term_influential_when_losses_combined(self) -> None:
        torch.manual_seed(0)
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3)
        lab_output = mapper.network(torch.randn(12, 10))
        labels = torch.tensor([0, 1, 2] * 4)

        structure = mapper._contrastive_loss(lab_output, labels).item()
        classification = (
            mapper.classification_weight
            * torch.nn.functional.cross_entropy(mapper.classification_head(lab_output), labels).item()
        )

        assert max(structure, classification) < 100.0 * min(structure, classification)

    def test_should_scale_normalise_lab_into_unit_range_when_normalising(self) -> None:
        lab_output = torch.tensor([[100.0, 127.5, -127.5], [0.0, -127.5, 127.5]])

        normalised = SupervisedPyTorchColorMapper._normalise_lab(lab_output)

        assert (
            torch.all(normalised[:, 0] >= 0.0)
            and torch.all(normalised[:, 0] <= 1.0)
            and torch.all(normalised[:, 1:].abs() <= 1.0)
        )


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

    def test_should_keep_previous_best_when_loss_not_improved(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=2)
        previous_best = {"marker": 1}

        result = mapper._checkpoint_if_improved(avg_loss=5.0, best_loss=1.0, best_state=previous_best)

        assert result == (1.0, previous_best)

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

        saved_state = torch.load(str(save_path), weights_only=True)  # nosemgrep
        assert all("classification" not in key for key in saved_state.keys())

    def test_should_reproduce_projector_outputs_when_weights_are_reloaded(self, tmp_path: Path) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4, seed=1)
        embeddings = np.random.randn(4, 10).astype(np.float32)
        save_path = tmp_path / "model.pth"
        mapper.save_weights(str(save_path))
        expected = [color.to_tuple() for color in mapper.embed_batch_to_lab(embeddings)]

        reloaded = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4, seed=2)
        reloaded.load_weights(str(save_path))

        actual = [color.to_tuple() for color in reloaded.embed_batch_to_lab(embeddings)]
        assert actual == expected

    def test_should_not_include_classification_head_in_saved_weights(self, tmp_path: Path) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=4)
        save_path = tmp_path / "model.pth"

        mapper.save_weights(str(save_path))

        saved_state = torch.load(str(save_path), weights_only=True)  # nosemgrep
        classification_keys = [k for k in saved_state.keys() if "classification" in k]
        assert len(classification_keys) == 0
