import logging
from typing import List

import numpy as np
import pytest
import torch
from pathlib import Path
from scipy.stats import spearmanr
from unittest.mock import patch

from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import (
    PyTorchColorMapper,
    LabProjectorNetwork,
    offdiagonal_entries,
)
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.shared.lab_utils import delta_e


def _structure_loss_on(mapper: PyTorchColorMapper, embeddings: np.ndarray) -> float:
    mapper.network.eval()
    with torch.no_grad():
        return mapper._structure_loss(torch.tensor(embeddings, dtype=torch.float32)).item()


def _projected_colors(mapper: PyTorchColorMapper, embeddings: np.ndarray) -> List[LabColor]:
    mapper.network.eval()
    with torch.no_grad():
        raw_lab = mapper.network(torch.tensor(embeddings, dtype=torch.float32)).cpu().numpy()
    return [
        LabColor(
            l=float(np.clip(row[0], 0.0, 100.0)),
            a=float(np.clip(row[1], -127.0, 127.0)),
            b=float(np.clip(row[2], -127.0, 127.0)),
        )
        for row in raw_lab
    ]


def _embedding_color_rank_correlation(mapper: PyTorchColorMapper, embeddings: np.ndarray) -> float:
    colors = _projected_colors(mapper, embeddings)
    unit_embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
    cosine_similarities = []
    color_distances = []
    for first in range(len(embeddings)):
        for second in range(first + 1, len(embeddings)):
            cosine_similarities.append(float(unit_embeddings[first] @ unit_embeddings[second]))
            color_distances.append(delta_e(colors[first], colors[second]))
    return float(spearmanr(cosine_similarities, color_distances).statistic)


class TestLabProjectorNetwork:
    def test_should_initialize_network(self) -> None:
        network = LabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4, dropout_rate=0.2)

        assert isinstance(network, torch.nn.Module)
        assert network.network is not None

    def test_should_forward_pass(self) -> None:
        network = LabProjectorNetwork(input_dim=10, hidden_dim_1=8, hidden_dim_2=4)
        input_tensor = torch.randn(2, 10)

        output = network.forward(input_tensor)

        assert output.shape == (2, 3)
        assert torch.all(output[:, 0] >= 0) and torch.all(output[:, 0] <= 100)
        assert torch.all(output[:, 1] >= -127.5) and torch.all(output[:, 1] <= 127.5)
        assert torch.all(output[:, 2] >= -127.5) and torch.all(output[:, 2] <= 127.5)


class TestPyTorchColorMapper:
    def test_should_initialize_mapper(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, hidden_dim_1=8, hidden_dim_2=4, dropout_rate=0.2)

        assert mapper.network is not None
        assert isinstance(mapper.device, torch.device)

    def test_should_embed_to_lab(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        embedding = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32)

        result = mapper.embed_to_lab(embedding)

        assert isinstance(result, LabColor)
        assert 0 <= result.l <= 100
        assert -128 <= result.a <= 127
        assert -128 <= result.b <= 127

    def test_should_embed_batch_to_lab(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ],
            dtype=np.float32,
        )

        results = mapper.embed_batch_to_lab(embeddings)

        assert len(results) == 2
        assert all(isinstance(color, LabColor) for color in results)

    def test_should_reduce_structure_loss_when_trained_on_seeded_batch(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu", seed=0)
        embeddings = np.random.randn(50, 10).astype(np.float32)
        loss_before = _structure_loss_on(mapper, embeddings)

        mapper.train(embeddings, epochs=100, learning_rate=0.01)

        loss_after = _structure_loss_on(mapper, embeddings)
        assert loss_after < loss_before, f"training did not reduce structure loss: {loss_after} !< {loss_before}"

    def test_should_save_weights(self, tmp_path: Path) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        save_path = tmp_path / "model.pth"

        mapper.save_weights(str(save_path))

        assert save_path.exists()

    def test_should_reproduce_saved_outputs_when_weights_are_reloaded(self, tmp_path: Path) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu", seed=1)
        embeddings = np.random.randn(4, 10).astype(np.float32)
        save_path = tmp_path / "model.pth"
        mapper.save_weights(str(save_path))
        expected = [color.to_tuple() for color in mapper.embed_batch_to_lab(embeddings)]

        reloaded = PyTorchColorMapper(input_dim=10, device="cpu", seed=2)
        reloaded.load_weights(str(save_path))

        actual = [color.to_tuple() for color in reloaded.embed_batch_to_lab(embeddings)]
        assert actual == expected

    def test_should_remove_generate_targets_when_objective_is_structure_preserving(self) -> None:
        assert not hasattr(PyTorchColorMapper, "_generate_targets")

    def test_should_map_near_duplicate_inputs_to_near_identical_lab_when_trained(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.standard_normal(16).astype(np.float32)
        near_duplicate = base + rng.standard_normal(16).astype(np.float32) * 0.01
        others = rng.standard_normal((10, 16)).astype(np.float32)
        embeddings = np.vstack([base, near_duplicate, others]).astype(np.float32)
        mapper = PyTorchColorMapper(input_dim=16, device="cpu")

        mapper.train(embeddings, epochs=300, learning_rate=0.01)

        first = mapper.embed_to_lab(embeddings[0])
        second = mapper.embed_to_lab(embeddings[1])
        assert delta_e(first, second) < 5.0

    def test_should_separate_dissimilar_inputs_when_trained(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.standard_normal(16).astype(np.float32)
        near_duplicate = base + rng.standard_normal(16).astype(np.float32) * 0.01
        others = rng.standard_normal((10, 16)).astype(np.float32)
        embeddings = np.vstack([base, near_duplicate, others]).astype(np.float32)
        mapper = PyTorchColorMapper(input_dim=16, device="cpu")

        mapper.train(embeddings, epochs=300, learning_rate=0.01)

        anchor = mapper.embed_to_lab(embeddings[0])
        dissimilar = mapper.embed_to_lab(embeddings[5])
        assert delta_e(anchor, dissimilar) > 20.0

    def test_should_lower_structure_loss_when_student_matches_teacher_ordering(self) -> None:
        teacher = torch.tensor([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
        student_aligned = teacher.clone()
        student_inverted = torch.tensor([[1.0, 0.1, 0.9], [0.1, 1.0, 0.8], [0.9, 0.8, 1.0]])

        aligned_loss = PyTorchColorMapper._similarity_discrepancy(student_aligned, teacher)
        inverted_loss = PyTorchColorMapper._similarity_discrepancy(student_inverted, teacher)

        assert aligned_loss.item() < inverted_loss.item()

    def test_should_exclude_self_pairs_when_building_offdiagonal_similarity(self) -> None:
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        entries = offdiagonal_entries(matrix)

        expected = torch.tensor([2.0, 3.0, 4.0, 6.0, 7.0, 8.0])
        assert torch.equal(torch.sort(entries).values, expected)

    def test_should_produce_finite_structure_loss_when_batch_is_typical(self) -> None:
        torch.manual_seed(0)
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        batch = torch.randn(8, 10)

        loss = mapper._structure_loss(batch)

        assert torch.isfinite(loss) and loss.item() >= 0.0

    def test_should_return_zero_structure_loss_when_batch_has_single_sample(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        batch = torch.randn(1, 10)

        loss = mapper._structure_loss(batch)

        assert loss.item() == 0.0

    def test_should_use_cpu_when_cuda_not_available(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            mapper = PyTorchColorMapper(device="cuda")

        assert mapper.device.type == "cpu"

    def test_should_produce_valid_color_when_trained_on_small_batch(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(5, 10).astype(np.float32)

        mapper.train(embeddings, epochs=1, learning_rate=0.001)

        assert isinstance(mapper.embed_to_lab(embeddings[0]), LabColor)

    def test_should_log_loss_when_epoch_count_reaches_ten(self, caplog: pytest.LogCaptureFixture) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(20, 10).astype(np.float32)

        with caplog.at_level(logging.INFO):
            mapper.train(embeddings, epochs=10, learning_rate=0.001)

        assert "Epoch [10/10]" in caplog.text

    def test_should_correlate_embedding_similarity_with_color_proximity_when_trained(self) -> None:
        rng = np.random.default_rng(7)
        base = rng.standard_normal(16).astype(np.float32)
        near_duplicate = base + rng.standard_normal(16).astype(np.float32) * 0.01
        others = rng.standard_normal((10, 16)).astype(np.float32)
        embeddings = np.vstack([base, near_duplicate, others]).astype(np.float32)
        mapper = PyTorchColorMapper(input_dim=16, device="cpu", seed=0)

        mapper.train(embeddings, epochs=300, learning_rate=0.01)

        correlation = _embedding_color_rank_correlation(mapper, embeddings)
        assert abs(correlation) >= 0.5, f"structure preservation too weak: |rho|={abs(correlation)} < 0.5"

    def test_should_reproduce_lab_output_when_same_seed(self) -> None:
        embedding = np.arange(10, dtype=np.float32)

        first = PyTorchColorMapper(input_dim=10, device="cpu", seed=123).embed_to_lab(embedding)
        second = PyTorchColorMapper(input_dim=10, device="cpu", seed=123).embed_to_lab(embedding)

        assert first.to_tuple() == second.to_tuple()

    def test_should_reproduce_batch_lab_outputs_when_same_mapper_maps_twice(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu", seed=123)
        batch = np.random.default_rng(0).standard_normal((3, 10)).astype(np.float32)

        first = [color.to_tuple() for color in mapper.embed_batch_to_lab(batch)]
        second = [color.to_tuple() for color in mapper.embed_batch_to_lab(batch)]

        assert first == second

    def test_should_capture_one_checkpoint_per_epoch(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(12, 10).astype(np.float32)

        mapper.train(embeddings, epochs=3, learning_rate=0.001)

        assert len(mapper.epoch_checkpoints()) == 3

    def test_should_restore_checkpoint_weights_deterministically(self) -> None:
        mapper = PyTorchColorMapper(input_dim=10, device="cpu", seed=5)
        embeddings = np.random.randn(12, 10).astype(np.float32)
        mapper.train(embeddings, epochs=2, learning_rate=0.05)
        embedding = np.arange(10, dtype=np.float32)

        mapper.restore_checkpoint(mapper.epoch_checkpoints()[0])
        first = mapper.embed_to_lab(embedding)
        mapper.restore_checkpoint(mapper.epoch_checkpoints()[0])
        second = mapper.embed_to_lab(embedding)

        assert first.to_tuple() == second.to_tuple()
