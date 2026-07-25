import logging

import numpy as np
import pytest
import torch
from pathlib import Path
from typing import Dict
from unittest.mock import patch

from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.concreteness_lexicon import ConcretenessLexicon


class _StubLexicon(ConcretenessLexicon):
    def __init__(self, scores: Dict[str, float]) -> None:
        self._scores = scores

    def score(self, text: str) -> float:
        return self._scores.get(text, 3.0)


def _circular_distance(first: float, second: float) -> float:
    delta = abs(first - second) % (2.0 * np.pi)
    return min(delta, 2.0 * np.pi - delta)


def _structured_loss_on(mapper: StructuredPyTorchColorMapper, embeddings: np.ndarray) -> float:
    hue_targets, lightness_targets, chroma_targets = mapper._prepare_targets(embeddings)
    mapper.network.eval()
    with torch.no_grad():
        pred_lightness, pred_hue, pred_chroma = mapper.network.forward_structured(
            torch.tensor(embeddings, dtype=torch.float32)
        )
        loss = mapper._compute_loss(
            pred_lightness, pred_hue, pred_chroma, lightness_targets, hue_targets, chroma_targets
        )
    return loss.item()


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

    def test_should_reduce_structured_loss_when_trained_on_seeded_batch(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3, seed=0)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        loss_before = _structured_loss_on(mapper, embeddings)

        mapper.train(embeddings, epochs=30, learning_rate=0.01)

        loss_after = _structured_loss_on(mapper, embeddings)
        assert loss_after < loss_before, f"training did not reduce structured loss: {loss_after} !< {loss_before}"

    def test_should_save_weights(self, tmp_path: Path) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        save_path = tmp_path / "model.pth"

        mapper.save_weights(str(save_path))

        assert save_path.exists()

    def test_should_reproduce_saved_outputs_when_weights_are_reloaded(self, tmp_path: Path) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", seed=1)
        embeddings = np.random.randn(4, 10).astype(np.float32)
        save_path = tmp_path / "model.pth"
        mapper.save_weights(str(save_path))
        expected = [color.to_tuple() for color in mapper.embed_batch_to_lab(embeddings)]

        reloaded = StructuredPyTorchColorMapper(input_dim=10, device="cpu", seed=2)
        reloaded.load_weights(str(save_path))

        actual = [color.to_tuple() for color in reloaded.embed_batch_to_lab(embeddings)]
        assert actual == expected

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

    def test_should_produce_valid_color_when_trained_on_small_batch(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=2)
        embeddings = np.random.randn(5, 10).astype(np.float32)

        mapper.train(embeddings, epochs=1, learning_rate=0.001)

        assert isinstance(mapper.embed_to_lab(embeddings[0]), LabColor)

    def test_should_log_loss_when_epoch_count_reaches_ten(self, caplog: pytest.LogCaptureFixture) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        with caplog.at_level(logging.INFO):
            mapper.train(embeddings, epochs=10, learning_rate=0.001)

        assert "Epoch [10/10]" in caplog.text

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

    def test_should_reproduce_lab_output_when_same_seed(self) -> None:
        embedding = np.arange(10, dtype=np.float32)

        first = StructuredPyTorchColorMapper(input_dim=10, device="cpu", seed=321).embed_to_lab(embedding)
        second = StructuredPyTorchColorMapper(input_dim=10, device="cpu", seed=321).embed_to_lab(embedding)

        assert first.to_tuple() == second.to_tuple()

    def test_should_capture_one_checkpoint_per_epoch(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3)
        embeddings = np.random.randn(20, 10).astype(np.float32)

        mapper.train(embeddings, epochs=4, learning_rate=0.001)

        assert len(mapper.epoch_checkpoints()) == 4

    def test_should_restore_checkpoint_weights_deterministically(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=3, seed=8)
        embeddings = np.random.randn(20, 10).astype(np.float32)
        mapper.train(embeddings, epochs=3, learning_rate=0.01)
        embedding = np.arange(10, dtype=np.float32)

        mapper.restore_checkpoint(mapper.epoch_checkpoints()[0])
        first = mapper.embed_to_lab(embedding)
        mapper.restore_checkpoint(mapper.epoch_checkpoints()[0])
        second = mapper.embed_to_lab(embedding)

        assert first.to_tuple() == second.to_tuple()


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


class TestStructuredCheckpointing:
    def test_should_keep_previous_best_when_loss_not_improved(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        previous_best = {"marker": 1}

        result = mapper._checkpoint_if_improved(avg_loss=5.0, best_loss=1.0, best_state=previous_best)

        assert result == (1.0, previous_best)


class TestStructuredSideInformation:
    def test_should_store_sentiment_scores_when_set(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")

        mapper.set_sentiment_scores(np.array([0.0, 1.0]))

        assert mapper._sentiment_scores is not None

    def test_should_store_training_texts_when_set(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")

        mapper.set_training_texts(["hello", "world"])

        assert mapper._training_texts == ["hello", "world"]

    def test_should_default_lexicon_when_not_injected(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")

        assert mapper._concreteness_lexicon is not None

    def test_should_raise_when_sentiment_length_mismatches_embeddings(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=2)
        embeddings = np.random.randn(4, 10).astype(np.float32)
        mapper.set_sentiment_scores(np.array([0.0, 1.0]))

        with pytest.raises(ValueError):
            mapper.train(embeddings, epochs=1, learning_rate=0.001)

    def test_should_raise_when_text_length_mismatches_embeddings(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=2)
        embeddings = np.random.randn(4, 10).astype(np.float32)
        mapper.set_training_texts(["a", "b"])

        with pytest.raises(ValueError):
            mapper.train(embeddings, epochs=1, learning_rate=0.001)

    def test_should_train_when_sentiment_and_texts_match_embeddings(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", num_clusters=2)
        embeddings = np.random.randn(4, 10).astype(np.float32)
        mapper.set_sentiment_scores(np.array([0.0, 1.0, 0.0, 1.0]))
        mapper.set_training_texts(["a", "b", "c", "d"])

        mapper.train(embeddings, epochs=1, learning_rate=0.001)

        assert isinstance(mapper.embed_to_lab(embeddings[0]), LabColor)


class TestSentimentDrivenLightness:
    def test_should_increase_lightness_when_sentiment_is_more_positive(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.zeros((4, 10), dtype=np.float32)

        mapper.set_sentiment_scores(np.array([0.1, 0.2, 0.1, 0.2]))
        low = mapper._derive_lightness_targets(embeddings).mean().item()
        mapper.set_sentiment_scores(np.array([0.8, 0.9, 0.8, 0.9]))
        high = mapper._derive_lightness_targets(embeddings).mean().item()

        assert high > low

    def test_should_map_negative_and_positive_labels_to_distinct_bands_when_sentiment_is_binary(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.zeros((4, 10), dtype=np.float32)
        mapper.set_sentiment_scores(np.array([0, 1, 0, 1]))

        targets = mapper._derive_lightness_targets(embeddings).squeeze(1)

        assert torch.all(targets[[0, 2]] < 50.0) and torch.all(targets[[1, 3]] > 50.0)
        assert torch.all(targets >= 0) and torch.all(targets <= 100.0)

    def test_should_return_neutral_lightness_when_no_sentiment_is_set(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.random.randn(5, 10).astype(np.float32)

        targets = mapper._derive_lightness_targets(embeddings)

        assert torch.all(targets == 50.0)

    def test_should_keep_lightness_in_valid_range_when_sentiment_is_extreme(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.zeros((3, 10), dtype=np.float32)
        mapper.set_sentiment_scores(np.array([-5.0, 0.5, 9.0]))

        targets = mapper._derive_lightness_targets(embeddings)

        assert torch.all(targets >= 0) and torch.all(targets <= 100.0)

    def test_should_not_use_embedding_mean_for_lightness_when_sentiment_is_set(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        embeddings = np.vstack([np.full(10, -3.0, dtype=np.float32), np.full(10, 7.0, dtype=np.float32)])
        mapper.set_sentiment_scores(np.array([0.5, 0.5]))

        targets = mapper._derive_lightness_targets(embeddings).squeeze(1)

        assert torch.allclose(targets[0], targets[1])


class TestConcretenessDrivenChroma:
    def test_should_give_higher_chroma_to_concrete_text_than_abstract_text_when_scored(self) -> None:
        lexicon = _StubLexicon({"concrete": 5.0, "abstract": 1.0})
        mapper = StructuredPyTorchColorMapper(
            input_dim=10, device="cpu", max_chroma=128.0, concreteness_lexicon=lexicon
        )
        embeddings = np.zeros((2, 10), dtype=np.float32)
        mapper.set_training_texts(["concrete", "abstract"])

        targets = mapper._derive_chroma_targets(embeddings).squeeze(1)

        assert targets[0] > targets[1]

    def test_should_return_neutral_chroma_when_text_is_out_of_vocabulary(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", max_chroma=128.0)
        embeddings = np.zeros((1, 10), dtype=np.float32)
        mapper.set_training_texts(["zzqqx vvwwk"])

        targets = mapper._derive_chroma_targets(embeddings)

        assert torch.all(targets == 64.0)

    def test_should_return_neutral_chroma_when_no_text_is_set(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu", max_chroma=128.0)
        embeddings = np.random.randn(5, 10).astype(np.float32)

        targets = mapper._derive_chroma_targets(embeddings)

        assert torch.all(targets == 64.0)

    def test_should_keep_chroma_in_valid_range_when_concreteness_is_extreme(self) -> None:
        lexicon = _StubLexicon({"a": 99.0, "b": -7.0})
        mapper = StructuredPyTorchColorMapper(
            input_dim=10, device="cpu", max_chroma=128.0, concreteness_lexicon=lexicon
        )
        embeddings = np.zeros((2, 10), dtype=np.float32)
        mapper.set_training_texts(["a", "b"])

        targets = mapper._derive_chroma_targets(embeddings)

        assert torch.all(targets >= 0) and torch.all(targets <= 128.0)


class TestSemanticHueOrdering:
    def test_should_assign_adjacent_hues_to_nearest_centroid_clusters_when_ordering_by_centroid(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        centers = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

        ranks = mapper._order_clusters_by_centroid(centers)
        angles = 2.0 * np.pi * ranks / len(centers)

        assert _circular_distance(angles[0], angles[1]) < _circular_distance(angles[0], angles[2])

    def test_should_order_clusters_deterministically_when_seed_is_fixed(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        centers = np.random.randn(5, 4).astype(np.float32)

        first = mapper._order_clusters_by_centroid(centers)
        second = mapper._order_clusters_by_centroid(centers)

        assert np.array_equal(first, second)

    def test_should_assign_single_rank_when_one_cluster(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=10, device="cpu")
        centers = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        ranks = mapper._order_clusters_by_centroid(centers)

        assert np.array_equal(ranks, np.zeros(1, dtype=np.int64))
