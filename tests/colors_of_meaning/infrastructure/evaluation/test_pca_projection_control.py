import numpy as np
import pytest
from scipy.stats import spearmanr

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.infrastructure.evaluation.pca_projection_control import (
    LAB_AXIS_RANGES,
    PcaProjectionControl,
    rescale_preserving_ranks,
)

TRAIN_SAMPLES = 40
EVAL_SAMPLES = 12
EMBEDDING_DIM = 9


def _embeddings(seed: int, count: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, EMBEDDING_DIM))


def _fitted_control() -> PcaProjectionControl:
    return PcaProjectionControl(seed=13).fit(_embeddings(1, TRAIN_SAMPLES))


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    upper_triangle = np.triu_indices(len(coordinates), k=1)
    return np.sqrt((differences**2).sum(axis=2))[upper_triangle]


def _lab_coordinates(colors: list) -> np.ndarray:
    return np.array([color.to_tuple() for color in colors], dtype=np.float64)


class TestPcaProjectionControl:
    def test_should_produce_one_lab_color_per_evaluation_embedding(self) -> None:
        colors = _fitted_control().transform(_embeddings(2, EVAL_SAMPLES))

        assert len(colors) == EVAL_SAMPLES

    def test_should_place_the_training_projection_inside_the_lab_ranges(self) -> None:
        control = PcaProjectionControl(seed=13)
        train = _embeddings(1, TRAIN_SAMPLES)
        control.fit(train)

        coordinates = _lab_coordinates(control.transform(train))

        assert all(
            np.all(coordinates[:, axis] >= low) and np.all(coordinates[:, axis] <= high)
            for axis, (low, high) in enumerate(LAB_AXIS_RANGES)
        )

    def test_should_span_the_full_lab_axis_range_when_transforming_the_training_split(self) -> None:
        control = PcaProjectionControl(seed=13)
        train = _embeddings(1, TRAIN_SAMPLES)
        control.fit(train)

        lightness = _lab_coordinates(control.transform(train))[:, 0]

        assert lightness.min() == pytest.approx(0.0, abs=1e-6) and lightness.max() == pytest.approx(100.0, abs=1e-6)

    def test_should_quantize_against_the_shared_codebook_without_raising(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)

        bins = codebook.quantize_batch(_fitted_control().transform(_embeddings(2, EVAL_SAMPLES)))

        assert len(bins) == EVAL_SAMPLES

    def test_should_reject_a_transform_before_the_control_is_fitted(self) -> None:
        with pytest.raises(ValueError, match="must be fitted before transforming"):
            PcaProjectionControl(seed=13).transform(_embeddings(2, EVAL_SAMPLES))

    def test_should_reproduce_the_projection_when_fitted_twice_with_the_same_seed(self) -> None:
        train = _embeddings(1, TRAIN_SAMPLES)
        evaluation = _embeddings(2, EVAL_SAMPLES)

        first = _lab_coordinates(PcaProjectionControl(seed=13).fit(train).transform(evaluation))
        second = _lab_coordinates(PcaProjectionControl(seed=13).fit(train).transform(evaluation))

        assert np.array_equal(first, second)

    def test_should_seed_the_randomized_solver_so_a_wide_fit_stays_reproducible(self) -> None:
        assert PcaProjectionControl(seed=13)._pca.random_state == 13

    def test_should_rescale_unseen_embeddings_with_the_training_statistics(self) -> None:
        control = PcaProjectionControl(seed=13)
        control.fit(_embeddings(1, TRAIN_SAMPLES) * 4.0)

        lightness = _lab_coordinates(control.transform(_embeddings(1, TRAIN_SAMPLES)))[:, 0]

        assert lightness.max() - lightness.min() < 90.0

    def test_should_log_the_explained_variance_when_fitted(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            _fitted_control()

        assert any("explained_variance_ratio" in vars(record) for record in caplog.records)


class TestRescalePreservingRanks:
    def test_should_place_every_coordinate_inside_the_lab_ranges(self) -> None:
        raw = np.random.default_rng(5).normal(size=(20, 3)) * 900.0 + 400.0

        coordinates = _lab_coordinates(rescale_preserving_ranks(raw))

        assert all(
            np.all(coordinates[:, axis] >= low - 1e-9) and np.all(coordinates[:, axis] <= high + 1e-9)
            for axis, (low, high) in enumerate(LAB_AXIS_RANGES)
        )

    def test_should_preserve_the_rank_order_of_every_pairwise_distance(self) -> None:
        raw = np.random.default_rng(6).normal(size=(24, 3)) * 900.0 + 400.0

        rescaled = _lab_coordinates(rescale_preserving_ranks(raw))

        correlation = spearmanr(_pairwise_distances(raw), _pairwise_distances(rescaled)).statistic
        assert correlation == pytest.approx(1.0, abs=1e-12)

    def test_should_preserve_rank_order_when_one_axis_is_all_but_degenerate(self) -> None:
        raw = np.random.default_rng(9).normal(size=(12, 3)) * np.array([1e-12, 1.5e-12, 0.0])

        rescaled = _lab_coordinates(rescale_preserving_ranks(raw))

        correlation = spearmanr(_pairwise_distances(raw), _pairwise_distances(rescaled)).statistic
        assert correlation == pytest.approx(1.0, abs=1e-12)

    def test_should_refuse_to_rescale_an_axis_too_small_to_keep_its_rank_order(self) -> None:
        raw = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1e-320], [2.0, 0.5, 0.0]])

        with pytest.raises(ValueError, match="too small to rescale"):
            rescale_preserving_ranks(raw)

    def test_should_return_the_axis_centres_when_every_coordinate_is_identical(self) -> None:
        raw = np.full((4, 3), 7.5)

        coordinates = _lab_coordinates(rescale_preserving_ranks(raw))

        assert np.allclose(coordinates, np.array([50.0, 0.0, 0.0]))
