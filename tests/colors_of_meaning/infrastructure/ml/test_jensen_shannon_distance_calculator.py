import time
from typing import List

import numpy as np
import pytest

from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import JensenShannonDistanceCalculator
from colors_of_meaning.domain.model.colored_document import ColoredDocument

BENCHMARK_BINS = 4096
BENCHMARK_QUERIES = 60
BENCHMARK_REFERENCES = 400
MINIMUM_BENCHMARK_SPEEDUP = 20.0


def _documents(histograms) -> List[ColoredDocument]:
    return [ColoredDocument(histogram=np.asarray(row, dtype=np.float64)) for row in histograms]


def _per_pair_matrix(
    calculator: JensenShannonDistanceCalculator,
    queries: List[ColoredDocument],
    references: List[ColoredDocument],
) -> np.ndarray:
    return np.array([[calculator.compute_distance(query, reference) for reference in references] for query in queries])


def _one_hot(bin_index: int, num_bins: int = 8) -> np.ndarray:
    histogram = np.zeros(num_bins, dtype=np.float64)
    histogram[bin_index] = 1.0
    return histogram


def _normalised_rows(seed: int, count: int, num_bins: int = 8, density: float = 1.0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    rows = generator.random((count, num_bins))
    if density < 1.0:
        rows = np.where(generator.random((count, num_bins)) < density, rows, 0.0)
        rows[:, 0] = 1.0
    return rows / rows.sum(axis=1, keepdims=True)


def _batch_matches_per_pair(calculator: JensenShannonDistanceCalculator, query_rows, reference_rows) -> bool:
    queries, references = _documents(query_rows), _documents(reference_rows)
    matrix = calculator.compute_distance_matrix(queries, references)
    return bool(np.allclose(matrix, _per_pair_matrix(calculator, queries, references), rtol=1e-9, atol=1e-12))


def _one_hot_corpus(count: int, seed: int) -> List[ColoredDocument]:
    generator = np.random.default_rng(seed)
    bins = generator.integers(0, BENCHMARK_BINS, count)
    return _documents([_one_hot(int(index), BENCHMARK_BINS) for index in bins])


def _elapsed_seconds(action) -> float:
    started_at = time.perf_counter()
    action()
    return time.perf_counter() - started_at


class TestJensenShannonDistanceCalculator:
    def test_should_compute_distance_between_documents(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_should_return_metric_name(self) -> None:
        calculator = JensenShannonDistanceCalculator()

        name = calculator.metric_name()

        assert name == "jensen_shannon"

    def test_should_raise_error_when_bins_mismatch(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.33, 0.33, 0.34], dtype=np.float64))

        with pytest.raises(ValueError, match="Documents must have the same number of bins"):
            calculator.compute_distance(doc1, doc2)

    def test_should_use_smoothing_epsilon(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-6)
        doc1 = ColoredDocument(histogram=np.array([1.0, 0.0], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.0, 1.0], dtype=np.float64))

        distance = calculator.compute_distance(doc1, doc2)

        assert isinstance(distance, float)
        assert distance > 0

    def test_should_return_zero_distance_for_identical_documents(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc1 = ColoredDocument(histogram=histogram.copy())
        doc2 = ColoredDocument(histogram=histogram.copy())

        distance = calculator.compute_distance(doc1, doc2)

        assert distance < 1e-10


class TestComputeDistanceMatrix:
    def test_should_match_the_per_pair_distance_when_documents_are_one_hot(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)

        assert _batch_matches_per_pair(
            calculator, [_one_hot(index) for index in (0, 3, 7)], [_one_hot(index) for index in (3, 5, 7, 0)]
        )

    def test_should_match_the_per_pair_distance_when_documents_are_dense(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)

        assert _batch_matches_per_pair(calculator, _normalised_rows(1, 5), _normalised_rows(2, 4))

    def test_should_match_the_per_pair_distance_when_supports_partially_overlap(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)

        assert _batch_matches_per_pair(
            calculator, _normalised_rows(3, 5, density=0.4), _normalised_rows(4, 4, density=0.4)
        )

    def test_should_match_the_per_pair_distance_when_supports_are_disjoint(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        queries = np.array([[0.6, 0.4, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]], dtype=np.float64)
        references = np.array([[0.0, 0.0, 0.3, 0.7], [0.0, 0.0, 0.9, 0.1]], dtype=np.float64)

        assert _batch_matches_per_pair(calculator, queries, references)

    def test_should_match_the_per_pair_distance_when_total_mass_differs_between_documents(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)

        assert _batch_matches_per_pair(
            calculator, _normalised_rows(5, 4) * (1.0 + 4e-7), _normalised_rows(6, 3) * (1.0 - 4e-7)
        )

    def test_should_match_the_per_pair_distance_when_smoothing_is_disabled(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=0.0)

        assert _batch_matches_per_pair(
            calculator, _normalised_rows(7, 4, density=0.4), _normalised_rows(8, 3, density=0.4)
        )

    def test_should_match_the_per_pair_distance_for_arbitrary_random_histograms(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        densities = [1.0, 0.6, 0.25, 0.1]

        assert all(
            _batch_matches_per_pair(
                calculator,
                _normalised_rows(seed, 4, num_bins=32, density=density),
                _normalised_rows(seed + 100, 5, num_bins=32, density=density),
            )
            for seed, density in enumerate(densities)
        )

    def test_should_return_exactly_zero_when_a_document_is_compared_with_itself(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        documents = _documents(np.vstack([_one_hot(2), _normalised_rows(9, 1)[0]]))

        matrix = calculator.compute_distance_matrix(documents, documents)

        assert np.array_equal(matrix.diagonal(), np.zeros(len(documents)))

    def test_should_give_every_disjoint_one_hot_pair_the_identical_distance(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        queries = _documents([_one_hot(index, BENCHMARK_BINS) for index in (0, 11, 4095)])
        references = _documents([_one_hot(index, BENCHMARK_BINS) for index in (7, 512, 4094)])

        matrix = calculator.compute_distance_matrix(queries, references)

        assert len(np.unique(matrix)) == 1

    def test_should_return_the_same_distance_when_one_hot_documents_are_relabelled(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        original = _documents([_one_hot(11, BENCHMARK_BINS)]), _documents([_one_hot(2043, BENCHMARK_BINS)])
        relabelled = _documents([_one_hot(1500, BENCHMARK_BINS)]), _documents([_one_hot(37, BENCHMARK_BINS)])

        assert calculator.compute_distance_matrix(*original) == calculator.compute_distance_matrix(*relabelled)

    def test_should_barely_move_when_arbitrary_documents_are_relabelled_by_the_same_permutation(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        rows = _normalised_rows(21, 2, num_bins=BENCHMARK_BINS, density=0.01)
        permutation = np.random.default_rng(3).permutation(BENCHMARK_BINS)

        original = calculator.compute_distance_matrix(_documents([rows[0]]), _documents([rows[1]]))
        relabelled = calculator.compute_distance_matrix(
            _documents([rows[0][permutation]]), _documents([rows[1][permutation]])
        )

        assert abs(original[0, 0] - relabelled[0, 0]) <= np.spacing(original[0, 0])

    def test_should_raise_error_when_bins_mismatch_across_the_batch(self) -> None:
        calculator = JensenShannonDistanceCalculator()
        queries = _documents([[0.5, 0.5]])
        references = _documents([[0.33, 0.33, 0.34]])

        with pytest.raises(ValueError, match="Documents must have the same number of bins"):
            calculator.compute_distance_matrix(queries, references)

    def test_should_return_an_empty_matrix_when_there_are_no_queries(self) -> None:
        calculator = JensenShannonDistanceCalculator()

        assert calculator.compute_distance_matrix([], _documents([_one_hot(1)])).shape == (0, 1)

    def test_should_return_an_empty_matrix_when_there_are_no_references(self) -> None:
        calculator = JensenShannonDistanceCalculator()

        assert calculator.compute_distance_matrix(_documents([_one_hot(1)]), []).shape == (1, 0)


@pytest.mark.benchmark
class TestBatchedDistanceBenchmark:
    def test_should_beat_the_per_pair_baseline_by_a_wide_margin_on_one_hot_documents(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        queries = _one_hot_corpus(BENCHMARK_QUERIES, seed=11)
        references = _one_hot_corpus(BENCHMARK_REFERENCES, seed=12)

        baseline = _elapsed_seconds(lambda: _per_pair_matrix(calculator, queries, references))
        batched = _elapsed_seconds(lambda: calculator.compute_distance_matrix(queries, references))

        assert baseline / batched > MINIMUM_BENCHMARK_SPEEDUP

    def test_should_agree_with_the_per_pair_baseline_on_the_benchmark_workload(self) -> None:
        calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
        queries = _one_hot_corpus(BENCHMARK_QUERIES, seed=11)
        references = _one_hot_corpus(BENCHMARK_REFERENCES, seed=12)

        matrix = calculator.compute_distance_matrix(queries, references)

        assert np.allclose(matrix, _per_pair_matrix(calculator, queries, references), rtol=1e-9, atol=1e-12)
