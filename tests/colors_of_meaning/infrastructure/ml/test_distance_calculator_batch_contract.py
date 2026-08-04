from typing import List

import numpy as np
import pytest

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.ml.cosine_histogram_distance_calculator import (
    CosineHistogramDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)

CODEBOOK = ColorCodebook.create_uniform_grid(bins_per_dimension=2)


def _calculators() -> List[DistanceCalculator]:
    return [
        JensenShannonDistanceCalculator(smoothing_epsilon=1e-8),
        CosineHistogramDistanceCalculator(smoothing_epsilon=1e-8),
        WassersteinDistanceCalculator(codebook=CODEBOOK),
        SlicedWassersteinDistanceCalculator(codebook=CODEBOOK, n_projections=8, seed=1),
    ]


def _documents(seed: int, count: int) -> List[ColoredDocument]:
    generator = np.random.default_rng(seed)
    histograms = generator.random((count, CODEBOOK.num_bins))
    return [ColoredDocument(histogram=row / row.sum()) for row in histograms]


def _per_pair_matrix(
    calculator: DistanceCalculator, queries: List[ColoredDocument], references: List[ColoredDocument]
) -> np.ndarray:
    return np.array([[calculator.compute_distance(query, reference) for reference in references] for query in queries])


@pytest.mark.parametrize("calculator", _calculators(), ids=lambda item: item.metric_name())
class TestBatchMatchesPerPairForEveryCalculator:
    def test_should_reproduce_the_per_pair_distances_when_computing_a_matrix(
        self, calculator: DistanceCalculator
    ) -> None:
        queries, references = _documents(seed=1, count=4), _documents(seed=2, count=5)

        matrix = calculator.compute_distance_matrix(queries, references)

        assert np.allclose(matrix, _per_pair_matrix(calculator, queries, references), rtol=1e-9, atol=1e-12)
