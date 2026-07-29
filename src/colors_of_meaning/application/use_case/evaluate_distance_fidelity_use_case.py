import logging
import uuid
from typing import List, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.domain.service.rank_correlation_calculator import RankCorrelationCalculator

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_SPEARMAN = 0.95
DEFAULT_MAX_ACCURACY_DELTA = 1.0

DocumentPair = Tuple[int, int]


class EvaluateDistanceFidelityUseCase:
    def __init__(
        self,
        proxy_calculator: DistanceCalculator,
        exact_calculator: DistanceCalculator,
        rank_correlation_calculator: RankCorrelationCalculator,
    ) -> None:
        self._proxy_calculator = proxy_calculator
        self._exact_calculator = exact_calculator
        self._rank_correlation_calculator = rank_correlation_calculator

    def execute(
        self,
        documents: Sequence[ColoredDocument],
        pair_count: int,
        seed: int,
        accuracy_delta: float,
        threshold_spearman: float = DEFAULT_THRESHOLD_SPEARMAN,
        max_accuracy_delta: float = DEFAULT_MAX_ACCURACY_DELTA,
    ) -> DistanceFidelity:
        self._reject_insufficient_documents(documents)
        pairs = self._draw_document_pairs(len(documents), pair_count, seed)
        proxy_distances = self._distance_vector(self._proxy_calculator, documents, pairs)
        exact_distances = self._distance_vector(self._exact_calculator, documents, pairs)
        spearman = self._rank_correlation_calculator.correlate(proxy_distances, exact_distances)
        fidelity = DistanceFidelity(
            spearman=spearman,
            accuracy_delta=accuracy_delta,
            pair_count=len(pairs),
            threshold_spearman=threshold_spearman,
            max_accuracy_delta=max_accuracy_delta,
        )
        self._log_fidelity(fidelity)
        return fidelity

    def _distance_vector(
        self,
        calculator: DistanceCalculator,
        documents: Sequence[ColoredDocument],
        pairs: List[DocumentPair],
    ) -> npt.NDArray[np.float64]:
        return np.array(
            [calculator.compute_distance(documents[first], documents[second]) for first, second in pairs],
            dtype=np.float64,
        )

    @staticmethod
    def _draw_document_pairs(document_count: int, pair_count: int, seed: int) -> List[DocumentPair]:
        generator = np.random.default_rng(seed)
        first_indices = generator.integers(0, document_count, size=pair_count)
        distinct_offsets = generator.integers(1, document_count, size=pair_count)
        second_indices = (first_indices + distinct_offsets) % document_count
        return list(zip(first_indices.tolist(), second_indices.tolist(), strict=True))

    @staticmethod
    def _reject_insufficient_documents(documents: Sequence[ColoredDocument]) -> None:
        if len(documents) < 2:
            raise ValueError("At least two documents are required to sample distance pairs")

    def _log_fidelity(self, fidelity: DistanceFidelity) -> None:
        logger.info(
            "Measured distance proxy fidelity against exact transport",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "proxy": self._proxy_calculator.metric_name(),
                "spearman": fidelity.spearman,
                "accuracy_delta": fidelity.accuracy_delta,
                "pair_count": fidelity.pair_count,
                "is_faithful": fidelity.is_faithful,
            },
        )
