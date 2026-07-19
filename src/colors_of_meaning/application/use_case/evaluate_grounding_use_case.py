import logging
import uuid
from typing import List

import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.grounding_report import GroundingReport
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator

logger = logging.getLogger(__name__)


class EvaluateGroundingUseCase:
    def __init__(self, distance_calculator: DistanceCalculator) -> None:
        self._distance = distance_calculator

    def execute(self, answer: ColoredDocument, context: ColoredDocument, threshold: float) -> GroundingReport:
        divergence = self._distance.compute_distance(answer, context)
        report = GroundingReport(divergence=divergence, threshold=threshold)
        self._log_verdict(answer, context, report)
        return report

    def execute_against_contexts(
        self, answer: ColoredDocument, contexts: List[ColoredDocument], threshold: float
    ) -> GroundingReport:
        if not contexts:
            raise ValueError("contexts must not be empty")
        return self.execute(answer, self._mix_palette(contexts), threshold)

    @staticmethod
    def _mix_palette(histograms: List[ColoredDocument]) -> ColoredDocument:
        mixture = np.asarray(np.mean([histogram.histogram for histogram in histograms], axis=0), dtype=np.float64)
        return ColoredDocument(histogram=mixture, document_id="context_palette")

    def _log_verdict(self, answer: ColoredDocument, context: ColoredDocument, report: GroundingReport) -> None:
        logger.info(
            "Evaluated answer grounding",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "answer_id": answer.document_id,
                "context_id": context.document_id,
                "metric": self._distance.metric_name(),
                "divergence": report.divergence,
                "threshold": report.threshold,
                "is_grounded": report.is_grounded,
            },
        )
