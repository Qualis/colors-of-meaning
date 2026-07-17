import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import (
    RetrievalEvaluateUseCase,
)
from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationCell:
    dataset: str
    method: str
    distance: str
    budget: Optional[int]
    requires_fidelity: bool
    bits_per_token: Optional[float] = None
    supports_retrieval: bool = False


@dataclass(frozen=True)
class EvaluatedCell:
    cell: EvaluationCell
    result: EvaluationResult
    seconds: float
    retrieval: Optional[EvaluationResult] = None
    retrieval_skip_reason: Optional[str] = None


class UnfaithfulProxyError(RuntimeError):
    def __init__(self, fidelity: DistanceFidelity) -> None:
        super().__init__(
            "Refusing to report scaled proxy results: distance proxy is unfaithful "
            f"({'; '.join(_unfaithful_reasons(fidelity))})"
        )
        self.fidelity = fidelity


def _unfaithful_reasons(fidelity: DistanceFidelity) -> List[str]:
    reasons = []
    if fidelity.spearman < fidelity.threshold_spearman:
        reasons.append(f"spearman={fidelity.spearman:.4f} < {fidelity.threshold_spearman}")
    if fidelity.accuracy_delta > fidelity.max_accuracy_delta:
        reasons.append(f"accuracy_delta={fidelity.accuracy_delta:.4f} > {fidelity.max_accuracy_delta}")
    return reasons


EvaluateUseCaseFactory = Callable[[EvaluationCell], EvaluateUseCase]
RetrievalEvaluateUseCaseFactory = Callable[[EvaluationCell], RetrievalEvaluateUseCase]


class EvaluationSuiteUseCase:
    def __init__(
        self,
        evaluate_use_case_factory: EvaluateUseCaseFactory,
        seed: Optional[int] = None,
        clock: Callable[[], float] = time.perf_counter,
        retrieval_use_case_factory: Optional[RetrievalEvaluateUseCaseFactory] = None,
    ) -> None:
        self._evaluate_use_case_factory = evaluate_use_case_factory
        self._seed = seed
        self._clock = clock
        self._retrieval_use_case_factory = retrieval_use_case_factory

    def execute(self, cells: Sequence[EvaluationCell], fidelity: DistanceFidelity) -> List[EvaluatedCell]:
        self._reject_unfaithful_scaled_cells(cells, fidelity)
        return [self._evaluate_cell(cell) for cell in cells]

    def _reject_unfaithful_scaled_cells(self, cells: Sequence[EvaluationCell], fidelity: DistanceFidelity) -> None:
        if not fidelity.is_faithful and any(cell.requires_fidelity for cell in cells):
            raise UnfaithfulProxyError(fidelity)

    def _evaluate_cell(self, cell: EvaluationCell) -> EvaluatedCell:
        evaluate_use_case = self._evaluate_use_case_factory(cell)
        started_at = self._clock()
        result = evaluate_use_case.execute(bits_per_token=cell.bits_per_token, max_samples=cell.budget, seed=self._seed)
        seconds = self._clock() - started_at
        retrieval, skip_reason = self._maybe_retrieval(cell)
        evaluated = EvaluatedCell(
            cell=cell, result=result, seconds=seconds, retrieval=retrieval, retrieval_skip_reason=skip_reason
        )
        self._log_cell(evaluated)
        return evaluated

    def _maybe_retrieval(self, cell: EvaluationCell) -> Tuple[Optional[EvaluationResult], Optional[str]]:
        if self._retrieval_use_case_factory is None:
            return None, None
        if not cell.supports_retrieval:
            return None, f"{cell.method} is classification-only; retrieval skipped"
        retrieval_use_case = self._retrieval_use_case_factory(cell)
        evaluation = retrieval_use_case.execute(
            bits_per_token=cell.bits_per_token, max_samples=cell.budget, seed=self._seed
        )
        return evaluation.result, None

    def _log_cell(self, evaluated: EvaluatedCell) -> None:
        logger.info("Completed evaluation suite cell", extra=_cell_log_payload(evaluated))


def _cell_log_payload(evaluated: EvaluatedCell) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "correlation_id": str(uuid.uuid4()),
        "dataset": evaluated.cell.dataset,
        "method": evaluated.cell.method,
        "distance": evaluated.cell.distance,
        "budget": evaluated.cell.budget,
        "bits_per_token": evaluated.cell.bits_per_token,
        "accuracy": evaluated.result.accuracy,
        "macro_f1": evaluated.result.macro_f1,
        "seconds": evaluated.seconds,
    }
    if evaluated.retrieval is not None:
        payload["mrr"] = evaluated.retrieval.mrr
        payload["recall_at_k"] = evaluated.retrieval.recall_at_k
    if evaluated.retrieval_skip_reason is not None:
        payload["retrieval_skip_reason"] = evaluated.retrieval_skip_reason
    return payload
