import logging
import math
import statistics
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy.typing as npt

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import (
    RetrievalEvaluateUseCase,
)
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.objective_comparison import (
    DEFAULT_ADOPTION_THRESHOLD_SIGMA,
    DEFAULT_MAX_ACCURACY_DROP,
    ObjectiveArmResult,
    ObjectiveComparison,
)
from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)

logger = logging.getLogger(__name__)

NO_SPREAD = 0.0


@dataclass(frozen=True, eq=False)
class ArmRequest:
    arm: str
    seed: int
    train_embeddings: npt.NDArray
    eval_embeddings: npt.NDArray


@dataclass(frozen=True, eq=False)
class DownstreamMetrics:
    accuracy: float
    macro_f1: float
    mrr: Optional[float] = None
    recall_at_k: Optional[Dict[int, float]] = None


@dataclass(frozen=True, eq=False)
class _Sweep:
    train_embeddings: npt.NDArray
    eval_embeddings: npt.NDArray
    seeds: Sequence[int]
    downstream_seeds: Sequence[int]
    downstream_arms: Sequence[str]
    correlation_id: str

    def request(self, arm: str, seed: int) -> ArmRequest:
        return ArmRequest(
            arm=arm, seed=seed, train_embeddings=self.train_embeddings, eval_embeddings=self.eval_embeddings
        )


def _resolved(downstream_seeds: Optional[Sequence[int]], seeds: Sequence[int]) -> List[int]:
    return list(seeds if downstream_seeds is None else downstream_seeds)


def _resolved_correlation_id(correlation_id: Optional[str]) -> str:
    return correlation_id if correlation_id is not None else str(uuid.uuid4())


def _reject_undefined_correlation(arm: str, seed: int, correlation: float) -> None:
    if math.isnan(correlation):
        raise ValueError(
            f"arm {arm} at seed {seed} produced an undefined structure correlation; "
            "its Lab output is constant, so the projector collapsed"
        )
    if not -1.0 <= correlation <= 1.0:
        raise ValueError(f"arm {arm} at seed {seed} produced a correlation outside [-1, 1], got {correlation}")


def _downstream_from(
    classifications: Sequence[EvaluationResult], measured: Sequence[EvaluationResult]
) -> DownstreamMetrics:
    return DownstreamMetrics(
        accuracy=statistics.fmean([result.accuracy for result in classifications]),
        macro_f1=statistics.fmean([result.macro_f1 for result in classifications]),
        mrr=_mean_reciprocal_rank(measured),
        recall_at_k=_mean_recall_at_k(measured),
    )


LabColorsFactory = Callable[[ArmRequest], List[LabColor]]
EvaluateUseCaseFactory = Callable[[ArmRequest], EvaluateUseCase]
RetrievalUseCaseFactory = Callable[[ArmRequest], RetrievalEvaluateUseCase]


def _recall_depths(results: Sequence[EvaluationResult]) -> List[int]:
    return sorted({depth for result in results for depth in result.recall_at_k})


def _mean_recall_at_k(results: Sequence[EvaluationResult]) -> Optional[Dict[int, float]]:
    if not results:
        return None
    return {
        depth: statistics.fmean([result.recall_at_k[depth] for result in results]) for depth in _recall_depths(results)
    }


def _mean_reciprocal_rank(results: Sequence[EvaluationResult]) -> Optional[float]:
    if not results:
        return None
    return statistics.fmean([result.mrr for result in results])


def _downstream_fields(downstream: Optional[DownstreamMetrics]) -> Dict[str, Any]:
    if downstream is None:
        return {}
    return {
        "accuracy": downstream.accuracy,
        "macro_f1": downstream.macro_f1,
        "mrr": downstream.mrr,
        "recall_at_k": downstream.recall_at_k,
    }


class CompareStructureObjectivesUseCase:
    def __init__(
        self,
        lab_colors_factory: LabColorsFactory,
        structure_preservation_evaluator: StructurePreservationEvaluator,
        evaluate_use_case_factory: Optional[EvaluateUseCaseFactory] = None,
        retrieval_use_case_factory: Optional[RetrievalUseCaseFactory] = None,
        adoption_threshold_sigma: float = DEFAULT_ADOPTION_THRESHOLD_SIGMA,
        max_accuracy_drop: float = DEFAULT_MAX_ACCURACY_DROP,
        downstream_budget: Optional[int] = None,
        downstream_bits_per_token: Optional[float] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.lab_colors_factory = lab_colors_factory
        self.structure_preservation_evaluator = structure_preservation_evaluator
        self.evaluate_use_case_factory = evaluate_use_case_factory
        self.retrieval_use_case_factory = retrieval_use_case_factory
        self.adoption_threshold_sigma = adoption_threshold_sigma
        self.max_accuracy_drop = max_accuracy_drop
        self.downstream_budget = downstream_budget
        self.downstream_bits_per_token = downstream_bits_per_token
        self.clock = clock
        self._scores: Dict[Tuple[str, int], float] = {}
        self._scored_split: Optional[Tuple[int, int]] = None

    def execute(
        self,
        train_embeddings: npt.NDArray,
        eval_embeddings: npt.NDArray,
        arms: Sequence[str],
        seeds: Sequence[int],
        baseline_arm: str,
        controls: Sequence[str] = (),
        downstream_arms: Sequence[str] = (),
        downstream_seeds: Optional[Sequence[int]] = None,
        log_decision: bool = True,
        correlation_id: Optional[str] = None,
    ) -> ObjectiveComparison:
        self._discard_scores_from_another_split(train_embeddings, eval_embeddings)
        sweep = _Sweep(
            train_embeddings=train_embeddings,
            eval_embeddings=eval_embeddings,
            seeds=list(seeds),
            downstream_seeds=_resolved(downstream_seeds, seeds),
            downstream_arms=list(downstream_arms),
            correlation_id=_resolved_correlation_id(correlation_id),
        )
        comparison = ObjectiveComparison(
            results=[self._arm_result(arm, sweep) for arm in arms],
            baseline_arm=baseline_arm,
            adoption_threshold_sigma=self.adoption_threshold_sigma,
            max_accuracy_drop=self.max_accuracy_drop,
            controls=[self._arm_result(control, sweep) for control in controls],
        )
        if log_decision:
            self.log_decision(comparison, sweep.correlation_id)
        return comparison

    def _discard_scores_from_another_split(self, train_embeddings: npt.NDArray, eval_embeddings: npt.NDArray) -> None:
        if self._scored_split != (id(train_embeddings), id(eval_embeddings)):
            self._scores = {}
            self._scored_split = (id(train_embeddings), id(eval_embeddings))

    def _arm_result(self, arm: str, sweep: _Sweep) -> ObjectiveArmResult:
        correlations = [self._score_seed(arm, seed, sweep) for seed in sweep.seeds]
        return ObjectiveArmResult(
            arm=arm,
            mean_rho=statistics.fmean(correlations),
            stdev_rho=self._spread(correlations),
            seeds=len(correlations),
            **_downstream_fields(self._downstream_metrics(arm, sweep)),
        )

    def _score_seed(self, arm: str, seed: int, sweep: _Sweep) -> float:
        if (arm, seed) in self._scores:
            return self._scores[(arm, seed)]
        started_at = self.clock()
        lab_colors = self.lab_colors_factory(sweep.request(arm, seed))
        correlation = self.structure_preservation_evaluator.evaluate(sweep.eval_embeddings, lab_colors)
        _reject_undefined_correlation(arm, seed, correlation)
        self._log_seed(arm, seed, correlation, self.clock() - started_at, sweep.correlation_id)
        self._scores[(arm, seed)] = correlation
        return correlation

    def _downstream_metrics(self, arm: str, sweep: _Sweep) -> Optional[DownstreamMetrics]:
        factory = self.evaluate_use_case_factory
        if factory is None or arm not in sweep.downstream_arms:
            return None
        classifications = [self._classify_seed(factory, arm, seed, sweep) for seed in sweep.downstream_seeds]
        return _downstream_from(classifications, self._measured_retrievals(arm, sweep))

    def _measured_retrievals(self, arm: str, sweep: _Sweep) -> List[EvaluationResult]:
        retrievals = [self._retrieve_seed(arm, seed, sweep) for seed in sweep.downstream_seeds]
        return [retrieval for retrieval in retrievals if retrieval is not None]

    def _classify_seed(self, factory: EvaluateUseCaseFactory, arm: str, seed: int, sweep: _Sweep) -> EvaluationResult:
        result = factory(sweep.request(arm, seed)).execute(
            bits_per_token=self.downstream_bits_per_token, max_samples=self.downstream_budget, seed=seed
        )
        self._log_downstream(arm, seed, result.accuracy, sweep.correlation_id)
        return result

    def _retrieve_seed(self, arm: str, seed: int, sweep: _Sweep) -> Optional[EvaluationResult]:
        if self.retrieval_use_case_factory is None:
            return None
        evaluation = self.retrieval_use_case_factory(sweep.request(arm, seed)).execute(
            bits_per_token=self.downstream_bits_per_token, max_samples=self.downstream_budget, seed=seed
        )
        return evaluation.result

    @staticmethod
    def _spread(correlations: Sequence[float]) -> float:
        if len(correlations) < 2:
            return NO_SPREAD
        return statistics.stdev(correlations)

    @staticmethod
    def _log_seed(arm: str, seed: int, correlation: float, seconds: float, correlation_id: str) -> None:
        logger.info(
            "Scored a structure objective arm",
            extra={"correlation_id": correlation_id, "arm": arm, "seed": seed, "rho": correlation, "seconds": seconds},
        )

    @staticmethod
    def _log_downstream(arm: str, seed: int, accuracy: float, correlation_id: str) -> None:
        logger.info(
            "Evaluated a structure objective arm downstream",
            extra={"correlation_id": correlation_id, "arm": arm, "seed": seed, "accuracy": accuracy},
        )

    @staticmethod
    def log_decision(comparison: ObjectiveComparison, correlation_id: str) -> None:
        logger.info(
            "Applied the pre-registered objective adoption rule",
            extra={
                "correlation_id": correlation_id,
                "adopted_arm": comparison.adopted_arm(),
                "baseline_arm": comparison.baseline_arm,
                "baseline_rho": comparison.baseline().mean_rho,
                "margins_in_sigma": {
                    challenger.arm: comparison.margin_in_sigma(challenger) for challenger in comparison.challengers()
                },
            },
        )
