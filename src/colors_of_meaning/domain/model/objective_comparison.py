import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

DEFAULT_ADOPTION_THRESHOLD_SIGMA = 2.0
DEFAULT_MAX_ACCURACY_DROP = 0.01


@dataclass(frozen=True)
class ObjectiveArmResult:
    arm: str
    mean_rho: float
    stdev_rho: float
    seeds: int
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    mrr: Optional[float] = None
    recall_at_k: Optional[Dict[int, float]] = None

    def __post_init__(self) -> None:
        self._require_positive_seeds()
        self._require_non_negative_stdev()
        self._require_correlation_range()
        self._require_optional_unit_metrics()

    def strength(self) -> float:
        return abs(self.mean_rho)

    def _require_positive_seeds(self) -> None:
        if self.seeds < 1:
            raise ValueError(f"seeds must be at least 1, got {self.seeds}")

    def _require_non_negative_stdev(self) -> None:
        if self.stdev_rho < 0:
            raise ValueError(f"stdev_rho must be non-negative, got {self.stdev_rho}")

    def _require_correlation_range(self) -> None:
        if not -1.0 <= self.mean_rho <= 1.0:
            raise ValueError(f"mean_rho must be between -1 and 1, got {self.mean_rho}")

    def _require_optional_unit_metrics(self) -> None:
        for name, value in (("accuracy", self.accuracy), ("macro_f1", self.macro_f1), ("mrr", self.mrr)):
            self._require_unit_interval(name, value)
        for k, recall in (self.recall_at_k or {}).items():
            self._require_unit_interval(f"recall@{k}", recall)

    @staticmethod
    def _require_unit_interval(name: str, value: Optional[float]) -> None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")


@dataclass(frozen=True)
class ObjectiveComparison:
    results: Sequence[ObjectiveArmResult]
    baseline_arm: str
    adoption_threshold_sigma: float = DEFAULT_ADOPTION_THRESHOLD_SIGMA
    max_accuracy_drop: float = DEFAULT_MAX_ACCURACY_DROP
    controls: Sequence[ObjectiveArmResult] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self._require_non_negative("adoption_threshold_sigma", self.adoption_threshold_sigma)
        self._require_non_negative("max_accuracy_drop", self.max_accuracy_drop)
        self._require_unique_arms()
        self._require_matching_seed_counts()
        self.baseline()

    def baseline(self) -> ObjectiveArmResult:
        for result in self.results:
            if result.arm == self.baseline_arm:
                return result
        raise ValueError(f"baseline arm {self.baseline_arm} is missing from the results")

    def challengers(self) -> List[ObjectiveArmResult]:
        return [result for result in self.results if result.arm != self.baseline_arm]

    def pooled_stdev(self, challenger: ObjectiveArmResult) -> float:
        baseline = self.baseline()
        return math.sqrt((baseline.stdev_rho**2 + challenger.stdev_rho**2) / 2.0)

    def margin_in_sigma(self, challenger: ObjectiveArmResult) -> float:
        difference = challenger.strength() - self.baseline().strength()
        pooled = self.pooled_stdev(challenger)
        if pooled == 0.0:
            return 0.0 if difference == 0.0 else math.copysign(math.inf, difference)
        return difference / pooled

    def has_measurable_spread(self, challenger: ObjectiveArmResult) -> bool:
        return self.pooled_stdev(challenger) > 0.0

    def clears_adoption_rule(self, challenger: ObjectiveArmResult) -> bool:
        return self._clears_correlation_margin(challenger) and self._clears_accuracy_guard(challenger)

    def adopted_arm(self) -> str:
        qualifying = [challenger for challenger in self.challengers() if self.clears_adoption_rule(challenger)]
        if not qualifying:
            return self.baseline_arm
        return max(qualifying, key=lambda candidate: candidate.strength()).arm

    def _clears_correlation_margin(self, challenger: ObjectiveArmResult) -> bool:
        if not self.has_measurable_spread(challenger):
            return False
        return self.margin_in_sigma(challenger) > self.adoption_threshold_sigma

    def _clears_accuracy_guard(self, challenger: ObjectiveArmResult) -> bool:
        baseline_accuracy = self.baseline().accuracy
        if baseline_accuracy is None or challenger.accuracy is None:
            return False
        return challenger.accuracy >= baseline_accuracy - self.max_accuracy_drop

    def _scored(self) -> List[ObjectiveArmResult]:
        return list(self.results) + list(self.controls)

    def _require_matching_seed_counts(self) -> None:
        seed_counts = {result.seeds for result in self._scored()}
        if len(seed_counts) > 1:
            raise ValueError(f"every arm must be scored over the same seed count, got {sorted(seed_counts)}")

    def _require_unique_arms(self) -> None:
        arms = [result.arm for result in self._scored()]
        if len(arms) != len(set(arms)):
            raise ValueError(f"arm names must be unique, got {arms}")

    @staticmethod
    def _require_non_negative(name: str, value: float) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
