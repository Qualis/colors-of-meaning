from typing import Callable, Dict, List, Optional

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.authorship_report import (
    AuthorshipCorpus,
    AuthorshipHeldOut,
    AuthorshipReport,
    AuthorshipScalingPoint,
)
from colors_of_meaning.domain.repository.authorship_scaling_repository import (
    AuthorshipScalingRepository,
)

EvaluateUseCaseFactory = Callable[[str], EvaluateUseCase]
ScalingSweep = Callable[[], List[AuthorshipScalingPoint]]


class AuthorshipReportUseCase:
    def __init__(
        self,
        evaluate_use_case_factory: EvaluateUseCaseFactory,
        scaling_repository: AuthorshipScalingRepository,
    ) -> None:
        self.evaluate_use_case_factory = evaluate_use_case_factory
        self.scaling_repository = scaling_repository

    def execute(
        self,
        corpus: AuthorshipCorpus,
        methods: List[str],
        bits_per_token: Dict[str, Optional[float]],
        max_samples: Optional[int],
        seed: int,
        refresh_scaling: bool,
        scaling_sweep: ScalingSweep,
    ) -> AuthorshipReport:
        held_out = [self._evaluate(method, bits_per_token.get(method), max_samples, seed) for method in methods]
        scaling = self._resolve_scaling(refresh_scaling, scaling_sweep)
        chance = 1.0 / corpus.author_count()
        return AuthorshipReport(corpus=corpus, held_out=held_out, chance=chance, scaling=scaling)

    def _evaluate(
        self,
        method: str,
        bits_per_token: Optional[float],
        max_samples: Optional[int],
        seed: int,
    ) -> AuthorshipHeldOut:
        result = self.evaluate_use_case_factory(method).execute(
            bits_per_token=bits_per_token, max_samples=max_samples, seed=seed
        )
        return AuthorshipHeldOut(
            method=method,
            bits_per_token=bits_per_token,
            accuracy=result.accuracy,
            macro_f1=result.macro_f1,
        )

    def _resolve_scaling(self, refresh_scaling: bool, scaling_sweep: ScalingSweep) -> List[AuthorshipScalingPoint]:
        if refresh_scaling:
            points = scaling_sweep()
            self.scaling_repository.save(points)
            return points
        return self.scaling_repository.load()
