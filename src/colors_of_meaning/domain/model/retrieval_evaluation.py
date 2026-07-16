from dataclasses import dataclass
from typing import Dict

from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


@dataclass(frozen=True)
class RetrievalEvaluation:
    result: EvaluationResult
    ndcg_at_k: Dict[int, float]

    def __post_init__(self) -> None:
        for k, value in self.ndcg_at_k.items():
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"ndcg@{k} must be between 0 and 1, got {value}")
