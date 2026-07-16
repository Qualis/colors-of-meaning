from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class MetricsCalculator(ABC):
    @abstractmethod
    def calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        bits_per_token: Optional[float] = None,
    ) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def calculate_retrieval_metrics(
        self,
        queries: List[EvaluationSample],
        search_results: List[List[Tuple[EvaluationSample, float]]],
        k_values: List[int],
        bits_per_token: Optional[float] = None,
    ) -> EvaluationResult:
        raise NotImplementedError

    @abstractmethod
    def compute_recall_at_k(
        self,
        relevant_labels: List[int],
        retrieved_labels: List[int],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def compute_reciprocal_rank(
        self,
        relevant_label: int,
        retrieved_labels: List[int],
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def calculate_ndcg_at_k(
        self,
        queries: List[EvaluationSample],
        search_results: List[List[Tuple[EvaluationSample, float]]],
        k_values: List[int],
    ) -> Dict[int, float]:
        raise NotImplementedError
