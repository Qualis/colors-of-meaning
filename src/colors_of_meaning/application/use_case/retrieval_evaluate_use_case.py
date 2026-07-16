import logging
import uuid
from typing import Dict, List, Optional

from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.retrieval_evaluation import RetrievalEvaluation
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.metrics_calculator import MetricsCalculator
from colors_of_meaning.domain.service.retriever import Retriever

logger = logging.getLogger(__name__)


class RetrievalEvaluateUseCase:
    def __init__(
        self,
        retriever: Retriever,
        metrics_calculator: MetricsCalculator,
        dataset_repository: DatasetRepository,
        k_values: List[int],
    ) -> None:
        if not k_values:
            raise ValueError("k_values must not be empty")
        self.retriever = retriever
        self.metrics_calculator = metrics_calculator
        self.dataset_repository = dataset_repository
        self.k_values = k_values

    def execute(
        self,
        bits_per_token: Optional[float] = None,
        max_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> RetrievalEvaluation:
        train_samples = self.dataset_repository.get_samples(split="train", max_samples=max_samples, seed=seed)
        test_samples = self.dataset_repository.get_samples(split="test", max_samples=max_samples, seed=seed)

        self.retriever.fit(train_samples)

        search_depth = max(self.k_values)
        search_results = [self.retriever.search(query, search_depth) for query in test_samples]

        result = self.metrics_calculator.calculate_retrieval_metrics(
            queries=test_samples,
            search_results=search_results,
            k_values=self.k_values,
            bits_per_token=bits_per_token,
        )
        ndcg_at_k = self.metrics_calculator.calculate_ndcg_at_k(test_samples, search_results, self.k_values)

        self._log_outcome(test_samples, result, ndcg_at_k)

        return RetrievalEvaluation(result=result, ndcg_at_k=ndcg_at_k)

    def _log_outcome(
        self,
        test_samples: List[EvaluationSample],
        result: EvaluationResult,
        ndcg_at_k: Dict[int, float],
    ) -> None:
        logger.info(
            "Retrieval evaluation complete",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "retriever": type(self.retriever).__name__,
                "dataset": type(self.dataset_repository).__name__,
                "k_values": self.k_values,
                "query_count": len(test_samples),
                "mrr": result.mrr,
                "recall_at_k": result.recall_at_k,
                "ndcg_at_k": ndcg_at_k,
            },
        )
