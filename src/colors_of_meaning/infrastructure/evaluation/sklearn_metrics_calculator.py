from typing import List, Dict, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, ndcg_score  # type: ignore

from colors_of_meaning.domain.service.metrics_calculator import MetricsCalculator
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class SklearnMetricsCalculator(MetricsCalculator):
    def calculate_classification_metrics(
        self,
        y_true: List[int],
        y_pred: List[int],
        bits_per_token: Optional[float] = None,
    ) -> EvaluationResult:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)

        return EvaluationResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            recall_at_k={},
            mrr=0.0,
            bits_per_token=bits_per_token,
        )

    def calculate_retrieval_metrics(
        self,
        queries: List[EvaluationSample],
        search_results: List[List[Tuple[EvaluationSample, float]]],
        k_values: List[int],
        bits_per_token: Optional[float] = None,
    ) -> EvaluationResult:
        all_recalls_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
        all_reciprocal_ranks: List[float] = []

        for query, results in zip(queries, search_results):
            self._process_query_results(query, results, k_values, all_recalls_at_k, all_reciprocal_ranks)

        avg_recall_at_k = self._compute_average_recall(all_recalls_at_k)
        mrr = self._compute_mean_reciprocal_rank(all_reciprocal_ranks)

        return EvaluationResult(
            accuracy=0.0,
            macro_f1=0.0,
            recall_at_k=avg_recall_at_k,
            mrr=mrr,
            bits_per_token=bits_per_token,
        )

    def _process_query_results(
        self,
        query: EvaluationSample,
        results: List[Tuple[EvaluationSample, float]],
        k_values: List[int],
        all_recalls_at_k: Dict[int, List[float]],
        all_reciprocal_ranks: List[float],
    ) -> None:
        query_label = query.label
        retrieved_labels = [sample.label for sample, _ in results]
        rr = self.compute_reciprocal_rank(query_label, retrieved_labels)
        all_reciprocal_ranks.append(rr)
        for k in k_values:
            recall = self.compute_recall_at_k([query_label], retrieved_labels[:k])
            all_recalls_at_k[k].append(recall)

    def _compute_average_recall(self, all_recalls_at_k: Dict[int, List[float]]) -> Dict[int, float]:
        return {k: sum(recalls) / len(recalls) if recalls else 0.0 for k, recalls in all_recalls_at_k.items()}

    def _compute_mean_reciprocal_rank(self, all_reciprocal_ranks: List[float]) -> float:
        return sum(all_reciprocal_ranks) / len(all_reciprocal_ranks) if all_reciprocal_ranks else 0.0

    def compute_recall_at_k(
        self,
        relevant_labels: List[int],
        retrieved_labels: List[int],
    ) -> float:
        if not relevant_labels:
            return 0.0

        relevant_set = set(relevant_labels)
        retrieved_set = set(retrieved_labels)
        num_relevant_retrieved = len(relevant_set.intersection(retrieved_set))

        return num_relevant_retrieved / len(relevant_set)

    def compute_reciprocal_rank(
        self,
        relevant_label: int,
        retrieved_labels: List[int],
    ) -> float:
        for rank, label in enumerate(retrieved_labels, start=1):
            if label == relevant_label:
                return 1.0 / rank
        return 0.0

    def calculate_ndcg_at_k(
        self,
        queries: List[EvaluationSample],
        search_results: List[List[Tuple[EvaluationSample, float]]],
        k_values: List[int],
    ) -> Dict[int, float]:
        return {k: self._mean_ndcg_at_k(queries, search_results, k) for k in k_values}

    def _mean_ndcg_at_k(
        self,
        queries: List[EvaluationSample],
        search_results: List[List[Tuple[EvaluationSample, float]]],
        k: int,
    ) -> float:
        scores = [
            self._single_query_ndcg(query, results, k)
            for query, results in zip(queries, search_results)
            if len(results) >= 2
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _single_query_ndcg(
        self,
        query: EvaluationSample,
        results: List[Tuple[EvaluationSample, float]],
        k: int,
    ) -> float:
        relevance = [1.0 if sample.label == query.label else 0.0 for sample, _ in results]
        ranking_scores = [-distance for _, distance in results]
        return float(ndcg_score([relevance], [ranking_scores], k=k))
