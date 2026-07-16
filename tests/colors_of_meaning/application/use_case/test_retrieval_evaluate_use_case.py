from typing import Dict, List, Tuple

import pytest
from unittest.mock import Mock

from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import RetrievalEvaluateUseCase
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.service.retriever import Retriever
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)


def _sample(text: str, label: int, split: str) -> EvaluationSample:
    return EvaluationSample(text=text, label=label, split=split)


class _StubRetriever(Retriever):
    def __init__(self, ranking: Dict[str, List[Tuple[EvaluationSample, float]]]) -> None:
        self.ranking = ranking
        self.fitted_with: List[EvaluationSample] = []
        self.search_calls: List[Tuple[EvaluationSample, int]] = []

    def fit(self, samples: List[EvaluationSample]) -> None:
        self.fitted_with = samples

    def search(self, query: EvaluationSample, k: int) -> List[Tuple[EvaluationSample, float]]:
        self.search_calls.append((query, k))
        return self.ranking.get(query.text, [])[:k]


def _dataset_repository(train: List[EvaluationSample], test: List[EvaluationSample]) -> Mock:
    repository = Mock()
    repository.get_samples.side_effect = [train, test]
    return repository


class TestRetrievalEvaluateUseCase:
    def test_should_raise_when_k_values_is_empty(self) -> None:
        with pytest.raises(ValueError, match="k_values must not be empty"):
            RetrievalEvaluateUseCase(Mock(), Mock(), Mock(), [])

    def test_should_fit_retriever_on_train_split(self) -> None:
        train = [_sample("t0", 0, "train")]
        test = [_sample("q0", 0, "test")]
        retriever = _StubRetriever({"q0": [(_sample("t0", 0, "test"), 0.1)]})
        use_case = RetrievalEvaluateUseCase(
            retriever, SklearnMetricsCalculator(), _dataset_repository(train, test), [1]
        )

        use_case.execute()

        assert retriever.fitted_with == train

    def test_should_search_each_query_with_the_deepest_k(self) -> None:
        train = [_sample("t0", 0, "train")]
        test = [_sample("q0", 0, "test")]
        retriever = _StubRetriever({"q0": [(_sample("t0", 0, "test"), 0.1)]})
        use_case = RetrievalEvaluateUseCase(
            retriever, SklearnMetricsCalculator(), _dataset_repository(train, test), [1, 5, 10]
        )

        use_case.execute()

        assert retriever.search_calls[0][1] == 10

    def test_should_report_non_zero_mrr_for_a_same_class_hit(self) -> None:
        train = [_sample("t0", 0, "train")]
        test = [_sample("q0", 0, "test")]
        retriever = _StubRetriever({"q0": [(_sample("t0", 0, "test"), 0.1)]})
        use_case = RetrievalEvaluateUseCase(
            retriever, SklearnMetricsCalculator(), _dataset_repository(train, test), [1]
        )

        evaluation = use_case.execute()

        assert evaluation.result.mrr == 1.0

    def test_should_report_reciprocal_rank_of_a_second_ranked_hit(self) -> None:
        train = [_sample("t0", 0, "train")]
        test = [_sample("q0", 0, "test")]
        retriever = _StubRetriever({"q0": [(_sample("t1", 1, "test"), 0.1), (_sample("t0", 0, "test"), 0.2)]})
        use_case = RetrievalEvaluateUseCase(
            retriever, SklearnMetricsCalculator(), _dataset_repository(train, test), [2]
        )

        evaluation = use_case.execute()

        assert evaluation.result.mrr == 0.5

    def test_should_return_ndcg_for_each_requested_k(self) -> None:
        train = [_sample("t0", 0, "train")]
        test = [_sample("q0", 0, "test")]
        retriever = _StubRetriever({"q0": [(_sample("t0", 0, "test"), 0.1), (_sample("t1", 1, "test"), 0.2)]})
        use_case = RetrievalEvaluateUseCase(
            retriever, SklearnMetricsCalculator(), _dataset_repository(train, test), [2]
        )

        evaluation = use_case.execute()

        assert evaluation.ndcg_at_k[2] == 1.0

    def test_should_forward_seed_and_max_samples_to_the_train_split(self) -> None:
        repository = _dataset_repository([_sample("t0", 0, "train")], [_sample("q0", 0, "test")])
        retriever = _StubRetriever({"q0": [(_sample("t0", 0, "test"), 0.1)]})
        use_case = RetrievalEvaluateUseCase(retriever, SklearnMetricsCalculator(), repository, [1])

        use_case.execute(max_samples=50, seed=7)

        repository.get_samples.assert_any_call(split="train", max_samples=50, seed=7)

    def test_should_pass_bits_per_token_to_the_metrics_calculator(self) -> None:
        metrics_calculator = Mock()
        metrics_calculator.calculate_retrieval_metrics.return_value = EvaluationResult(
            accuracy=0.0, macro_f1=0.0, recall_at_k={1: 1.0}, mrr=1.0, bits_per_token=12.0
        )
        metrics_calculator.calculate_ndcg_at_k.return_value = {1: 1.0}
        repository = _dataset_repository([_sample("t0", 0, "train")], [_sample("q0", 0, "test")])
        use_case = RetrievalEvaluateUseCase(_StubRetriever({}), metrics_calculator, repository, [1])

        use_case.execute(bits_per_token=12.0)

        assert metrics_calculator.calculate_retrieval_metrics.call_args[1]["bits_per_token"] == 12.0
