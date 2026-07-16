import pytest

from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


def _sample(text: str, label: int) -> EvaluationSample:
    return EvaluationSample(text=text, label=label, split="test")


class TestSklearnMetricsCalculator:
    @pytest.fixture
    def calculator(self) -> SklearnMetricsCalculator:
        return SklearnMetricsCalculator()

    def test_should_calculate_perfect_accuracy(self, calculator: SklearnMetricsCalculator) -> None:
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 1]

        result = calculator.calculate_classification_metrics(y_true, y_pred)

        assert result.accuracy == 1.0

    def test_should_calculate_zero_accuracy(self, calculator: SklearnMetricsCalculator) -> None:
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]

        result = calculator.calculate_classification_metrics(y_true, y_pred)

        assert result.accuracy == 0.0

    def test_should_calculate_macro_f1_score(self, calculator: SklearnMetricsCalculator) -> None:
        y_true = [0, 1, 2, 0, 1]
        y_pred = [0, 1, 2, 0, 1]

        result = calculator.calculate_classification_metrics(y_true, y_pred)

        assert result.macro_f1 == 1.0

    def test_should_include_bits_per_token_when_provided(self, calculator: SklearnMetricsCalculator) -> None:
        y_true = [0, 1]
        y_pred = [0, 1]

        result = calculator.calculate_classification_metrics(y_true, y_pred, bits_per_token=12.0)

        assert result.bits_per_token == 12.0

    def test_should_have_none_bits_per_token_when_not_provided(self, calculator: SklearnMetricsCalculator) -> None:
        y_true = [0, 1]
        y_pred = [0, 1]

        result = calculator.calculate_classification_metrics(y_true, y_pred)

        assert result.bits_per_token is None

    def test_should_compute_recall_at_k_for_single_relevant_label(self, calculator: SklearnMetricsCalculator) -> None:
        relevant_labels = [5]
        retrieved_labels = [3, 5, 7, 9]

        recall = calculator.compute_recall_at_k(relevant_labels, retrieved_labels)

        assert recall == 1.0

    def test_should_compute_zero_recall_when_relevant_not_in_retrieved(
        self, calculator: SklearnMetricsCalculator
    ) -> None:
        relevant_labels = [5]
        retrieved_labels = [1, 2, 3, 4]

        recall = calculator.compute_recall_at_k(relevant_labels, retrieved_labels)

        assert recall == 0.0

    def test_should_return_zero_recall_for_empty_relevant_labels(self, calculator: SklearnMetricsCalculator) -> None:
        relevant_labels = []
        retrieved_labels = [1, 2, 3]

        recall = calculator.compute_recall_at_k(relevant_labels, retrieved_labels)

        assert recall == 0.0

    def test_should_compute_reciprocal_rank_for_first_position(self, calculator: SklearnMetricsCalculator) -> None:
        relevant_label = 5
        retrieved_labels = [5, 3, 7, 9]

        rr = calculator.compute_reciprocal_rank(relevant_label, retrieved_labels)

        assert rr == 1.0

    def test_should_compute_reciprocal_rank_for_second_position(self, calculator: SklearnMetricsCalculator) -> None:
        relevant_label = 5
        retrieved_labels = [3, 5, 7, 9]

        rr = calculator.compute_reciprocal_rank(relevant_label, retrieved_labels)

        assert rr == 0.5

    def test_should_compute_reciprocal_rank_for_third_position(self, calculator: SklearnMetricsCalculator) -> None:
        relevant_label = 5
        retrieved_labels = [3, 7, 5, 9]

        rr = calculator.compute_reciprocal_rank(relevant_label, retrieved_labels)

        assert rr == pytest.approx(1.0 / 3.0)

    def test_should_return_zero_reciprocal_rank_when_not_found(self, calculator: SklearnMetricsCalculator) -> None:
        relevant_label = 5
        retrieved_labels = [1, 2, 3, 4]

        rr = calculator.compute_reciprocal_rank(relevant_label, retrieved_labels)

        assert rr == 0.0

    def test_should_calculate_retrieval_metrics_with_queries(self, calculator: SklearnMetricsCalculator) -> None:
        queries = [
            EvaluationSample(text="query1", label=0, split="test"),
            EvaluationSample(text="query2", label=1, split="test"),
        ]
        search_results = [
            [(EvaluationSample(text="result1", label=0, split="test"), 0.1)],
            [(EvaluationSample(text="result2", label=1, split="test"), 0.2)],
        ]
        k_values = [1, 5]

        result = calculator.calculate_retrieval_metrics(queries, search_results, k_values)

        assert result.mrr == 1.0

    def test_should_calculate_zero_mrr_when_no_relevant_results(self, calculator: SklearnMetricsCalculator) -> None:
        queries = [
            EvaluationSample(text="query1", label=0, split="test"),
        ]
        search_results = [
            [(EvaluationSample(text="result1", label=1, split="test"), 0.1)],
        ]
        k_values = [1]

        result = calculator.calculate_retrieval_metrics(queries, search_results, k_values)

        assert result.mrr == 0.0

    def test_should_include_recall_at_k_for_multiple_k_values(self, calculator: SklearnMetricsCalculator) -> None:
        queries = [
            EvaluationSample(text="query1", label=0, split="test"),
        ]
        search_results = [
            [
                (EvaluationSample(text="result1", label=0, split="test"), 0.1),
                (EvaluationSample(text="result2", label=1, split="test"), 0.2),
            ],
        ]
        k_values = [1, 2, 5]

        result = calculator.calculate_retrieval_metrics(queries, search_results, k_values)

        assert 1 in result.recall_at_k
        assert 2 in result.recall_at_k
        assert 5 in result.recall_at_k

    def test_should_handle_zero_division_in_macro_f1(self, calculator: SklearnMetricsCalculator) -> None:
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]

        result = calculator.calculate_classification_metrics(y_true, y_pred)

        assert result.macro_f1 >= 0.0

    def test_should_compute_perfect_ndcg_when_relevant_is_ranked_first(
        self, calculator: SklearnMetricsCalculator
    ) -> None:
        queries = [_sample("q", 0)]
        search_results = [[(_sample("r0", 0), 0.1), (_sample("r1", 1), 0.2)]]

        ndcg = calculator.calculate_ndcg_at_k(queries, search_results, [2])

        assert ndcg[2] == 1.0

    def test_should_penalise_ndcg_when_relevant_is_ranked_second(self, calculator: SklearnMetricsCalculator) -> None:
        queries = [_sample("q", 0)]
        search_results = [[(_sample("r0", 1), 0.1), (_sample("r1", 0), 0.2)]]

        ndcg = calculator.calculate_ndcg_at_k(queries, search_results, [2])

        assert ndcg[2] == pytest.approx(0.63093, abs=1e-4)

    def test_should_return_zero_ndcg_when_rankings_have_fewer_than_two_documents(
        self, calculator: SklearnMetricsCalculator
    ) -> None:
        queries = [_sample("q", 0)]
        search_results = [[(_sample("r0", 0), 0.1)]]

        ndcg = calculator.calculate_ndcg_at_k(queries, search_results, [1])

        assert ndcg[1] == 0.0

    def test_should_return_zero_ndcg_when_there_are_no_queries(self, calculator: SklearnMetricsCalculator) -> None:
        ndcg = calculator.calculate_ndcg_at_k([], [], [5])

        assert ndcg[5] == 0.0
