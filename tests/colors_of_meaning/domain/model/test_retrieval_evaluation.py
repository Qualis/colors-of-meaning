import pytest

from colors_of_meaning.domain.model.retrieval_evaluation import RetrievalEvaluation
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


def _result() -> EvaluationResult:
    return EvaluationResult(accuracy=0.0, macro_f1=0.0, recall_at_k={1: 0.5}, mrr=0.5)


class TestRetrievalEvaluation:
    def test_should_store_the_evaluation_result(self) -> None:
        evaluation = RetrievalEvaluation(result=_result(), ndcg_at_k={1: 0.5})

        assert evaluation.result.mrr == 0.5

    def test_should_store_ndcg_values(self) -> None:
        evaluation = RetrievalEvaluation(result=_result(), ndcg_at_k={1: 0.75})

        assert evaluation.ndcg_at_k[1] == 0.75

    def test_should_raise_when_ndcg_is_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="ndcg@1 must be between 0 and 1"):
            RetrievalEvaluation(result=_result(), ndcg_at_k={1: 1.5})

    def test_should_raise_when_k_is_not_positive(self) -> None:
        with pytest.raises(ValueError, match="k must be positive"):
            RetrievalEvaluation(result=_result(), ndcg_at_k={0: 0.5})
