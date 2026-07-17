from typing import Callable, List
from unittest.mock import Mock

import pytest

from colors_of_meaning.application.use_case.evaluation_suite_use_case import (
    EvaluatedCell,
    EvaluationCell,
    EvaluationSuiteUseCase,
    UnfaithfulProxyError,
    _cell_log_payload,
)
from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.retrieval_evaluation import RetrievalEvaluation


def _evaluation_result() -> EvaluationResult:
    return EvaluationResult(accuracy=0.8, macro_f1=0.78, recall_at_k={}, mrr=0.7)


def _retrieval_result(mrr: float = 0.9) -> EvaluationResult:
    return EvaluationResult(accuracy=0.0, macro_f1=0.0, recall_at_k={5: 0.8}, mrr=mrr)


def _retrieval_use_case(result: EvaluationResult) -> Mock:
    use_case = Mock()
    use_case.execute.return_value = RetrievalEvaluation(result=result, ndcg_at_k={})
    return use_case


def _retrieval_factory(use_case: Mock) -> Mock:
    return Mock(return_value=use_case)


def _fidelity(is_faithful: bool) -> DistanceFidelity:
    spearman = 0.99 if is_faithful else 0.10
    return DistanceFidelity(
        spearman=spearman, accuracy_delta=0.2, pair_count=100, threshold_spearman=0.95, max_accuracy_delta=1.0
    )


def _cell(
    distance: str = "sliced",
    budget: int = 4000,
    requires_fidelity: bool = True,
    method: str = "color",
    supports_retrieval: bool = False,
) -> EvaluationCell:
    return EvaluationCell(
        dataset="ag_news",
        method=method,
        distance=distance,
        budget=budget,
        requires_fidelity=requires_fidelity,
        bits_per_token=12.0,
        supports_retrieval=supports_retrieval,
    )


def _incrementing_clock(values: List[float]) -> Callable[[], float]:
    iterator = iter(values)
    return lambda: next(iterator)


def _factory(evaluate_use_case: Mock) -> Mock:
    return Mock(return_value=evaluate_use_case)


def _evaluate_use_case(result: EvaluationResult) -> Mock:
    evaluate_use_case = Mock()
    evaluate_use_case.execute.return_value = result
    return evaluate_use_case


class TestEvaluationSuiteUseCase:
    def test_should_produce_one_evaluated_cell_per_cell_when_proxy_is_faithful(self) -> None:
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(_evaluation_result())))

        evaluated = suite.execute([_cell(), _cell(distance="wasserstein", requires_fidelity=False)], _fidelity(True))

        assert len(evaluated) == 2

    def test_should_carry_cell_and_result_into_each_evaluated_cell(self) -> None:
        cell = _cell()
        result = _evaluation_result()
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(result)))

        evaluated = suite.execute([cell], _fidelity(True))

        assert evaluated[0] == EvaluatedCell(cell=cell, result=result, seconds=evaluated[0].seconds)

    def test_should_raise_when_scaled_proxy_cell_present_and_proxy_unfaithful(self) -> None:
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(_evaluation_result())))

        with pytest.raises(UnfaithfulProxyError):
            suite.execute([_cell(requires_fidelity=True)], _fidelity(False))

    def test_should_report_only_spearman_in_message_when_only_spearman_fails(self) -> None:
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(_evaluation_result())))
        unfaithful = DistanceFidelity(
            spearman=0.10, accuracy_delta=0.2, pair_count=100, threshold_spearman=0.95, max_accuracy_delta=1.0
        )

        with pytest.raises(UnfaithfulProxyError) as raised:
            suite.execute([_cell()], unfaithful)

        assert "spearman" in str(raised.value) and "accuracy_delta" not in str(raised.value)

    def test_should_report_only_accuracy_delta_in_message_when_only_delta_fails(self) -> None:
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(_evaluation_result())))
        unfaithful = DistanceFidelity(
            spearman=0.99, accuracy_delta=2.0, pair_count=100, threshold_spearman=0.95, max_accuracy_delta=1.0
        )

        with pytest.raises(UnfaithfulProxyError) as raised:
            suite.execute([_cell()], unfaithful)

        assert "accuracy_delta" in str(raised.value) and "spearman" not in str(raised.value)

    def test_should_run_cells_when_unfaithful_but_no_cell_requires_fidelity(self) -> None:
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(_evaluation_result())))

        evaluated = suite.execute([_cell(distance="wasserstein", requires_fidelity=False)], _fidelity(False))

        assert len(evaluated) == 1

    def test_should_forward_cell_budget_to_evaluate_use_case(self) -> None:
        evaluate_use_case = _evaluate_use_case(_evaluation_result())
        suite = EvaluationSuiteUseCase(_factory(evaluate_use_case))

        suite.execute([_cell(budget=4000)], _fidelity(True))

        assert evaluate_use_case.execute.call_args[1]["max_samples"] == 4000

    def test_should_forward_seed_to_evaluate_use_case(self) -> None:
        evaluate_use_case = _evaluate_use_case(_evaluation_result())
        suite = EvaluationSuiteUseCase(_factory(evaluate_use_case), seed=99)

        suite.execute([_cell()], _fidelity(True))

        assert evaluate_use_case.execute.call_args[1]["seed"] == 99

    def test_should_record_elapsed_seconds_from_clock(self) -> None:
        suite = EvaluationSuiteUseCase(
            _factory(_evaluate_use_case(_evaluation_result())), clock=_incrementing_clock([10.0, 12.5])
        )

        evaluated = suite.execute([_cell()], _fidelity(True))

        assert evaluated[0].seconds == pytest.approx(2.5)

    def test_should_build_an_evaluate_use_case_per_cell(self) -> None:
        factory = _factory(_evaluate_use_case(_evaluation_result()))
        suite = EvaluationSuiteUseCase(factory)

        suite.execute([_cell(), _cell(distance="wasserstein", requires_fidelity=False)], _fidelity(True))

        assert factory.call_count == 2

    def test_should_leave_retrieval_absent_when_no_retrieval_factory_provided(self) -> None:
        suite = EvaluationSuiteUseCase(_factory(_evaluate_use_case(_evaluation_result())))

        evaluated = suite.execute([_cell(supports_retrieval=True)], _fidelity(True))

        assert evaluated[0].retrieval is None

    def test_should_merge_real_retrieval_result_when_method_supports_retrieval(self) -> None:
        suite = EvaluationSuiteUseCase(
            _factory(_evaluate_use_case(_evaluation_result())),
            retrieval_use_case_factory=_retrieval_factory(_retrieval_use_case(_retrieval_result(mrr=0.9))),
        )

        evaluated = suite.execute([_cell(supports_retrieval=True)], _fidelity(True))

        assert evaluated[0].retrieval.mrr == 0.9

    def test_should_keep_classification_result_when_retrieval_is_merged(self) -> None:
        suite = EvaluationSuiteUseCase(
            _factory(_evaluate_use_case(_evaluation_result())),
            retrieval_use_case_factory=_retrieval_factory(_retrieval_use_case(_retrieval_result())),
        )

        evaluated = suite.execute([_cell(supports_retrieval=True)], _fidelity(True))

        assert evaluated[0].result.accuracy == 0.8

    def test_should_record_skip_reason_when_method_does_not_support_retrieval(self) -> None:
        suite = EvaluationSuiteUseCase(
            _factory(_evaluate_use_case(_evaluation_result())),
            retrieval_use_case_factory=_retrieval_factory(_retrieval_use_case(_retrieval_result())),
        )

        evaluated = suite.execute([_cell(method="tfidf", supports_retrieval=False)], _fidelity(True))

        assert "classification-only" in evaluated[0].retrieval_skip_reason

    def test_should_still_report_classification_when_retrieval_skipped_for_tfidf(self) -> None:
        suite = EvaluationSuiteUseCase(
            _factory(_evaluate_use_case(_evaluation_result())),
            retrieval_use_case_factory=_retrieval_factory(_retrieval_use_case(_retrieval_result())),
        )

        evaluated = suite.execute([_cell(method="tfidf", supports_retrieval=False)], _fidelity(True))

        assert evaluated[0].retrieval is None

    def test_should_forward_cell_budget_to_retrieval_use_case(self) -> None:
        retrieval_use_case = _retrieval_use_case(_retrieval_result())
        suite = EvaluationSuiteUseCase(
            _factory(_evaluate_use_case(_evaluation_result())),
            retrieval_use_case_factory=_retrieval_factory(retrieval_use_case),
        )

        suite.execute([_cell(budget=4000, supports_retrieval=True)], _fidelity(True))

        assert retrieval_use_case.execute.call_args[1]["max_samples"] == 4000


class TestCellLogPayload:
    def test_should_include_bits_per_token_in_log_payload(self) -> None:
        payload = _cell_log_payload(EvaluatedCell(cell=_cell(), result=_evaluation_result(), seconds=1.0))

        assert payload["bits_per_token"] == 12.0

    def test_should_include_mrr_in_log_payload_when_retrieval_present(self) -> None:
        payload = _cell_log_payload(
            EvaluatedCell(cell=_cell(), result=_evaluation_result(), seconds=1.0, retrieval=_retrieval_result(mrr=0.9))
        )

        assert payload["mrr"] == 0.9

    def test_should_include_skip_reason_in_log_payload_when_retrieval_skipped(self) -> None:
        payload = _cell_log_payload(
            EvaluatedCell(
                cell=_cell(), result=_evaluation_result(), seconds=1.0, retrieval_skip_reason="skipped for tfidf"
            )
        )

        assert payload["retrieval_skip_reason"] == "skipped for tfidf"
