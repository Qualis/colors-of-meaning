from typing import Dict, List, Optional, Sequence
from unittest.mock import Mock

import numpy as np
import pytest

from colors_of_meaning.application.use_case.compare_structure_objectives_use_case import (
    ArmRequest,
    CompareStructureObjectivesUseCase,
)
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.retrieval_evaluation import RetrievalEvaluation
from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)

BASELINE_ARM = "cosine_centred"
CHALLENGER_ARM = "delta_e_correlation"
SEEDS = [1, 2, 3]
ARM_CORRELATIONS = {BASELINE_ARM: -0.30, CHALLENGER_ARM: -0.60, "noise": -0.01}


def _embeddings(count: int = 4) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(count, 5))


def _lab_colors(count: int = 4) -> List[LabColor]:
    return [LabColor(l=float(index), a=0.0, b=0.0) for index in range(count)]


class _ScriptedEvaluator(StructurePreservationEvaluator):
    def __init__(self, correlations: Sequence[float]) -> None:
        self.correlations = list(correlations)
        self.calls = 0
        self.scored_embeddings: List[np.ndarray] = []

    def evaluate(self, embeddings: np.ndarray, lab_colors: List[LabColor]) -> float:
        self.scored_embeddings.append(embeddings)
        correlation = self.correlations[self.calls % len(self.correlations)]
        self.calls += 1
        return correlation

    def metric_name(self) -> str:
        return "scripted"


class _ArmEvaluator(StructurePreservationEvaluator):
    def __init__(self, requests: List[ArmRequest]) -> None:
        self.requests = requests

    def evaluate(self, embeddings: np.ndarray, lab_colors: List[LabColor]) -> float:
        return ARM_CORRELATIONS[self.requests[-1].arm]

    def metric_name(self) -> str:
        return "per_arm"


class _RecordingFactory:
    def __init__(self, requests: List[ArmRequest]) -> None:
        self.requests = requests

    def __call__(self, request: ArmRequest) -> List[LabColor]:
        self.requests.append(request)
        return _lab_colors()


def _evaluation_result(accuracy: float = 0.8, recall_at_k: Optional[Dict[int, float]] = None) -> EvaluationResult:
    return EvaluationResult(
        accuracy=accuracy, macro_f1=accuracy, recall_at_k=recall_at_k or {5: 0.5}, mrr=0.4, bits_per_token=12.0
    )


def _classification_factory(accuracy: float = 0.8) -> Mock:
    use_case = Mock()
    use_case.execute.return_value = _evaluation_result(accuracy)
    return Mock(return_value=use_case)


def _retrieval_factory() -> Mock:
    use_case = Mock()
    use_case.execute.return_value = RetrievalEvaluation(result=_evaluation_result(), ndcg_at_k={5: 0.5})
    return Mock(return_value=use_case)


def _use_case(**overrides) -> CompareStructureObjectivesUseCase:
    requests: List[ArmRequest] = overrides.pop("requests", [])
    settings = {
        "lab_colors_factory": _RecordingFactory(requests),
        "structure_preservation_evaluator": _ArmEvaluator(requests),
    }
    settings.update(overrides)
    return CompareStructureObjectivesUseCase(**settings)


def _execute(use_case: CompareStructureObjectivesUseCase, **overrides):
    settings = {
        "train_embeddings": _embeddings(),
        "eval_embeddings": _embeddings(),
        "arms": [BASELINE_ARM, CHALLENGER_ARM],
        "seeds": SEEDS,
        "baseline_arm": BASELINE_ARM,
    }
    settings.update(overrides)
    return use_case.execute(**settings)


class TestArmScoring:
    def test_should_score_every_arm_at_every_seed(self) -> None:
        requests: List[ArmRequest] = []

        _execute(_use_case(requests=requests))

        assert len(requests) == len(SEEDS) * 2

    def test_should_pass_the_arm_and_seed_to_the_factory(self) -> None:
        requests: List[ArmRequest] = []

        _execute(_use_case(requests=requests))

        assert [(request.arm, request.seed) for request in requests[: len(SEEDS)]] == [
            (BASELINE_ARM, seed) for seed in SEEDS
        ]

    def test_should_hand_both_embedding_splits_to_the_factory(self) -> None:
        requests: List[ArmRequest] = []
        train = _embeddings(6)

        _execute(_use_case(requests=requests), train_embeddings=train)

        assert requests[0].train_embeddings is train

    def test_should_hand_the_held_out_split_to_the_factory_as_the_evaluation_set(self) -> None:
        requests: List[ArmRequest] = []
        held_out = _embeddings(5)

        _execute(_use_case(requests=requests), eval_embeddings=held_out)

        assert requests[0].eval_embeddings is held_out

    def test_should_score_the_correlation_on_the_held_out_split_not_the_training_split(self) -> None:
        evaluator = _ScriptedEvaluator([-0.4])
        held_out = _embeddings(5)

        _execute(
            _use_case(structure_preservation_evaluator=evaluator),
            eval_embeddings=held_out,
            arms=[BASELINE_ARM],
            seeds=[1],
        )

        assert evaluator.scored_embeddings[0] is held_out

    def test_should_score_each_arm_and_seed_exactly_once_across_repeated_executions(self) -> None:
        evaluator = _ScriptedEvaluator([-0.4])
        use_case = _use_case(structure_preservation_evaluator=evaluator)
        held_out = _embeddings(4)
        train = _embeddings(6)

        _execute(use_case, arms=[BASELINE_ARM], seeds=[1], eval_embeddings=held_out, train_embeddings=train)
        _execute(use_case, arms=[BASELINE_ARM], seeds=[1], eval_embeddings=held_out, train_embeddings=train)

        assert evaluator.calls == 1

    def test_should_rescore_every_arm_when_the_evaluation_split_changes(self) -> None:
        evaluator = _ScriptedEvaluator([-0.4])
        use_case = _use_case(structure_preservation_evaluator=evaluator)

        _execute(use_case, arms=[BASELINE_ARM], seeds=[1], eval_embeddings=_embeddings(4))
        _execute(use_case, arms=[BASELINE_ARM], seeds=[1], eval_embeddings=_embeddings(5))

        assert evaluator.calls == 2

    def test_should_reject_an_arm_whose_projector_collapsed_to_a_constant_color(self) -> None:
        use_case = _use_case(structure_preservation_evaluator=_ScriptedEvaluator([float("nan")]))

        with pytest.raises(ValueError, match="undefined structure correlation"):
            _execute(use_case, arms=[BASELINE_ARM], seeds=[1])

    def test_should_reject_a_seed_correlation_outside_the_unit_range(self) -> None:
        use_case = _use_case(structure_preservation_evaluator=_ScriptedEvaluator([-1.5]))

        with pytest.raises(ValueError, match=r"correlation outside \[-1, 1\]"):
            _execute(use_case, arms=[BASELINE_ARM], seeds=[1])

    def test_should_rescore_every_arm_when_the_training_split_changes(self) -> None:
        evaluator = _ScriptedEvaluator([-0.4])
        use_case = _use_case(structure_preservation_evaluator=evaluator)
        held_out = _embeddings(4)

        _execute(use_case, arms=[BASELINE_ARM], seeds=[1], eval_embeddings=held_out, train_embeddings=_embeddings(6))
        _execute(use_case, arms=[BASELINE_ARM], seeds=[1], eval_embeddings=held_out, train_embeddings=_embeddings(7))

        assert evaluator.calls == 2

    def test_should_average_the_seed_correlations_into_the_arm_mean(self) -> None:
        use_case = _use_case(structure_preservation_evaluator=_ScriptedEvaluator([-0.4, -0.5, -0.6]))

        comparison = _execute(use_case, arms=[BASELINE_ARM], seeds=SEEDS)

        assert comparison.baseline().mean_rho == pytest.approx(-0.5, abs=1e-9)

    def test_should_report_the_sample_standard_deviation_across_seeds(self) -> None:
        use_case = _use_case(structure_preservation_evaluator=_ScriptedEvaluator([-0.4, -0.5, -0.6]))

        comparison = _execute(use_case, arms=[BASELINE_ARM], seeds=SEEDS)

        assert comparison.baseline().stdev_rho == pytest.approx(0.1, abs=1e-9)

    def test_should_report_a_zero_spread_when_a_single_seed_is_requested(self) -> None:
        use_case = _use_case(structure_preservation_evaluator=_ScriptedEvaluator([-0.4]))

        comparison = _execute(use_case, arms=[BASELINE_ARM], seeds=[7])

        assert comparison.baseline().stdev_rho == 0.0

    def test_should_record_the_seed_count_on_the_arm_result(self) -> None:
        comparison = _execute(_use_case(), arms=[BASELINE_ARM])

        assert comparison.baseline().seeds == len(SEEDS)

    def test_should_score_controls_separately_from_the_arms(self) -> None:
        comparison = _execute(_use_case(), arms=[BASELINE_ARM], controls=["noise"])

        assert [control.arm for control in comparison.controls] == ["noise"]

    def test_should_log_each_arm_and_seed_with_a_correlation_id(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            _execute(_use_case(), arms=[BASELINE_ARM], seeds=[1])

        assert any(vars(record).get("arm") == BASELINE_ARM for record in caplog.records)

    def test_should_carry_a_correlation_id_on_every_arm_and_seed_line(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            _execute(_use_case(), arms=[BASELINE_ARM], seeds=[1])

        seed_lines = [record for record in caplog.records if vars(record).get("arm") == BASELINE_ARM]
        assert seed_lines and all(vars(record).get("correlation_id") for record in seed_lines)


class TestDownstreamEvaluation:
    def test_should_skip_the_downstream_evaluation_for_arms_that_are_not_nominated(self) -> None:
        factory = _classification_factory()

        _execute(_use_case(evaluate_use_case_factory=factory), downstream_arms=[CHALLENGER_ARM])

        assert [call.args[0].arm for call in factory.call_args_list] == [CHALLENGER_ARM] * len(SEEDS)

    def test_should_leave_accuracy_unmeasured_when_no_downstream_factory_is_injected(self) -> None:
        comparison = _execute(_use_case(), downstream_arms=[BASELINE_ARM])

        assert comparison.baseline().accuracy is None

    def test_should_average_the_downstream_accuracy_over_the_downstream_seeds(self) -> None:
        use_case = _use_case(evaluate_use_case_factory=_classification_factory(accuracy=0.75))

        comparison = _execute(use_case, downstream_arms=[BASELINE_ARM])

        assert comparison.baseline().accuracy == pytest.approx(0.75, abs=1e-9)

    def test_should_use_the_dedicated_downstream_seeds_when_they_are_supplied(self) -> None:
        factory = _classification_factory()

        _execute(_use_case(evaluate_use_case_factory=factory), downstream_arms=[BASELINE_ARM], downstream_seeds=[9])

        assert [call.args[0].seed for call in factory.call_args_list] == [9]

    def test_should_forward_the_downstream_budget_to_the_evaluation(self) -> None:
        factory = _classification_factory()
        use_case = _use_case(evaluate_use_case_factory=factory, downstream_budget=4000)

        _execute(use_case, downstream_arms=[BASELINE_ARM], downstream_seeds=[9])

        assert factory.return_value.execute.call_args.kwargs["max_samples"] == 4000

    def test_should_forward_the_downstream_bits_per_token_to_the_evaluation(self) -> None:
        factory = _classification_factory()
        use_case = _use_case(evaluate_use_case_factory=factory, downstream_bits_per_token=12.0)

        _execute(use_case, downstream_arms=[BASELINE_ARM], downstream_seeds=[9])

        assert factory.return_value.execute.call_args.kwargs["bits_per_token"] == 12.0

    def test_should_leave_retrieval_unmeasured_when_no_retrieval_factory_is_injected(self) -> None:
        use_case = _use_case(evaluate_use_case_factory=_classification_factory())

        comparison = _execute(use_case, downstream_arms=[BASELINE_ARM])

        assert comparison.baseline().mrr is None

    def test_should_average_the_retrieval_reciprocal_rank_when_retrieval_is_measured(self) -> None:
        use_case = _use_case(
            evaluate_use_case_factory=_classification_factory(), retrieval_use_case_factory=_retrieval_factory()
        )

        comparison = _execute(use_case, downstream_arms=[BASELINE_ARM])

        assert comparison.baseline().mrr == pytest.approx(0.4, abs=1e-9)

    def test_should_average_recall_at_each_depth_when_retrieval_is_measured(self) -> None:
        use_case = _use_case(
            evaluate_use_case_factory=_classification_factory(), retrieval_use_case_factory=_retrieval_factory()
        )

        comparison = _execute(use_case, downstream_arms=[BASELINE_ARM])

        assert comparison.baseline().recall_at_k == {5: pytest.approx(0.5, abs=1e-9)}


class TestAdoptionDecision:
    def test_should_report_the_baseline_when_no_challenger_has_downstream_accuracy(self) -> None:
        comparison = _execute(_use_case())

        assert comparison.adopted_arm() == BASELINE_ARM

    def test_should_adopt_a_stronger_challenger_that_holds_its_accuracy(self) -> None:
        use_case = _use_case(
            evaluate_use_case_factory=_classification_factory(),
            structure_preservation_evaluator=_ScriptedEvaluator([-0.30, -0.31, -0.29, -0.60, -0.61, -0.59]),
        )

        comparison = _execute(use_case, downstream_arms=[BASELINE_ARM, CHALLENGER_ARM])

        assert comparison.adopted_arm() == CHALLENGER_ARM

    def test_should_log_the_adopted_arm_with_its_margin(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            _execute(_use_case())

        assert any("margins_in_sigma" in vars(record) for record in caplog.records)

    def test_should_use_the_injected_correlation_id_on_every_line(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            _execute(_use_case(), arms=[BASELINE_ARM], seeds=[1], correlation_id="fixed-id")

        assert all(
            vars(record).get("correlation_id") == "fixed-id"
            for record in caplog.records
            if "correlation_id" in vars(record)
        )

    def test_should_withhold_the_decision_log_when_the_pass_is_only_nominating(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO"):
            _execute(_use_case(), log_decision=False)

        assert not any("margins_in_sigma" in vars(record) for record in caplog.records)


class TestElapsedTiming:
    def test_should_log_the_clock_delta_as_the_arm_elapsed_time(self, caplog: pytest.LogCaptureFixture) -> None:
        ticks = iter([10.0, 13.5])
        use_case = _use_case(clock=lambda: next(ticks))

        with caplog.at_level("INFO"):
            _execute(use_case, arms=[BASELINE_ARM], seeds=[1])

        assert any(vars(record).get("seconds") == 3.5 for record in caplog.records)
