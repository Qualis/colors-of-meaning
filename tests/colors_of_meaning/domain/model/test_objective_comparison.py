import math

import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.objective_comparison import (
    ObjectiveArmResult,
    ObjectiveComparison,
)

BASELINE_ARM = "cosine_centred"
CHALLENGER_ARM = "delta_e_correlation"


def _arm(arm: str, mean_rho: float, stdev_rho: float = 0.02, accuracy: float = 0.81) -> ObjectiveArmResult:
    return ObjectiveArmResult(arm=arm, mean_rho=mean_rho, stdev_rho=stdev_rho, seeds=8, accuracy=accuracy)


def _comparison(*results: ObjectiveArmResult) -> ObjectiveComparison:
    return ObjectiveComparison(results=list(results), baseline_arm=BASELINE_ARM)


class TestObjectiveArmResult:
    def test_should_report_absolute_correlation_as_strength(self) -> None:
        assert_that(_arm(BASELINE_ARM, -0.39).strength()).is_close_to(0.39, 1e-9)

    def test_should_reject_a_seed_count_below_one(self) -> None:
        with pytest.raises(ValueError, match="seeds must be at least 1"):
            ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-0.39, stdev_rho=0.01, seeds=0)

    def test_should_reject_a_negative_standard_deviation(self) -> None:
        with pytest.raises(ValueError, match="stdev_rho must be non-negative"):
            ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-0.39, stdev_rho=-0.01, seeds=8)

    def test_should_reject_a_correlation_outside_the_unit_range(self) -> None:
        with pytest.raises(ValueError, match="mean_rho must be between -1 and 1"):
            ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-1.4, stdev_rho=0.01, seeds=8)

    def test_should_reject_an_accuracy_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="accuracy must be between 0 and 1"):
            ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-0.39, stdev_rho=0.01, seeds=8, accuracy=1.2)

    def test_should_reject_a_recall_outside_the_unit_interval(self) -> None:
        with pytest.raises(ValueError, match="recall@5 must be between 0 and 1"):
            ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-0.39, stdev_rho=0.01, seeds=8, recall_at_k={5: 1.5})

    def test_should_accept_a_measured_recall_inside_the_unit_interval(self) -> None:
        result = ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-0.39, stdev_rho=0.01, seeds=8, recall_at_k={5: 0.6})

        assert_that(result.recall_at_k).is_equal_to({5: 0.6})


class TestObjectiveComparisonValidation:
    def test_should_reject_a_missing_baseline_arm(self) -> None:
        with pytest.raises(ValueError, match="baseline arm cosine_centred is missing"):
            ObjectiveComparison(results=[_arm(CHALLENGER_ARM, -0.45)], baseline_arm=BASELINE_ARM)

    def test_should_reject_arms_scored_over_different_seed_counts(self) -> None:
        uneven = ObjectiveArmResult(arm=CHALLENGER_ARM, mean_rho=-0.45, stdev_rho=0.02, seeds=3, accuracy=0.8)

        with pytest.raises(ValueError, match="same seed count"):
            _comparison(_arm(BASELINE_ARM, -0.39), uneven)

    def test_should_reject_a_control_scored_over_a_different_seed_count(self) -> None:
        uneven = ObjectiveArmResult(arm="noise", mean_rho=-0.05, stdev_rho=0.01, seeds=3)

        with pytest.raises(ValueError, match="same seed count"):
            ObjectiveComparison(results=[_arm(BASELINE_ARM, -0.39)], baseline_arm=BASELINE_ARM, controls=[uneven])

    def test_should_reject_an_arm_that_is_also_reported_as_a_control(self) -> None:
        with pytest.raises(ValueError, match="arm names must be unique"):
            ObjectiveComparison(
                results=[_arm(BASELINE_ARM, -0.39)], baseline_arm=BASELINE_ARM, controls=[_arm(BASELINE_ARM, -0.39)]
            )

    def test_should_reject_duplicate_arm_names(self) -> None:
        with pytest.raises(ValueError, match="arm names must be unique"):
            _comparison(_arm(BASELINE_ARM, -0.39), _arm(BASELINE_ARM, -0.41))

    def test_should_reject_a_negative_adoption_threshold(self) -> None:
        with pytest.raises(ValueError, match="adoption_threshold_sigma must be non-negative"):
            ObjectiveComparison(
                results=[_arm(BASELINE_ARM, -0.39)], baseline_arm=BASELINE_ARM, adoption_threshold_sigma=-1.0
            )

    def test_should_reject_a_negative_accuracy_guard(self) -> None:
        with pytest.raises(ValueError, match="max_accuracy_drop must be non-negative"):
            ObjectiveComparison(results=[_arm(BASELINE_ARM, -0.39)], baseline_arm=BASELINE_ARM, max_accuracy_drop=-0.1)

    def test_should_expose_the_baseline_result(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.45))

        assert_that(comparison.baseline().arm).is_equal_to(BASELINE_ARM)

    def test_should_list_every_arm_except_the_baseline_as_a_challenger(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.45))

        assert_that([challenger.arm for challenger in comparison.challengers()]).is_equal_to([CHALLENGER_ARM])

    def test_should_default_to_no_controls(self) -> None:
        assert_that(_comparison(_arm(BASELINE_ARM, -0.39)).controls).is_equal_to(())


class TestAdoptionRule:
    def test_should_keep_the_baseline_when_the_margin_is_under_the_threshold(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.41))

        assert_that(comparison.adopted_arm()).is_equal_to(BASELINE_ARM)

    def test_should_adopt_the_challenger_when_the_margin_clears_the_threshold(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.55))

        assert_that(comparison.adopted_arm()).is_equal_to(CHALLENGER_ARM)

    def test_should_keep_the_baseline_when_the_challenger_fails_the_accuracy_guard(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.55, accuracy=0.79))

        assert_that(comparison.adopted_arm()).is_equal_to(BASELINE_ARM)

    def test_should_accept_a_challenger_exactly_one_point_below_the_baseline_accuracy(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.55, accuracy=0.80))

        assert_that(comparison.adopted_arm()).is_equal_to(CHALLENGER_ARM)

    def test_should_keep_the_baseline_when_the_challenger_has_no_measured_accuracy(self) -> None:
        challenger = ObjectiveArmResult(arm=CHALLENGER_ARM, mean_rho=-0.55, stdev_rho=0.02, seeds=8)

        assert_that(_comparison(_arm(BASELINE_ARM, -0.39), challenger).adopted_arm()).is_equal_to(BASELINE_ARM)

    def test_should_keep_the_baseline_when_the_baseline_has_no_measured_accuracy(self) -> None:
        baseline = ObjectiveArmResult(arm=BASELINE_ARM, mean_rho=-0.39, stdev_rho=0.02, seeds=8)

        assert_that(_comparison(baseline, _arm(CHALLENGER_ARM, -0.55)).adopted_arm()).is_equal_to(BASELINE_ARM)

    def test_should_choose_the_strongest_challenger_when_several_clear_the_rule(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.55), _arm("margin_ranking", -0.62))

        assert_that(comparison.adopted_arm()).is_equal_to("margin_ranking")

    def test_should_pool_the_seed_standard_deviations_of_baseline_and_challenger(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.03), _arm(CHALLENGER_ARM, -0.45, stdev_rho=0.05))

        pooled = comparison.pooled_stdev(comparison.challengers()[0])

        assert_that(pooled).is_close_to(math.sqrt((0.03**2 + 0.05**2) / 2.0), 1e-12)

    def test_should_express_the_margin_in_units_of_pooled_standard_deviation(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.40, stdev_rho=0.02), _arm(CHALLENGER_ARM, -0.50, stdev_rho=0.02))

        assert_that(comparison.margin_in_sigma(comparison.challengers()[0])).is_close_to(5.0, 1e-9)

    def test_should_report_an_infinite_margin_when_both_arms_have_zero_spread(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.55, stdev_rho=0.0))

        assert_that(comparison.margin_in_sigma(comparison.challengers()[0])).is_equal_to(math.inf)

    def test_should_keep_the_baseline_when_the_seed_spread_is_unmeasurable(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.55, stdev_rho=0.0))

        assert_that(comparison.adopted_arm()).is_equal_to(BASELINE_ARM)

    def test_should_report_an_unmeasurable_spread_when_both_arms_are_spreadless(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.55, stdev_rho=0.0))

        assert_that(comparison.has_measurable_spread(comparison.challengers()[0])).is_false()

    def test_should_keep_the_baseline_when_the_margin_lands_exactly_on_the_threshold(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.25, stdev_rho=0.25), _arm(CHALLENGER_ARM, -0.75, stdev_rho=0.25))

        assert_that(comparison.adopted_arm()).is_equal_to(BASELINE_ARM)

    def test_should_report_a_negative_infinite_margin_when_a_spreadless_challenger_is_weaker(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.55, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.39, stdev_rho=0.0))

        assert_that(comparison.margin_in_sigma(comparison.challengers()[0])).is_equal_to(-math.inf)

    def test_should_report_a_zero_margin_when_spreadless_arms_tie(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.39, stdev_rho=0.0))

        assert_that(comparison.margin_in_sigma(comparison.challengers()[0])).is_equal_to(0.0)
