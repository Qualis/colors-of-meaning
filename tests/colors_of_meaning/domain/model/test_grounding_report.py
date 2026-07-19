import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.grounding_report import GroundingReport


def _report(divergence: float = 5.0, threshold: float = 10.0) -> GroundingReport:
    return GroundingReport(divergence=divergence, threshold=threshold)


class TestGroundingReport:
    def test_should_store_divergence_when_created(self) -> None:
        assert_that(_report(divergence=3.5).divergence).is_equal_to(3.5)

    def test_should_store_threshold_when_created(self) -> None:
        assert_that(_report(threshold=7.5).threshold).is_equal_to(7.5)

    def test_should_be_grounded_when_divergence_below_threshold(self) -> None:
        assert_that(_report(divergence=4.0, threshold=10.0).is_grounded).is_true()

    def test_should_be_grounded_when_divergence_equals_threshold(self) -> None:
        assert_that(_report(divergence=10.0, threshold=10.0).is_grounded).is_true()

    def test_should_not_be_grounded_when_divergence_above_threshold(self) -> None:
        assert_that(_report(divergence=12.0, threshold=10.0).is_grounded).is_false()

    def test_should_raise_error_when_divergence_is_negative(self) -> None:
        with pytest.raises(ValueError, match="divergence must be non-negative"):
            _report(divergence=-0.1)

    def test_should_raise_error_when_threshold_is_negative(self) -> None:
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            _report(threshold=-1.0)

    def test_should_be_immutable(self) -> None:
        report = _report()

        with pytest.raises(AttributeError):
            report.divergence = 0.0  # type: ignore[misc]
