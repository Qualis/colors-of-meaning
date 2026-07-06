import numpy as np
import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.narrative_arc import (
    ArcPoint,
    FlaggedBeat,
    NarrativeArc,
    NarrativeBeat,
)


def _histogram(bins: int = 4) -> ColoredDocument:
    values = np.full(bins, 1.0 / bins, dtype=np.float64)
    return ColoredDocument(histogram=values)


def _point(index: int, colour: LabColor) -> ArcPoint:
    return ArcPoint(
        beat=NarrativeBeat(index=index, title=f"Beat {index}", text="prose"), colour=colour, histogram=_histogram()
    )


def _arc(coherence_series: list) -> NarrativeArc:
    points = [_point(i, LabColor(l=50.0, a=10.0, b=0.0)) for i in range(len(coherence_series))]
    drift = [0.1] * (len(points) - 1)
    return NarrativeArc(points=points, drift_series=drift, coherence_series=coherence_series)


class TestNarrativeBeat:
    def test_should_carry_index_when_constructed(self) -> None:
        beat = NarrativeBeat(index=2, title="Rising action", text="body")

        assert_that(beat.index).is_equal_to(2)

    def test_should_carry_title_when_constructed(self) -> None:
        beat = NarrativeBeat(index=0, title="Opening", text="body")

        assert_that(beat.title).is_equal_to("Opening")

    def test_should_reject_negative_index(self) -> None:
        assert_that(NarrativeBeat).raises(ValueError).when_called_with(index=-1, title="t", text="x")


class TestArcPointDecomposition:
    def test_should_read_lightness_from_colour(self) -> None:
        point = _point(0, LabColor(l=72.0, a=0.0, b=0.0))

        assert point.lightness == 72.0

    def test_should_compute_chroma_from_a_and_b(self) -> None:
        point = _point(0, LabColor(l=50.0, a=3.0, b=4.0))

        assert point.chroma == pytest.approx(5.0)

    def test_should_compute_hue_angle_in_degrees(self) -> None:
        point = _point(0, LabColor(l=50.0, a=0.0, b=10.0))

        assert point.hue_degrees == pytest.approx(90.0)

    def test_should_wrap_negative_hue_into_zero_to_360(self) -> None:
        point = _point(0, LabColor(l=50.0, a=0.0, b=-10.0))

        assert point.hue_degrees == pytest.approx(270.0)


class TestNarrativeArcValidation:
    def test_should_reject_empty_points(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            NarrativeArc(points=[], drift_series=[], coherence_series=[])

    def test_should_reject_drift_series_of_wrong_length(self) -> None:
        points = [_point(0, LabColor(l=50.0, a=1.0, b=1.0)), _point(1, LabColor(l=50.0, a=1.0, b=1.0))]

        with pytest.raises(ValueError, match="drift_series"):
            NarrativeArc(points=points, drift_series=[], coherence_series=[0.1, 0.2])

    def test_should_reject_coherence_series_of_wrong_length(self) -> None:
        points = [_point(0, LabColor(l=50.0, a=1.0, b=1.0)), _point(1, LabColor(l=50.0, a=1.0, b=1.0))]

        with pytest.raises(ValueError, match="coherence_series"):
            NarrativeArc(points=points, drift_series=[0.1], coherence_series=[0.1])

    def test_should_allow_single_beat_arc_with_empty_drift(self) -> None:
        arc = NarrativeArc(points=[_point(0, LabColor(l=50.0, a=1.0, b=1.0))], drift_series=[], coherence_series=[0.2])

        assert_that(arc.beat_count).is_equal_to(1)


class TestNarrativeArcViews:
    def test_should_expose_lightness_series_in_point_order(self) -> None:
        arc = _arc([0.1, 0.2, 0.3])

        assert_that(arc.lightness_series).is_equal_to([50.0, 50.0, 50.0])

    def test_should_expose_one_colour_per_point(self) -> None:
        arc = _arc([0.1, 0.2])

        assert_that(len(arc.colours)).is_equal_to(2)

    def test_should_expose_chroma_series(self) -> None:
        arc = _arc([0.1, 0.2])

        assert arc.chroma_series == pytest.approx([10.0, 10.0])

    def test_should_expose_hue_series(self) -> None:
        arc = _arc([0.1, 0.2])

        assert arc.hue_series == pytest.approx([0.0, 0.0])


class TestNarrativeArcFlagging:
    def test_should_flag_beats_whose_coherence_exceeds_threshold(self) -> None:
        arc = _arc([0.1, 0.9, 0.2])

        flagged = arc.flagged_beats(threshold=0.5)

        assert_that(flagged).is_equal_to([FlaggedBeat(index=1, coherence=0.9)])

    def test_should_not_flag_beats_at_or_below_threshold(self) -> None:
        arc = _arc([0.5, 0.5])

        assert_that(arc.flagged_beats(threshold=0.5)).is_empty()

    def test_should_report_flagged_beat_source_index(self) -> None:
        arc = _arc([0.1, 0.2, 0.99])

        assert_that(arc.flagged_beats(threshold=0.5)[0].index).is_equal_to(2)
