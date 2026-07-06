import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.book_specification import (
    BeatBrief,
    BookSpecification,
    ToneTarget,
)


class TestToneTarget:
    def test_should_store_lightness_when_valid(self) -> None:
        target = ToneTarget(lightness=40.0, chroma=20.0, hue=200.0)

        assert_that(target.lightness).is_equal_to(40.0)

    def test_should_allow_all_axes_unset(self) -> None:
        target = ToneTarget()

        assert_that(target.hue).is_none()

    def test_should_reject_lightness_above_range(self) -> None:
        with pytest.raises(ValueError):
            ToneTarget(lightness=120.0)

    def test_should_reject_negative_chroma(self) -> None:
        with pytest.raises(ValueError):
            ToneTarget(chroma=-1.0)

    def test_should_reject_hue_at_upper_bound(self) -> None:
        with pytest.raises(ValueError):
            ToneTarget(hue=360.0)


class TestBeatBrief:
    def test_should_store_index_when_valid(self) -> None:
        brief = BeatBrief(index=2, title="Storm", synopsis="A storm breaks.")

        assert_that(brief.index).is_equal_to(2)

    def test_should_default_target_tone_to_none(self) -> None:
        brief = BeatBrief(index=0, title="Opening", synopsis="A calm morning.")

        assert_that(brief.target_tone).is_none()

    def test_should_reject_negative_index(self) -> None:
        with pytest.raises(ValueError):
            BeatBrief(index=-1, title="Storm", synopsis="A storm breaks.")

    def test_should_reject_empty_title(self) -> None:
        with pytest.raises(ValueError):
            BeatBrief(index=0, title="", synopsis="A storm breaks.")

    def test_should_reject_empty_synopsis(self) -> None:
        with pytest.raises(ValueError):
            BeatBrief(index=0, title="Storm", synopsis="")


class TestBookSpecification:
    def test_should_store_summary_when_valid(self) -> None:
        specification = BookSpecification(summary="A quiet town.", num_beats=3, words_per_beat=200)

        assert_that(specification.summary).is_equal_to("A quiet town.")

    def test_should_reject_blank_summary(self) -> None:
        with pytest.raises(ValueError):
            BookSpecification(summary="   ", num_beats=3, words_per_beat=200)

    def test_should_reject_non_positive_num_beats(self) -> None:
        with pytest.raises(ValueError):
            BookSpecification(summary="A town.", num_beats=0, words_per_beat=200)

    def test_should_reject_non_positive_words_per_beat(self) -> None:
        with pytest.raises(ValueError):
            BookSpecification(summary="A town.", num_beats=3, words_per_beat=0)

    def test_should_reject_mismatched_tone_targets_length(self) -> None:
        with pytest.raises(ValueError):
            BookSpecification(
                summary="A town.",
                num_beats=2,
                words_per_beat=200,
                beat_tone_targets=[ToneTarget(lightness=10.0)],
            )

    def test_should_return_tone_target_for_index(self) -> None:
        specification = BookSpecification(
            summary="A town.",
            num_beats=2,
            words_per_beat=200,
            beat_tone_targets=[ToneTarget(lightness=10.0), ToneTarget(lightness=90.0)],
        )

        assert_that(specification.tone_target_for(1).lightness).is_equal_to(90.0)

    def test_should_return_none_tone_target_when_targets_absent(self) -> None:
        specification = BookSpecification(summary="A town.", num_beats=2, words_per_beat=200)

        assert_that(specification.tone_target_for(0)).is_none()

    def test_should_return_none_tone_target_when_index_out_of_range(self) -> None:
        specification = BookSpecification(
            summary="A town.",
            num_beats=1,
            words_per_beat=200,
            beat_tone_targets=[ToneTarget(lightness=10.0)],
        )

        assert_that(specification.tone_target_for(5)).is_none()
