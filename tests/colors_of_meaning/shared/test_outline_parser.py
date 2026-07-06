from assertpy import assert_that

from colors_of_meaning.shared.outline_parser import parse_outline

_LONG_BODY = "The heroine leaves her village at dawn and the whole valley watches her go. " * 4
_SECOND_BODY = "A storm strands the travellers and old grievances surface around the fire. " * 4
_SHORT_BODY = "Too brief."


def _headed_outline() -> str:
    return f"# Book Title\n\n## Chapter One\n\n{_LONG_BODY}\n\n## Chapter Two\n\n{_SECOND_BODY}\n"


class TestHeadingDelimitedOutline:
    def test_should_produce_one_beat_per_qualifying_heading(self) -> None:
        beats = parse_outline(_headed_outline(), min_chars=50)

        assert_that(len(beats)).is_equal_to(2)

    def test_should_read_the_heading_text_as_the_title(self) -> None:
        beats = parse_outline(_headed_outline(), min_chars=50)

        assert_that(beats[0].title).is_equal_to("Chapter One")

    def test_should_capture_the_section_body_as_beat_text(self) -> None:
        beats = parse_outline(_headed_outline(), min_chars=50)

        assert_that(beats[1].text).contains("storm strands")

    def test_should_index_kept_beats_contiguously_from_zero(self) -> None:
        beats = parse_outline(_headed_outline(), min_chars=50)

        assert_that([beat.index for beat in beats]).is_equal_to([0, 1])

    def test_should_drop_a_heading_whose_body_is_below_minimum_length(self) -> None:
        outline = f"## Kept\n\n{_LONG_BODY}\n\n## Dropped\n\n{_SHORT_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that([beat.title for beat in beats]).is_equal_to(["Kept"])

    def test_should_drop_a_heading_with_an_empty_body(self) -> None:
        outline = f"## Empty\n\n## Full\n\n{_LONG_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that([beat.title for beat in beats]).is_equal_to(["Full"])


class TestBlankLineDelimitedOutline:
    def test_should_split_on_blank_lines_when_no_headings_present(self) -> None:
        outline = f"{_LONG_BODY}\n\n{_SECOND_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that(len(beats)).is_equal_to(2)

    def test_should_leave_the_title_empty_for_blank_line_blocks(self) -> None:
        outline = f"{_LONG_BODY}\n\n{_SECOND_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that(beats[0].title).is_equal_to("")

    def test_should_drop_a_whitespace_only_block(self) -> None:
        outline = f"{_LONG_BODY}\n\n   \n\n{_SECOND_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that(len(beats)).is_equal_to(2)

    def test_should_drop_blocks_below_minimum_length(self) -> None:
        outline = f"{_LONG_BODY}\n\n{_SHORT_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that(len(beats)).is_equal_to(1)

    def test_should_reindex_after_dropping_short_blocks(self) -> None:
        outline = f"{_SHORT_BODY}\n\n{_LONG_BODY}\n\n{_SECOND_BODY}\n"

        beats = parse_outline(outline, min_chars=50)

        assert_that([beat.index for beat in beats]).is_equal_to([0, 1])

    def test_should_return_no_beats_when_every_block_is_too_short(self) -> None:
        beats = parse_outline(f"{_SHORT_BODY}\n\n{_SHORT_BODY}\n", min_chars=50)

        assert_that(beats).is_empty()
