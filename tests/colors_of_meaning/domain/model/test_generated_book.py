from typing import List

import numpy as np
import pytest
from assertpy import assert_that

from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.generated_book import (
    GeneratedBook,
    GeneratedChapter,
    GenerationMetadata,
    TokenUsage,
)
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.narrative_arc import ArcPoint, NarrativeArc, NarrativeBeat


def _arc(beat_count: int) -> NarrativeArc:
    points = [
        ArcPoint(
            beat=NarrativeBeat(index=index, title=f"Beat {index}", text="prose"),
            colour=LabColor(l=50.0, a=0.0, b=0.0),
            histogram=ColoredDocument(histogram=np.array([1.0]), document_id=f"beat_{index}"),
        )
        for index in range(beat_count)
    ]
    return NarrativeArc(
        points=points,
        drift_series=[0.0] * (beat_count - 1),
        coherence_series=[0.0] * beat_count,
    )


def _chapters(beat_count: int) -> List[GeneratedChapter]:
    return [
        GeneratedChapter(brief=BeatBrief(index=index, title=f"Beat {index}", synopsis="s"), prose="text")
        for index in range(beat_count)
    ]


def _metadata() -> GenerationMetadata:
    return GenerationMetadata(total_tokens=TokenUsage(1, 2), total_retries=0, flagged_beat_indices=[])


def _specification() -> BookSpecification:
    return BookSpecification(summary="A town.", num_beats=2, words_per_beat=200)


class TestTokenUsage:
    def test_should_sum_input_and_output(self) -> None:
        usage = TokenUsage(input_tokens=10, output_tokens=25)

        assert_that(usage.total_tokens).is_equal_to(35)

    def test_should_reject_negative_input_tokens(self) -> None:
        with pytest.raises(ValueError):
            TokenUsage(input_tokens=-1, output_tokens=0)

    def test_should_reject_negative_output_tokens(self) -> None:
        with pytest.raises(ValueError):
            TokenUsage(input_tokens=0, output_tokens=-1)


class TestGeneratedChapter:
    def test_should_store_prose_when_present(self) -> None:
        chapter = GeneratedChapter(brief=BeatBrief(index=0, title="Opening", synopsis="s"), prose="Once upon a time.")

        assert_that(chapter.prose).is_equal_to("Once upon a time.")

    def test_should_reject_blank_prose(self) -> None:
        with pytest.raises(ValueError):
            GeneratedChapter(brief=BeatBrief(index=0, title="Opening", synopsis="s"), prose="   ")


class TestGenerationMetadata:
    def test_should_store_flagged_indices(self) -> None:
        metadata = GenerationMetadata(total_tokens=TokenUsage(1, 1), total_retries=2, flagged_beat_indices=[3])

        assert_that(metadata.flagged_beat_indices).is_equal_to([3])

    def test_should_reject_negative_retries(self) -> None:
        with pytest.raises(ValueError):
            GenerationMetadata(total_tokens=TokenUsage(1, 1), total_retries=-1, flagged_beat_indices=[])


class TestGeneratedBook:
    def test_should_report_chapter_count(self) -> None:
        book = GeneratedBook(specification=_specification(), chapters=_chapters(2), arc=_arc(2), metadata=_metadata())

        assert_that(book.chapter_count).is_equal_to(2)

    def test_should_reject_empty_chapters(self) -> None:
        with pytest.raises(ValueError):
            GeneratedBook(specification=_specification(), chapters=[], arc=_arc(2), metadata=_metadata())

    def test_should_reject_chapter_arc_length_mismatch(self) -> None:
        with pytest.raises(ValueError):
            GeneratedBook(specification=_specification(), chapters=_chapters(3), arc=_arc(2), metadata=_metadata())
