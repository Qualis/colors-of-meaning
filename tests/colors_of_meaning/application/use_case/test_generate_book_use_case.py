from typing import Callable, List, Optional, Tuple

import numpy as np
from unittest.mock import Mock

from colors_of_meaning.application.use_case.analyze_narrative_arc_use_case import (
    AnalyzeNarrativeArcUseCase,
)
from colors_of_meaning.application.use_case.generate_book_use_case import GenerateBookUseCase
from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.generated_book import TokenUsage
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.narrative_arc import ArcPoint, NarrativeArc, NarrativeBeat
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.domain.service.text_generator import TextGenerator

import pytest

_SPEC = BookSpecification(summary="A quiet town.", num_beats=3, words_per_beat=200)
_DRIFT_MARK = "DRIFT"


class _StubTextGenerator(TextGenerator):
    def __init__(self, outline: List[BeatBrief], prose_fn: Callable[[int, Optional[str]], str]) -> None:
        self._outline = outline
        self._prose_fn = prose_fn
        self.prose_calls: List[Tuple[int, str, Optional[str]]] = []
        self._input = 0
        self._output = 0

    def generate_outline(self, specification: BookSpecification) -> List[BeatBrief]:
        self._input += 5
        self._output += 20
        return self._outline

    def generate_beat_prose(
        self, brief: BeatBrief, preceding_context: str, corrective_note: Optional[str] = None
    ) -> str:
        self.prose_calls.append((brief.index, preceding_context, corrective_note))
        self._input += 10
        self._output += 50
        return self._prose_fn(brief.index, corrective_note)

    def consumed_tokens(self) -> TokenUsage:
        return TokenUsage(input_tokens=self._input, output_tokens=self._output)


class _FakeArcAnalyzer:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def execute(self, beats: List[NarrativeBeat]) -> NarrativeArc:
        self.calls.append([beat.text for beat in beats])
        coherence = [100.0 if _DRIFT_MARK in beat.text else 1.0 for beat in beats]
        points = [_point(beat) for beat in beats]
        return NarrativeArc(points=points, drift_series=[0.0] * (len(beats) - 1), coherence_series=coherence)


def _point(beat: NarrativeBeat) -> ArcPoint:
    return ArcPoint(
        beat=beat,
        colour=LabColor(l=50.0, a=0.0, b=0.0),
        histogram=ColoredDocument(histogram=np.array([1.0]), document_id=f"beat_{beat.index}"),
    )


def _outline(count: int) -> List[BeatBrief]:
    return [BeatBrief(index=index, title=f"Beat {index}", synopsis=f"Synopsis {index}") for index in range(count)]


def _clean_prose(index: int, corrective_note: Optional[str]) -> str:
    return f"Chapter {index} prose."


def _drift_until_corrected(target: int) -> Callable[[int, Optional[str]], str]:
    def prose_fn(index: int, corrective_note: Optional[str]) -> str:
        if index == target and corrective_note is None:
            return f"Chapter {index} {_DRIFT_MARK} prose."
        return f"Chapter {index} prose."

    return prose_fn


def _always_drift(target: int) -> Callable[[int, Optional[str]], str]:
    def prose_fn(index: int, corrective_note: Optional[str]) -> str:
        if index == target:
            return f"Chapter {index} {_DRIFT_MARK} prose."
        return f"Chapter {index} prose."

    return prose_fn


def _use_case(generator: TextGenerator, retry_budget: int = 2) -> GenerateBookUseCase:
    return GenerateBookUseCase(
        text_generator=generator,
        narrative_arc_analyzer=_FakeArcAnalyzer(),  # type: ignore[arg-type]
        drift_threshold=25.0,
        retry_budget=retry_budget,
    )


def _notes_for(generator: _StubTextGenerator, index: int) -> List[Optional[str]]:
    return [note for beat_index, _, note in generator.prose_calls if beat_index == index]


def _chapter_indices(book_chapters: List) -> List[int]:
    return [chapter.brief.index for chapter in book_chapters]


def test_should_generate_one_chapter_per_beat_in_reading_order() -> None:
    generator = _StubTextGenerator(_outline(3), _clean_prose)

    book = _use_case(generator).execute(_SPEC)

    assert _chapter_indices(book.chapters) == [0, 1, 2]


def test_should_not_pass_preceding_context_to_the_first_beat() -> None:
    generator = _StubTextGenerator(_outline(2), _clean_prose)

    _use_case(generator).execute(_SPEC)

    assert generator.prose_calls[0][1] == ""


def test_should_pass_the_previous_chapter_as_context_to_later_beats() -> None:
    generator = _StubTextGenerator(_outline(2), _clean_prose)

    _use_case(generator).execute(_SPEC)

    assert "Chapter 0 prose." in generator.prose_calls[1][1]


def test_should_regenerate_a_drifting_beat_with_a_corrective_note() -> None:
    generator = _StubTextGenerator(_outline(2), _drift_until_corrected(target=1))

    _use_case(generator).execute(_SPEC)

    assert any(note is not None for note in _notes_for(generator, index=1))


def test_should_count_a_recovered_beat_as_one_retry() -> None:
    generator = _StubTextGenerator(_outline(2), _drift_until_corrected(target=1))

    book = _use_case(generator).execute(_SPEC)

    assert book.metadata.total_retries == 1


def test_should_not_flag_a_beat_that_recovers_within_budget() -> None:
    generator = _StubTextGenerator(_outline(2), _drift_until_corrected(target=1))

    book = _use_case(generator).execute(_SPEC)

    assert book.metadata.flagged_beat_indices == []


def test_should_flag_a_beat_that_stays_over_threshold() -> None:
    generator = _StubTextGenerator(_outline(2), _always_drift(target=1))

    book = _use_case(generator, retry_budget=1).execute(_SPEC)

    assert 1 in book.metadata.flagged_beat_indices


def test_should_bound_retries_to_the_configured_budget() -> None:
    generator = _StubTextGenerator(_outline(2), _always_drift(target=1))

    book = _use_case(generator, retry_budget=2).execute(_SPEC)

    assert book.metadata.total_retries == 2


def test_should_accumulate_token_totals_in_metadata() -> None:
    generator = _StubTextGenerator(_outline(2), _clean_prose)

    book = _use_case(generator).execute(_SPEC)

    assert book.metadata.total_tokens.total_tokens == 145


def test_should_return_the_final_arc_over_all_chapters() -> None:
    generator = _StubTextGenerator(_outline(3), _clean_prose)

    book = _use_case(generator).execute(_SPEC)

    assert book.arc.beat_count == 3


def test_should_raise_when_the_outline_is_empty() -> None:
    generator = _StubTextGenerator([], _clean_prose)

    with pytest.raises(ValueError):
        _use_case(generator).execute(_SPEC)


class _L1Distance(DistanceCalculator):
    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        return float(np.sum(np.abs(doc1.histogram - doc2.histogram)))

    def metric_name(self) -> str:
        return "l1"


def _real_analyzer() -> AnalyzeNarrativeArcUseCase:
    embedder = Mock()
    embedder.encode.return_value = np.zeros(4, dtype=np.float32)
    embedder.encode_document_sentences.return_value = np.zeros((2, 4), dtype=np.float32)
    mapper = Mock(spec=ColorMapper)
    mapper.embed_to_lab.return_value = LabColor(l=50.0, a=0.0, b=0.0)
    mapper.embed_batch_to_lab.return_value = [LabColor(l=50.0, a=0.0, b=0.0)] * 2
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)
    return AnalyzeNarrativeArcUseCase(embedder, mapper, codebook, _L1Distance())


def test_should_produce_a_real_arc_through_the_narrative_compass() -> None:
    generator = _StubTextGenerator(_outline(2), _clean_prose)
    use_case = GenerateBookUseCase(
        text_generator=generator,
        narrative_arc_analyzer=_real_analyzer(),
        drift_threshold=1000.0,
        retry_budget=0,
    )

    book = use_case.execute(BookSpecification(summary="A quiet town.", num_beats=2, words_per_beat=200))

    assert book.arc.beat_count == 2
