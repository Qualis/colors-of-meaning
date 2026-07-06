import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from colors_of_meaning.application.use_case.analyze_narrative_arc_use_case import (
    AnalyzeNarrativeArcUseCase,
)
from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.generated_book import (
    GeneratedBook,
    GeneratedChapter,
    GenerationMetadata,
    TokenUsage,
)
from colors_of_meaning.domain.model.narrative_arc import NarrativeBeat
from colors_of_meaning.domain.service.text_generator import TextGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _BeatOutcome:
    beat: NarrativeBeat
    coherence: float
    retries: int
    flagged: bool


@dataclass
class _ChapterAccumulator:
    chapters: List[GeneratedChapter]
    beats: List[NarrativeBeat]
    flagged_indices: List[int]
    total_retries: int
    preceding_context: str


class GenerateBookUseCase:
    def __init__(
        self,
        text_generator: TextGenerator,
        narrative_arc_analyzer: AnalyzeNarrativeArcUseCase,
        drift_threshold: float,
        retry_budget: int,
    ) -> None:
        self._text_generator = text_generator
        self._narrative_arc_analyzer = narrative_arc_analyzer
        self._drift_threshold = drift_threshold
        self._retry_budget = retry_budget

    def execute(self, specification: BookSpecification) -> GeneratedBook:
        correlation_id = str(uuid.uuid4())
        outline = self._text_generator.generate_outline(specification)
        if not outline:
            raise ValueError("generated outline must contain at least one beat")
        self._log_outline(correlation_id, len(outline))
        accumulator = self._write_chapters(outline, correlation_id)
        arc = self._narrative_arc_analyzer.execute(accumulator.beats)
        metadata = GenerationMetadata(
            total_tokens=self._text_generator.consumed_tokens(),
            total_retries=accumulator.total_retries,
            flagged_beat_indices=accumulator.flagged_indices,
        )
        book = GeneratedBook(
            specification=specification,
            chapters=accumulator.chapters,
            arc=arc,
            metadata=metadata,
        )
        self._log_summary(correlation_id, book)
        return book

    def _write_chapters(self, outline: List[BeatBrief], correlation_id: str) -> _ChapterAccumulator:
        accumulator = _ChapterAccumulator(
            chapters=[], beats=[], flagged_indices=[], total_retries=0, preceding_context=""
        )
        for brief in outline:
            outcome = self._generate_beat(brief, accumulator.beats, accumulator.preceding_context, correlation_id)
            self._record_beat(accumulator, brief, outcome)
        return accumulator

    @staticmethod
    def _record_beat(accumulator: _ChapterAccumulator, brief: BeatBrief, outcome: _BeatOutcome) -> None:
        accumulator.beats.append(outcome.beat)
        accumulator.chapters.append(GeneratedChapter(brief=brief, prose=outcome.beat.text))
        accumulator.preceding_context = _extend_context(accumulator.preceding_context, brief.title, outcome.beat.text)
        accumulator.total_retries += outcome.retries
        if outcome.flagged:
            accumulator.flagged_indices.append(brief.index)

    def _generate_beat(
        self,
        brief: BeatBrief,
        prior_beats: List[NarrativeBeat],
        preceding_context: str,
        correlation_id: str,
    ) -> _BeatOutcome:
        tokens_before = self._text_generator.consumed_tokens()
        beat, coherence = self._draft(brief, prior_beats, preceding_context, None)
        best = (beat, coherence)
        attempt = 0
        while coherence > self._drift_threshold and attempt < self._retry_budget:
            attempt += 1
            beat, coherence = self._draft(brief, prior_beats, preceding_context, self._corrective_note(coherence))
            best = _lower_coherence(best, beat, coherence)
        outcome = self._finalize(best, attempt)
        self._log_beat(correlation_id, brief.index, tokens_before, outcome)
        return outcome

    def _draft(
        self,
        brief: BeatBrief,
        prior_beats: List[NarrativeBeat],
        preceding_context: str,
        corrective_note: Optional[str],
    ) -> Tuple[NarrativeBeat, float]:
        prose = self._text_generator.generate_beat_prose(brief, preceding_context, corrective_note)
        beat = NarrativeBeat(index=brief.index, title=brief.title, text=prose)
        beats_so_far = [*prior_beats, beat]
        arc = self._narrative_arc_analyzer.execute(beats_so_far)
        return beat, arc.coherence_series[-1]

    def _finalize(self, best: Tuple[NarrativeBeat, float], attempt: int) -> _BeatOutcome:
        beat, coherence = best
        return _BeatOutcome(
            beat=beat,
            coherence=coherence,
            retries=attempt,
            flagged=coherence > self._drift_threshold,
        )

    def _corrective_note(self, coherence: float) -> str:
        return (
            "The previous draft of this chapter drifted from the book's established tone and topic "
            f"(coherence {coherence:.2f} exceeded the threshold {self._drift_threshold:.2f}). "
            "Stay closer to the palette of the preceding chapters: keep the same overall mood, "
            "level of concreteness, and subject matter."
        )

    def _log_outline(self, correlation_id: str, outline_size: int) -> None:
        logger.info(
            "Planned book outline",
            extra={"correlation_id": correlation_id, "outline_size": outline_size},
        )

    def _log_beat(self, correlation_id: str, beat_index: int, tokens_before: TokenUsage, outcome: _BeatOutcome) -> None:
        after = self._text_generator.consumed_tokens()
        logger.info(
            "Generated book beat",
            extra={
                "correlation_id": correlation_id,
                "beat_index": beat_index,
                "tokens_in": after.input_tokens - tokens_before.input_tokens,
                "tokens_out": after.output_tokens - tokens_before.output_tokens,
                "retries": outcome.retries,
                "coherence": outcome.coherence,
                "flagged": outcome.flagged,
            },
        )

    def _log_summary(self, correlation_id: str, book: GeneratedBook) -> None:
        logger.info(
            "Completed book generation",
            extra={
                "correlation_id": correlation_id,
                "chapters": book.chapter_count,
                "total_tokens": book.metadata.total_tokens.total_tokens,
                "total_retries": book.metadata.total_retries,
                "flagged_beats": book.metadata.flagged_beat_indices,
            },
        )


def _lower_coherence(
    best: Tuple[NarrativeBeat, float], beat: NarrativeBeat, coherence: float
) -> Tuple[NarrativeBeat, float]:
    if coherence < best[1]:
        return (beat, coherence)
    return best


def _extend_context(preceding_context: str, title: str, prose: str) -> str:
    chapter = f"## {title}\n{prose}"
    if not preceding_context:
        return chapter
    return f"{preceding_context}\n\n{chapter}"
