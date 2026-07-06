from dataclasses import dataclass
from typing import List

from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.narrative_arc import NarrativeArc


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if self.input_tokens < 0:
            raise ValueError(f"input_tokens must be non-negative, got {self.input_tokens}")
        if self.output_tokens < 0:
            raise ValueError(f"output_tokens must be non-negative, got {self.output_tokens}")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class GeneratedChapter:
    brief: BeatBrief
    prose: str

    def __post_init__(self) -> None:
        if not self.prose.strip():
            raise ValueError("prose must not be empty")


@dataclass(frozen=True)
class GenerationMetadata:
    total_tokens: TokenUsage
    total_retries: int
    flagged_beat_indices: List[int]

    def __post_init__(self) -> None:
        if self.total_retries < 0:
            raise ValueError(f"total_retries must be non-negative, got {self.total_retries}")


@dataclass(frozen=True)
class GeneratedBook:
    specification: BookSpecification
    chapters: List[GeneratedChapter]
    arc: NarrativeArc
    metadata: GenerationMetadata

    def __post_init__(self) -> None:
        if not self.chapters:
            raise ValueError("GeneratedBook requires at least one chapter")
        if len(self.chapters) != self.arc.beat_count:
            raise ValueError(f"chapters ({len(self.chapters)}) must match arc beats ({self.arc.beat_count})")

    @property
    def chapter_count(self) -> int:
        return len(self.chapters)
