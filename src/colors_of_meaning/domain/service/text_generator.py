from abc import ABC, abstractmethod
from typing import List, Optional

from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.generated_book import TokenUsage


class GenerationError(Exception):
    pass


class TextGenerator(ABC):
    @abstractmethod
    def generate_outline(self, specification: BookSpecification) -> List[BeatBrief]:
        raise NotImplementedError

    @abstractmethod
    def generate_beat_prose(
        self,
        brief: BeatBrief,
        preceding_context: str,
        corrective_note: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def consumed_tokens(self) -> TokenUsage:
        raise NotImplementedError
