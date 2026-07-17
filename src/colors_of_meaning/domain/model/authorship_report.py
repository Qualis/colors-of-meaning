from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class AuthorshipHeldOut:
    method: str
    bits_per_token: Optional[float]
    accuracy: float
    macro_f1: float


@dataclass(frozen=True)
class AuthorshipScalingPoint:
    paragraphs_per_work: int
    train_paragraphs: int
    mean_accuracy: float
    std_accuracy: float

    def times_chance(self, chance: float) -> float:
        return self.mean_accuracy / chance


@dataclass(frozen=True)
class AuthorshipCorpus:
    authors_works: List[Tuple[str, int]]
    split_counts: List[Tuple[str, int]]

    def author_count(self) -> int:
        return len(self.authors_works)

    def total_works(self) -> int:
        return sum(works for _author, works in self.authors_works)


@dataclass(frozen=True)
class AuthorshipReport:
    corpus: AuthorshipCorpus
    held_out: List[AuthorshipHeldOut]
    chance: float
    scaling: List[AuthorshipScalingPoint]
