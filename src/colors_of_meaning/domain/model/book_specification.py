from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ToneTarget:
    lightness: Optional[float] = None
    chroma: Optional[float] = None
    hue: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate_lightness()
        self._validate_chroma()
        self._validate_hue()

    def _validate_lightness(self) -> None:
        if self.lightness is not None and not 0.0 <= self.lightness <= 100.0:
            raise ValueError(f"lightness must be in [0, 100], got {self.lightness}")

    def _validate_chroma(self) -> None:
        if self.chroma is not None and self.chroma < 0.0:
            raise ValueError(f"chroma must be non-negative, got {self.chroma}")

    def _validate_hue(self) -> None:
        if self.hue is not None and not 0.0 <= self.hue < 360.0:
            raise ValueError(f"hue must be in [0, 360), got {self.hue}")


@dataclass(frozen=True)
class BeatBrief:
    index: int
    title: str
    synopsis: str
    target_tone: Optional[ToneTarget] = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")
        if not self.title:
            raise ValueError("title must not be empty")
        if not self.synopsis:
            raise ValueError("synopsis must not be empty")


@dataclass(frozen=True)
class BookSpecification:
    summary: str
    num_beats: int
    words_per_beat: int
    tone: Optional[str] = None
    beat_tone_targets: Optional[List[ToneTarget]] = None

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary must not be empty")
        if self.num_beats <= 0:
            raise ValueError(f"num_beats must be positive, got {self.num_beats}")
        if self.words_per_beat <= 0:
            raise ValueError(f"words_per_beat must be positive, got {self.words_per_beat}")
        self._validate_tone_targets()

    def _validate_tone_targets(self) -> None:
        if self.beat_tone_targets is not None and len(self.beat_tone_targets) != self.num_beats:
            raise ValueError(f"beat_tone_targets must have {self.num_beats} entries, got {len(self.beat_tone_targets)}")

    def tone_target_for(self, index: int) -> Optional[ToneTarget]:
        if self.beat_tone_targets is None or not 0 <= index < len(self.beat_tone_targets):
            return None
        return self.beat_tone_targets[index]
