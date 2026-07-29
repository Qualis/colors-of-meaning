import math
from dataclasses import dataclass
from typing import List

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor


@dataclass(frozen=True)
class NarrativeBeat:
    index: int
    title: str
    text: str

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be non-negative, got {self.index}")


@dataclass(frozen=True)
class ArcPoint:
    beat: NarrativeBeat
    colour: LabColor
    histogram: ColoredDocument

    @property
    def lightness(self) -> float:
        return self.colour.l

    @property
    def chroma(self) -> float:
        return math.hypot(self.colour.a, self.colour.b)

    @property
    def hue_degrees(self) -> float:
        return math.degrees(math.atan2(self.colour.b, self.colour.a)) % 360.0


@dataclass(frozen=True)
class FlaggedBeat:
    index: int
    coherence: float


@dataclass(frozen=True)
class NarrativeArc:
    points: List[ArcPoint]
    drift_series: List[float]
    coherence_series: List[float]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("NarrativeArc requires at least one point")
        if len(self.drift_series) != len(self.points) - 1:
            raise ValueError(f"drift_series must have {len(self.points) - 1} values, got {len(self.drift_series)}")
        if len(self.coherence_series) != len(self.points):
            raise ValueError(f"coherence_series must have {len(self.points)} values, got {len(self.coherence_series)}")

    @property
    def beat_count(self) -> int:
        return len(self.points)

    @property
    def lightness_series(self) -> List[float]:
        return [point.lightness for point in self.points]

    @property
    def chroma_series(self) -> List[float]:
        return [point.chroma for point in self.points]

    @property
    def hue_series(self) -> List[float]:
        return [point.hue_degrees for point in self.points]

    @property
    def colours(self) -> List[LabColor]:
        return [point.colour for point in self.points]

    def flagged_beats(self, threshold: float) -> List[FlaggedBeat]:
        return [
            FlaggedBeat(index=point.beat.index, coherence=coherence)
            for point, coherence in zip(self.points, self.coherence_series, strict=True)
            if coherence > threshold
        ]
