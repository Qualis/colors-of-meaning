from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LabColor:
    l: float  # noqa: E741
    a: float
    b: float

    def __post_init__(self) -> None:
        if not 0 <= self.l <= 100:
            raise ValueError(f"L must be in [0, 100], got {self.l}")
        if not -128 <= self.a <= 127:
            raise ValueError(f"a must be in [-128, 127], got {self.a}")
        if not -128 <= self.b <= 127:
            raise ValueError(f"b must be in [-128, 127], got {self.b}")

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.l, self.a, self.b)

    @classmethod
    def from_tuple(cls, lab_tuple: Tuple[float, float, float]) -> "LabColor":
        return cls(l=lab_tuple[0], a=lab_tuple[1], b=lab_tuple[2])

    @classmethod
    def from_unclamped(cls, lightness: float, a_axis: float, b_axis: float) -> "LabColor":
        return cls(
            l=float(max(0.0, min(100.0, lightness))),
            a=float(max(-128.0, min(127.0, a_axis))),
            b=float(max(-128.0, min(127.0, b_axis))),
        )

    def clamp(self) -> "LabColor":
        return LabColor.from_unclamped(self.l, self.a, self.b)
