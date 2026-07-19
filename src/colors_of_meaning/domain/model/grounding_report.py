from dataclasses import dataclass


@dataclass(frozen=True)
class GroundingReport:
    divergence: float
    threshold: float

    def __post_init__(self) -> None:
        self._validate_non_negative("divergence", self.divergence)
        self._validate_non_negative("threshold", self.threshold)

    @property
    def is_grounded(self) -> bool:
        return self.divergence <= self.threshold

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")
