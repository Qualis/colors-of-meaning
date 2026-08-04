from dataclasses import dataclass
from functools import cached_property
from typing import List
import numpy as np
import numpy.typing as npt

from colors_of_meaning.domain.model.lab_color import LabColor

QUANTIZE_CHUNK_SIZE = 512


@dataclass(frozen=True)
class ColorCodebook:
    colors: List[LabColor]
    num_bins: int

    def __post_init__(self) -> None:
        if len(self.colors) != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} colors, got {len(self.colors)}")
        if self.num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {self.num_bins}")

    @cached_property
    def _palette_coordinates(self) -> npt.NDArray:
        return np.array([[color.l, color.a, color.b] for color in self.colors], dtype=np.float64)

    def quantize(self, color: LabColor) -> int:
        return int(self.quantize_batch([color])[0])

    def quantize_batch(self, colors: List[LabColor]) -> npt.NDArray[np.int64]:
        queries = np.array([[color.l, color.a, color.b] for color in colors], dtype=np.float64).reshape(-1, 3)
        chunks = [
            self._nearest_palette_indices(queries[start : start + QUANTIZE_CHUNK_SIZE])
            for start in range(0, len(queries), QUANTIZE_CHUNK_SIZE)
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    def _nearest_palette_indices(self, queries: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        squared_distances: npt.NDArray[np.float64] = np.sum(
            (self._palette_coordinates[None, :, :] - queries[:, None, :]) ** 2, axis=2
        )
        return np.asarray(np.argmin(squared_distances, axis=1), dtype=np.int64)

    def get_color(self, bin_index: int) -> LabColor:
        if not 0 <= bin_index < self.num_bins:
            raise ValueError(f"bin_index must be in [0, {self.num_bins}), got {bin_index}")
        return self.colors[bin_index]

    @classmethod
    def create_uniform_grid(cls, bins_per_dimension: int = 16) -> "ColorCodebook":
        num_bins = bins_per_dimension**3
        colors = []

        l_values = np.linspace(0, 100, bins_per_dimension)
        a_values = np.linspace(-128, 127, bins_per_dimension)
        b_values = np.linspace(-128, 127, bins_per_dimension)

        for lightness in l_values:
            for a_val in a_values:
                for b_val in b_values:
                    colors.append(LabColor(l=float(lightness), a=float(a_val), b=float(b_val)))

        return cls(colors=colors, num_bins=num_bins)
