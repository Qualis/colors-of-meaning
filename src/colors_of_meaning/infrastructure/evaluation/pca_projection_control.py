import logging
import uuid
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA  # type: ignore

from colors_of_meaning.domain.model.lab_color import LabColor

logger = logging.getLogger(__name__)

LAB_COMPONENTS = 3
LAB_AXIS_RANGES: Tuple[Tuple[float, float], ...] = ((0.0, 100.0), (-127.0, 127.0), (-127.0, 127.0))
DEGENERATE_AXIS_EPSILON = 1e-12
MINIMUM_RESCALABLE_EXTENT = 1e-150

FloatArray = npt.NDArray[np.float64]


def _axis_centres() -> FloatArray:
    return np.array([(low + high) / 2.0 for low, high in LAB_AXIS_RANGES], dtype=np.float64)


def _axis_half_extents() -> FloatArray:
    return np.array([(high - low) / 2.0 for low, high in LAB_AXIS_RANGES], dtype=np.float64)


def _as_lab_colors(coordinates: FloatArray) -> List[LabColor]:
    return [LabColor.from_unclamped(row[0], row[1], row[2]) for row in coordinates]


def rescale_preserving_ranks(coordinates: npt.NDArray) -> List[LabColor]:
    centred = np.asarray(coordinates, dtype=np.float64) - np.asarray(coordinates, dtype=np.float64).mean(axis=0)
    scale = _uniform_containment_scale(centred)
    return _as_lab_colors(centred * scale + _axis_centres())


def _uniform_containment_scale(centred: FloatArray) -> float:
    extents = np.abs(centred).max(axis=0)
    active = extents > 0.0
    if not bool(active.any()):
        return 1.0
    _reject_unrescalable_extent(float(extents[active].min()))
    return float((_axis_half_extents()[active] / extents[active]).min())


def _reject_unrescalable_extent(smallest_extent: float) -> None:
    if smallest_extent < MINIMUM_RESCALABLE_EXTENT:
        raise ValueError(
            f"coordinates span {smallest_extent} on one axis, which is too small to rescale into the Lab box "
            "without losing the rank order this projection promises to preserve"
        )


class PcaProjectionControl:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._pca = PCA(n_components=LAB_COMPONENTS, random_state=seed)
        self._minimums: Optional[FloatArray] = None
        self._ranges: Optional[FloatArray] = None

    def fit(self, embeddings: npt.NDArray) -> "PcaProjectionControl":
        components = self._pca.fit_transform(np.asarray(embeddings, dtype=np.float64))
        self._minimums = np.asarray(components.min(axis=0), dtype=np.float64)
        self._ranges = np.maximum(
            np.asarray(components.max(axis=0), dtype=np.float64) - self._minimums, DEGENERATE_AXIS_EPSILON
        )
        self._log_fit(len(components))
        return self

    def transform(self, embeddings: npt.NDArray) -> List[LabColor]:
        if self._minimums is None or self._ranges is None:
            raise ValueError("PcaProjectionControl must be fitted before transforming embeddings")
        components = self._pca.transform(np.asarray(embeddings, dtype=np.float64))
        unit_components = (np.asarray(components, dtype=np.float64) - self._minimums) / self._ranges
        return _as_lab_colors(self._to_lab_ranges(unit_components))

    @staticmethod
    def _to_lab_ranges(unit_components: FloatArray) -> FloatArray:
        lows = np.array([low for low, _ in LAB_AXIS_RANGES], dtype=np.float64)
        highs = np.array([high for _, high in LAB_AXIS_RANGES], dtype=np.float64)
        return np.asarray(np.clip(lows + unit_components * (highs - lows), lows, highs), dtype=np.float64)

    def _log_fit(self, sample_count: int) -> None:
        logger.info(
            "Fitted the PCA-3 projection control",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "seed": self._seed,
                "sample_count": sample_count,
                "explained_variance_ratio": [float(value) for value in self._pca.explained_variance_ratio_],
            },
        )
