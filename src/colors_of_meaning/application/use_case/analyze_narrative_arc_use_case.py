import logging
import uuid
from typing import Dict, List, Protocol, Sequence

import numpy as np
import numpy.typing as npt

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.narrative_arc import (
    ArcPoint,
    NarrativeArc,
    NarrativeBeat,
)
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator

logger = logging.getLogger(__name__)


class BeatEmbedder(Protocol):
    def encode(self, text: str) -> npt.NDArray:
        raise NotImplementedError

    def encode_document_sentences(self, document: str, batch_size: int = 32) -> npt.NDArray:
        raise NotImplementedError


class AnalyzeNarrativeArcUseCase:
    def __init__(
        self,
        embedding: BeatEmbedder,
        structured_mapper: ColorMapper,
        codebook: ColorCodebook,
        distance_calculator: DistanceCalculator,
    ) -> None:
        self._embedding = embedding
        self._mapper = structured_mapper
        self._distance = distance_calculator
        self._encode_document = EncodeDocumentUseCase(QuantizedColorMapper(structured_mapper, codebook))

    def execute(self, beats: Sequence[NarrativeBeat]) -> NarrativeArc:
        if not beats:
            raise ValueError("beats must not be empty")
        points = [self._build_point(beat) for beat in beats]
        palette = self._mix_palette([point.histogram for point in points])
        drift_series = self._drift_series(points)
        coherence_series = [self._distance.compute_distance(point.histogram, palette) for point in points]
        arc = NarrativeArc(points=points, drift_series=drift_series, coherence_series=coherence_series)
        self._log_summary(arc, palette)
        return arc

    def _build_point(self, beat: NarrativeBeat) -> ArcPoint:
        colour = self._mapper.embed_to_lab(self._embedding.encode(beat.text))
        sentence_embeddings = self._embedding.encode_document_sentences(beat.text)
        histogram = self._encode_document.execute(sentence_embeddings, document_id=f"beat_{beat.index}")
        return ArcPoint(beat=beat, colour=colour, histogram=histogram)

    def _drift_series(self, points: List[ArcPoint]) -> List[float]:
        return [
            self._distance.compute_distance(points[index].histogram, points[index + 1].histogram)
            for index in range(len(points) - 1)
        ]

    @staticmethod
    def _mix_palette(histograms: List[ColoredDocument]) -> ColoredDocument:
        mixture = np.asarray(np.mean([histogram.histogram for histogram in histograms], axis=0), dtype=np.float64)
        return ColoredDocument(histogram=mixture, document_id="whole_book_palette")

    def _log_summary(self, arc: NarrativeArc, palette: ColoredDocument) -> None:
        logger.info(
            "Analyzed narrative arc",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "beats": arc.beat_count,
                "per_beat_lchue": [
                    {"lightness": point.lightness, "chroma": point.chroma, "hue": point.hue_degrees}
                    for point in arc.points
                ],
                "drift_summary": _series_summary(arc.drift_series),
                "coherence_summary": _series_summary(arc.coherence_series),
                "palette_entropy": _entropy(palette.histogram),
            },
        )


def _series_summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _entropy(histogram: npt.NDArray) -> float:
    nonzero = histogram[histogram > 0]
    return float(-np.sum(nonzero * np.log(nonzero)))
