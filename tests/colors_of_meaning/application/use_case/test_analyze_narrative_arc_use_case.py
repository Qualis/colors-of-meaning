from typing import List, Optional
from unittest.mock import Mock

import numpy as np
import pytest

from colors_of_meaning.application.use_case.analyze_narrative_arc_use_case import (
    AnalyzeNarrativeArcUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.narrative_arc import NarrativeBeat
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)


class _L1Distance(DistanceCalculator):
    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        return float(np.sum(np.abs(doc1.histogram - doc2.histogram)))

    def metric_name(self) -> str:
        return "l1"


def _beats(count: int) -> List[NarrativeBeat]:
    return [NarrativeBeat(index=index, title=f"Beat {index}", text="a long enough beat body") for index in range(count)]


_DARK = LabColor(l=0.0, a=-128.0, b=-128.0)
_BRIGHT = LabColor(l=100.0, a=127.0, b=127.0)
_MID = LabColor(l=50.0, a=10.0, b=0.0)


def _distinct_beat_colours() -> List[List[LabColor]]:
    return [[_DARK, _DARK], [_BRIGHT, _BRIGHT], [_DARK, _BRIGHT]]


def _use_case(
    batch_returns: List[List[LabColor]],
    document_colour: LabColor = _MID,
    distance: Optional[DistanceCalculator] = None,
) -> AnalyzeNarrativeArcUseCase:
    embedder = Mock()
    embedder.encode.return_value = np.zeros(4, dtype=np.float32)
    embedder.encode_document_sentences.return_value = np.zeros((2, 4), dtype=np.float32)
    mapper = Mock(spec=ColorMapper)
    mapper.embed_to_lab.return_value = document_colour
    mapper.embed_batch_to_lab.side_effect = list(batch_returns)
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)
    return AnalyzeNarrativeArcUseCase(embedder, mapper, codebook, distance or _L1Distance())


def _mixed_palette(arc) -> ColoredDocument:
    mixture = np.mean([point.histogram.histogram for point in arc.points], axis=0)
    return ColoredDocument(histogram=np.asarray(mixture, dtype=np.float64))


class TestArcStructure:
    def test_should_return_one_point_per_beat(self) -> None:
        arc = _use_case(_distinct_beat_colours()).execute(_beats(3))

        assert arc.beat_count == 3

    def test_should_preserve_beat_order_in_points(self) -> None:
        arc = _use_case(_distinct_beat_colours()).execute(_beats(3))

        assert [point.beat.index for point in arc.points] == [0, 1, 2]

    def test_should_produce_drift_series_of_length_n_minus_one(self) -> None:
        arc = _use_case(_distinct_beat_colours()).execute(_beats(3))

        assert len(arc.drift_series) == 2

    def test_should_produce_coherence_series_of_length_n(self) -> None:
        arc = _use_case(_distinct_beat_colours()).execute(_beats(3))

        assert len(arc.coherence_series) == 3

    def test_should_raise_when_beats_is_empty(self) -> None:
        with pytest.raises(ValueError, match="beats must not be empty"):
            _use_case([]).execute([])

    def test_should_handle_a_single_beat_with_empty_drift(self) -> None:
        arc = _use_case([[_DARK, _DARK]]).execute(_beats(1))

        assert arc.drift_series == []


class TestDriftAndCoherenceComeFromThePort:
    def test_should_measure_drift_between_consecutive_beat_histograms(self) -> None:
        arc = _use_case(_distinct_beat_colours()).execute(_beats(3))

        expected = _L1Distance().compute_distance(arc.points[0].histogram, arc.points[1].histogram)
        assert arc.drift_series[0] == pytest.approx(expected)

    def test_should_measure_coherence_against_the_whole_book_palette(self) -> None:
        arc = _use_case(_distinct_beat_colours()).execute(_beats(3))

        expected = _L1Distance().compute_distance(arc.points[0].histogram, _mixed_palette(arc))
        assert arc.coherence_series[0] == pytest.approx(expected)


class TestAntiAggregationTrap:
    def test_should_read_beat_colour_from_the_document_embedding_not_the_sentence_mean(self) -> None:
        use_case = _use_case(
            batch_returns=[[LabColor(l=50.0, a=40.0, b=0.0), LabColor(l=50.0, a=-40.0, b=0.0)]],
            document_colour=LabColor(l=50.0, a=40.0, b=30.0),
        )

        arc = use_case.execute(_beats(1))

        assert arc.points[0].chroma == pytest.approx(50.0)


class _StubEmbedder:
    def encode(self, text: str) -> np.ndarray:
        return np.full(384, float(len(text) % 5) + 1.0, dtype=np.float32)

    def encode_document_sentences(self, document: str, batch_size: int = 32) -> np.ndarray:
        base = float(len(document) % 3) + 1.0
        return np.stack([np.full(384, base, dtype=np.float32), np.full(384, base + 0.5, dtype=np.float32)])


def _seeded_use_case() -> AnalyzeNarrativeArcUseCase:
    mapper = StructuredPyTorchColorMapper(input_dim=384, device="cpu", seed=42)
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)
    distance = SlicedWassersteinDistanceCalculator(codebook=codebook, seed=42)
    return AnalyzeNarrativeArcUseCase(_StubEmbedder(), mapper, codebook, distance)


def _determinism_beats() -> List[NarrativeBeat]:
    texts = ["the harbour is quiet at dusk", "a storm breaks over the reef tonight now", "she reaches the far city"]
    return [NarrativeBeat(index=index, title=f"b{index}", text=text) for index, text in enumerate(texts)]


class TestDeterminism:
    def test_should_produce_identical_coherence_across_two_seeded_runs(self) -> None:
        beats = _determinism_beats()

        first = _seeded_use_case().execute(beats)
        second = _seeded_use_case().execute(beats)

        assert first.coherence_series == second.coherence_series

    def test_should_produce_identical_drift_across_two_seeded_runs(self) -> None:
        beats = _determinism_beats()

        first = _seeded_use_case().execute(beats)
        second = _seeded_use_case().execute(beats)

        assert first.drift_series == second.drift_series

    def test_should_produce_identical_colours_across_two_seeded_runs(self) -> None:
        beats = _determinism_beats()

        first = _seeded_use_case().execute(beats)
        second = _seeded_use_case().execute(beats)

        assert [colour.to_tuple() for colour in first.colours] == [colour.to_tuple() for colour in second.colours]
