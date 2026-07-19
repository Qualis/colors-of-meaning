from typing import List, Optional

import numpy as np
import pytest

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_grounding_use_case import (
    EvaluateGroundingUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
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


class _CapturingDistance(DistanceCalculator):
    def __init__(self) -> None:
        self.captured_context: Optional[ColoredDocument] = None

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        self.captured_context = doc2
        return 0.0

    def metric_name(self) -> str:
        return "capturing"


def _histogram(*values: float) -> ColoredDocument:
    array = np.array(values, dtype=np.float64)
    return ColoredDocument(histogram=array / array.sum())


class TestDivergenceComesFromThePort:
    def test_should_return_the_port_distance_as_divergence(self) -> None:
        answer = _histogram(1.0, 0.0, 0.0, 0.0)
        context = _histogram(0.0, 1.0, 0.0, 0.0)

        report = EvaluateGroundingUseCase(_L1Distance()).execute(answer, context, threshold=1.0)

        assert report.divergence == pytest.approx(_L1Distance().compute_distance(answer, context))


class TestVerdictFollowsThreshold:
    def test_should_be_grounded_when_divergence_within_threshold(self) -> None:
        answer = _histogram(1.0, 0.0, 0.0, 0.0)

        report = EvaluateGroundingUseCase(_L1Distance()).execute(answer, answer, threshold=0.5)

        assert report.is_grounded is True

    def test_should_not_be_grounded_when_divergence_exceeds_threshold(self) -> None:
        answer = _histogram(1.0, 0.0, 0.0, 0.0)
        context = _histogram(0.0, 1.0, 0.0, 0.0)

        report = EvaluateGroundingUseCase(_L1Distance()).execute(answer, context, threshold=0.5)

        assert report.is_grounded is False


class TestUniformMixturePalette:
    def test_should_compare_the_answer_against_the_uniform_mixture_of_passages(self) -> None:
        distance = _CapturingDistance()
        answer = _histogram(1.0, 0.0, 0.0, 0.0)
        passages = [_histogram(1.0, 0.0, 0.0, 0.0), _histogram(0.0, 0.0, 0.0, 1.0)]

        EvaluateGroundingUseCase(distance).execute_against_contexts(answer, passages, threshold=1.0)

        expected_mixture = np.mean([passage.histogram for passage in passages], axis=0)
        assert np.allclose(distance.captured_context.histogram, expected_mixture)

    def test_should_raise_when_contexts_is_empty(self) -> None:
        answer = _histogram(1.0, 0.0, 0.0, 0.0)

        with pytest.raises(ValueError, match="contexts must not be empty"):
            EvaluateGroundingUseCase(_L1Distance()).execute_against_contexts(answer, [], threshold=1.0)


class _KeyedEmbedder:
    _SENTENCES = {
        "grounded": np.stack([np.full(384, 1.0, dtype=np.float32), np.full(384, 1.0, dtype=np.float32)]),
        "off_topic": np.stack([np.full(384, -3.0, dtype=np.float32), np.full(384, 3.0, dtype=np.float32)]),
    }

    def encode_document_sentences(self, document: str, batch_size: int = 32) -> np.ndarray:
        return self._SENTENCES[document]


def _encoded(embedder: _KeyedEmbedder, encode_use_case: EncodeDocumentUseCase, key: str) -> ColoredDocument:
    return encode_use_case.execute(embedder.encode_document_sentences(key), document_id=key)


def _seeded_grounding_setup() -> EncodeDocumentUseCase:
    mapper = StructuredPyTorchColorMapper(input_dim=384, device="cpu", seed=42)
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)
    return EncodeDocumentUseCase(QuantizedColorMapper(mapper, codebook))


def _seeded_distance() -> SlicedWassersteinDistanceCalculator:
    return SlicedWassersteinDistanceCalculator(
        codebook=ColorCodebook.create_uniform_grid(bins_per_dimension=4), seed=42
    )


def _grounding_divergences() -> List[float]:
    embedder = _KeyedEmbedder()
    encode_use_case = _seeded_grounding_setup()
    context = _encoded(embedder, encode_use_case, "grounded")
    use_case = EvaluateGroundingUseCase(_seeded_distance())
    grounded = use_case.execute_against_contexts(_encoded(embedder, encode_use_case, "grounded"), [context], 100.0)
    off_topic = use_case.execute_against_contexts(_encoded(embedder, encode_use_case, "off_topic"), [context], 100.0)
    return [grounded.divergence, off_topic.divergence]


class TestDirectionalFalsifiability:
    def test_should_score_the_same_topic_answer_below_the_off_topic_answer(self) -> None:
        grounded_divergence, off_topic_divergence = _grounding_divergences()

        assert grounded_divergence < off_topic_divergence

    def test_should_produce_identical_divergence_across_two_seeded_runs(self) -> None:
        first = _grounding_divergences()
        second = _grounding_divergences()

        assert first == second
