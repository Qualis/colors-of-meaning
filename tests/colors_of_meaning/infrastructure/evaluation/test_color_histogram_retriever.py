from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import pytest
from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.evaluation.color_histogram_retriever import (
    ColorHistogramRetriever,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_retrieval_core import (
    ColorHistogramRetrievalCore,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import WassersteinDistanceCalculator
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import SklearnMetricsCalculator
from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import RetrievalEvaluateUseCase
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor


class _StructurePreservingColorMapper(ColorMapper):
    def embed_to_lab(self, embedding: npt.NDArray) -> LabColor:
        return self.embed_batch_to_lab(np.asarray(embedding).reshape(1, -1))[0]

    def embed_batch_to_lab(self, embeddings: npt.NDArray) -> List[LabColor]:
        rows = np.asarray(embeddings, dtype=np.float64)
        return [
            LabColor(
                l=float(np.clip(50.0 + 4.0 * row[2], 0.0, 100.0)),
                a=float(np.clip(30.0 * (row[0] - row[1]), -127.0, 127.0)),
                b=float(np.clip(30.0 * (row[1] - row[0]), -127.0, 127.0)),
            )
            for row in rows
        ]

    def train(self, embeddings: npt.NDArray, epochs: int, learning_rate: float) -> None:
        return None

    def epoch_checkpoints(self) -> List[Any]:
        return []

    def restore_checkpoint(self, checkpoint: Any) -> None:
        return None

    def save_weights(self, path: str) -> None:
        return None

    def load_weights(self, path: str) -> None:
        return None


def _mocked_retriever(distance_side_effect: list | None = None) -> ColorHistogramRetriever:
    embedding_adapter = Mock()
    embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]
    encode_use_case = Mock()
    encode_use_case.execute.return_value = ColoredDocument(
        histogram=np.array([0.25, 0.25, 0.25, 0.25]), document_id="doc"
    )
    distance_calculator = Mock()
    if distance_side_effect is None:
        distance_calculator.compute_distance.return_value = 0.5
    else:
        distance_calculator.compute_distance.side_effect = distance_side_effect
    return ColorHistogramRetriever(embedding_adapter, encode_use_case, distance_calculator)


def _train_samples() -> list:
    return [
        EvaluationSample(text="alpha", label=0, split="train"),
        EvaluationSample(text="beta", label=1, split="train"),
        EvaluationSample(text="gamma", label=0, split="train"),
        EvaluationSample(text="delta", label=1, split="train"),
    ]


def _query() -> EvaluationSample:
    return EvaluationSample(text="query", label=0, split="test")


def _axis_embeddings(texts: list, seed: int) -> Dict[str, npt.NDArray]:
    rng = np.random.default_rng(seed)
    dimension = 16
    embeddings: Dict[str, npt.NDArray] = {}
    for axis, text in enumerate(texts):
        base = np.zeros(dimension, dtype=np.float32)
        base[axis] = 4.0
        embeddings[text] = (base + rng.standard_normal(dimension).astype(np.float32) * 0.05).reshape(1, dimension)
    return embeddings


def _fitted_real_retriever(embedding_by_text: Dict[str, npt.NDArray]) -> ColorHistogramRetriever:
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)
    quantized_mapper = QuantizedColorMapper(color_mapper=_StructurePreservingColorMapper(), codebook=codebook)
    encode_use_case = EncodeDocumentUseCase(quantized_mapper=quantized_mapper)
    distance_calculator = WassersteinDistanceCalculator(codebook=codebook)
    adapter = Mock()
    adapter.encode_document_sentences.side_effect = lambda text: embedding_by_text[text]
    return ColorHistogramRetriever(adapter, encode_use_case, distance_calculator)


class TestColorHistogramRetriever:
    @patch("hnswlib.Index")
    def test_should_return_ranked_samples_in_non_decreasing_distance_order(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        retriever = _mocked_retriever(distance_side_effect=[0.4, 0.1, 0.3, 0.2])
        retriever.fit(_train_samples())

        results = retriever.search(_query(), k=4)

        assert [distance for _, distance in results] == [0.1, 0.2, 0.3, 0.4]

    @patch("hnswlib.Index")
    def test_should_return_k_pairs_when_k_is_below_corpus_size(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        retriever = _mocked_retriever()
        retriever.fit(_train_samples())

        results = retriever.search(_query(), k=2)

        assert len(results) == 2

    @patch("hnswlib.Index")
    def test_should_return_corpus_size_pairs_when_k_exceeds_corpus_size(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        retriever = _mocked_retriever()
        retriever.fit(_train_samples())

        results = retriever.search(_query(), k=10)

        assert len(results) == 4

    @patch("hnswlib.Index")
    def test_should_map_ranked_result_to_its_training_sample(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        retriever = _mocked_retriever(distance_side_effect=[0.4, 0.1, 0.3, 0.2])
        retriever.fit(_train_samples())

        results = retriever.search(_query(), k=4)

        assert results[0][0].text == "beta"

    @patch("hnswlib.Index")
    def test_should_return_empty_results_when_k_is_not_positive(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        retriever = _mocked_retriever()
        retriever.fit(_train_samples())

        assert retriever.search(_query(), k=0) == []

    def test_should_raise_error_when_searching_before_fit(self) -> None:
        retriever = _mocked_retriever()

        with pytest.raises(RuntimeError, match="must be fitted before search"):
            retriever.search(_query(), k=3)

    def test_should_retrieve_the_query_document_first_when_query_matches_a_training_document(self) -> None:
        texts = ["alpha", "beta", "gamma", "delta"]
        embedding_by_text = _axis_embeddings(texts, seed=7)
        retriever = _fitted_real_retriever(embedding_by_text)
        retriever.fit([EvaluationSample(text=text, label=index % 2, split="train") for index, text in enumerate(texts)])

        results = retriever.search(EvaluationSample(text="alpha", label=0, split="test"), k=1)

        assert results[0][0].text == "alpha"

    def test_should_retrieve_identical_query_at_near_zero_distance(self) -> None:
        texts = ["alpha", "beta", "gamma", "delta"]
        embedding_by_text = _axis_embeddings(texts, seed=7)
        retriever = _fitted_real_retriever(embedding_by_text)
        retriever.fit([EvaluationSample(text=text, label=index % 2, split="train") for index, text in enumerate(texts)])

        results = retriever.search(EvaluationSample(text="alpha", label=0, split="test"), k=1)

        assert results[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_should_delegate_to_a_shared_retrieval_core(self) -> None:
        retriever = _mocked_retriever()

        assert isinstance(retriever.core, ColorHistogramRetrievalCore)


def _class_separated_corpus() -> tuple:
    rng = np.random.default_rng(123)
    dimension = 16
    embedding_by_text: Dict[str, npt.NDArray] = {}
    train_samples: List[EvaluationSample] = []
    test_samples: List[EvaluationSample] = []
    for label in (0, 1):
        axis = np.zeros(dimension, dtype=np.float32)
        axis[label] = 4.0
        for index in range(6):
            text = f"train_{label}_{index}"
            embedding_by_text[text] = (axis + rng.standard_normal(dimension).astype(np.float32) * 0.1).reshape(
                1, dimension
            )
            train_samples.append(EvaluationSample(text=text, label=label, split="train"))
        for index in range(3):
            text = f"test_{label}_{index}"
            embedding_by_text[text] = (axis + rng.standard_normal(dimension).astype(np.float32) * 0.1).reshape(
                1, dimension
            )
            test_samples.append(EvaluationSample(text=text, label=label, split="test"))
    return train_samples, test_samples, embedding_by_text


class TestColorRetrievalStack:
    def test_should_measure_non_zero_mrr_through_the_retrieval_use_case(self) -> None:
        train_samples, test_samples, embedding_by_text = _class_separated_corpus()
        retriever = _fitted_real_retriever(embedding_by_text)
        repository = Mock()
        repository.get_samples.side_effect = [train_samples, test_samples]
        use_case = RetrievalEvaluateUseCase(retriever, SklearnMetricsCalculator(), repository, [1, 5])

        evaluation = use_case.execute()

        assert evaluation.result.mrr > 0.0
