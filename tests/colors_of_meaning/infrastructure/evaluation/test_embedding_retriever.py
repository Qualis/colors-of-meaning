import numpy as np
import pytest
from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.evaluation.embedding_retriever import EmbeddingRetriever
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


def _adapter() -> Mock:
    adapter = Mock()
    adapter.encode_batch.return_value = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9],
    ]
    return adapter


def _train_samples() -> list:
    return [
        EvaluationSample(text="alpha", label=0, split="train"),
        EvaluationSample(text="beta", label=1, split="train"),
        EvaluationSample(text="gamma", label=0, split="train"),
        EvaluationSample(text="delta", label=1, split="train"),
    ]


def _query() -> EvaluationSample:
    return EvaluationSample(text="query", label=0, split="test")


class TestEmbeddingRetriever:
    @patch("hnswlib.Index")
    def test_should_build_l2_index_when_fitting(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        retriever = EmbeddingRetriever(_adapter())

        retriever.fit(_train_samples())

        mock_index_class.assert_called_once_with(space="l2", dim=4)

    @patch("hnswlib.Index")
    def test_should_store_embedding_dimension_when_fitting(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        retriever = EmbeddingRetriever(_adapter())

        retriever.fit(_train_samples())

        assert retriever.dimension == 4

    @patch("hnswlib.Index")
    def test_should_pin_single_thread_when_building_index(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        retriever = EmbeddingRetriever(_adapter())

        retriever.fit(_train_samples())

        mock_index.set_num_threads.assert_called_once_with(1)

    @patch("hnswlib.Index")
    def test_should_return_ranked_samples_with_l2_distances(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 2, 1]]), np.array([[0.1, 0.2, 0.3]]))
        adapter = _adapter()
        retriever = EmbeddingRetriever(adapter)
        retriever.fit(_train_samples())
        adapter.encode_batch.return_value = [[0.15, 0.25, 0.35, 0.45]]

        results = retriever.search(_query(), k=3)

        assert [(sample.text, distance) for sample, distance in results] == [
            ("alpha", 0.1),
            ("gamma", 0.2),
            ("beta", 0.3),
        ]

    @patch("hnswlib.Index")
    def test_should_query_min_of_k_and_corpus_size(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        adapter = _adapter()
        retriever = EmbeddingRetriever(adapter)
        retriever.fit(_train_samples())
        adapter.encode_batch.return_value = [[0.15, 0.25, 0.35, 0.45]]

        retriever.search(_query(), k=10)

        assert mock_index.knn_query.call_args[1]["k"] == 4

    @patch("hnswlib.Index")
    def test_should_return_empty_results_when_k_is_not_positive(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        retriever = EmbeddingRetriever(_adapter())
        retriever.fit(_train_samples())

        assert retriever.search(_query(), k=0) == []

    def test_should_raise_error_when_searching_before_fit(self) -> None:
        retriever = EmbeddingRetriever(_adapter())

        with pytest.raises(RuntimeError, match="must be fitted before search"):
            retriever.search(_query(), k=3)

    @patch("hnswlib.Index")
    def test_should_filter_negative_indices_from_results(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, -1, 2]]), np.array([[0.1, 0.2, 0.3]]))
        adapter = _adapter()
        retriever = EmbeddingRetriever(adapter)
        retriever.fit(_train_samples())
        adapter.encode_batch.return_value = [[0.1, 0.2, 0.3, 0.4]]

        results = retriever.search(_query(), k=3)

        assert [sample.text for sample, _ in results] == ["alpha", "gamma"]
