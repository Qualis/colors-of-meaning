import numpy as np
from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.evaluation.color_histogram_retrieval_core import (
    ColorHistogramRetrievalCore,
)
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.colored_document import ColoredDocument


def _core(distance_side_effect: list | None = None) -> ColorHistogramRetrievalCore:
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
    return ColorHistogramRetrievalCore(embedding_adapter, encode_use_case, distance_calculator, num_candidates=100)


def _train_samples() -> list:
    return [EvaluationSample(text=f"train {index}", label=index % 2, split="train") for index in range(4)]


def _query() -> EvaluationSample:
    return EvaluationSample(text="query", label=0, split="test")


class TestColorHistogramRetrievalCore:
    @patch("hnswlib.Index")
    def test_should_rank_candidates_in_non_decreasing_distance_order(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        core = _core(distance_side_effect=[0.4, 0.1, 0.3, 0.2])
        core.fit(_train_samples())

        ranked = core.rank(_query(), k=4, document_id="query")

        assert [distance for _, distance in ranked] == [0.1, 0.2, 0.3, 0.4]

    @patch("hnswlib.Index")
    def test_should_return_training_indices_ordered_by_distance(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        core = _core(distance_side_effect=[0.4, 0.1, 0.3, 0.2])
        core.fit(_train_samples())

        ranked = core.rank(_query(), k=4, document_id="query")

        assert [index for index, _ in ranked] == [1, 3, 2, 0]

    @patch("hnswlib.Index")
    def test_should_truncate_ranking_to_k(self, mock_index_class: Mock) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        core = _core(distance_side_effect=[0.4, 0.1, 0.3, 0.2])
        core.fit(_train_samples())

        ranked = core.rank(_query(), k=2, document_id="query")

        assert len(ranked) == 2

    def test_should_return_empty_ranking_when_k_is_not_positive(self) -> None:
        core = _core()

        assert core.rank(_query(), k=0, document_id="query") == []
