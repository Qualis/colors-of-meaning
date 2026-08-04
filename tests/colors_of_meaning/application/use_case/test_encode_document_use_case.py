from unittest.mock import Mock
import numpy as np
import pytest
from typing import List

from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.domain.model.colored_document import ColoredDocument


def _create_batch_embeddings() -> List[np.ndarray]:
    """Helper to create test batch embeddings."""
    return [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]),
    ]


def _assert_batch_results(results: List[ColoredDocument]) -> None:
    """Helper to assert batch encoding results."""
    assert len(results) == 2
    assert results[0].document_id == "doc1"
    assert results[1].document_id == "doc2"


def _assert_batch_sequences(results: List[ColoredDocument]) -> None:
    """Helper to assert color sequences in batch results."""
    assert results[0].color_sequence == [0, 1]
    assert results[1].color_sequence == [2, 2, 1]


class TestEncodeDocumentUseCase:
    def test_should_encode_document_from_embeddings(self) -> None:
        mock_quantized_mapper = Mock()
        mock_quantized_mapper.embed_batch_to_bins.return_value = [0, 1, 1, 2]
        mock_quantized_mapper.codebook.num_bins = 3

        use_case = EncodeDocumentUseCase(mock_quantized_mapper)
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        result = use_case.execute(embeddings, document_id="doc1")

        assert isinstance(result, ColoredDocument)
        assert result.document_id == "doc1"
        assert result.color_sequence == [0, 1, 1, 2]
        assert result.num_bins == 3
        mock_quantized_mapper.embed_batch_to_bins.assert_called_once_with(embeddings)

    def test_should_encode_batch_of_documents(self) -> None:
        mock_quantized_mapper = Mock()
        mock_quantized_mapper.embed_batch_to_bins.side_effect = [[0, 1], [2, 2, 1]]
        mock_quantized_mapper.codebook.num_bins = 3

        use_case = EncodeDocumentUseCase(mock_quantized_mapper)
        results = use_case.execute_batch(_create_batch_embeddings(), ["doc1", "doc2"])

        _assert_batch_results(results)
        _assert_batch_sequences(results)

    def test_should_encode_one_document_per_embedding_row_in_a_single_pass(self) -> None:
        mock_quantized_mapper = Mock()
        mock_quantized_mapper.embed_batch_to_bins.return_value = [2, 0, 1]
        mock_quantized_mapper.codebook.num_bins = 3
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        results = EncodeDocumentUseCase(mock_quantized_mapper).execute_per_embedding(embeddings, "train")

        assert [document.color_sequence for document in results] == [[2], [0], [1]]

    def test_should_name_each_per_embedding_document_by_its_row_index(self) -> None:
        mock_quantized_mapper = Mock()
        mock_quantized_mapper.embed_batch_to_bins.return_value = [2, 0]
        mock_quantized_mapper.codebook.num_bins = 3

        results = EncodeDocumentUseCase(mock_quantized_mapper).execute_per_embedding(
            np.array([[1.0, 2.0], [3.0, 4.0]]), "validation"
        )

        assert [document.document_id for document in results] == ["validation_0", "validation_1"]

    def test_should_embed_all_rows_with_one_batched_call_when_encoding_per_embedding(self) -> None:
        mock_quantized_mapper = Mock()
        mock_quantized_mapper.embed_batch_to_bins.return_value = [2, 0]
        mock_quantized_mapper.codebook.num_bins = 3
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        EncodeDocumentUseCase(mock_quantized_mapper).execute_per_embedding(embeddings, "train")

        mock_quantized_mapper.embed_batch_to_bins.assert_called_once_with(embeddings)

    def test_should_raise_error_when_batch_sizes_mismatch(self) -> None:
        mock_quantized_mapper = Mock()
        mock_quantized_mapper.codebook.num_bins = 3

        use_case = EncodeDocumentUseCase(mock_quantized_mapper)
        embeddings_list = [np.array([[1.0, 2.0]])]
        document_ids = ["doc1", "doc2"]

        with pytest.raises(ValueError, match="Mismatch between embeddings and document IDs"):
            use_case.execute_batch(embeddings_list, document_ids)
