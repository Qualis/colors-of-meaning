import pickle
from unittest.mock import Mock, patch
import numpy as np
from typing import List

from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import SentenceEmbeddingAdapter


def _assert_sentence_count(result: List[str]) -> None:
    """Helper to assert sentence count and first sentence."""
    assert len(result) == 4
    assert "First sentence." in result


def _assert_remaining_sentences(result: List[str]) -> None:
    """Helper to assert remaining sentence content."""
    assert "Second sentence!" in result
    assert "Third sentence?" in result
    assert "Remainder" in result


class TestSentenceEmbeddingAdapter:
    def test_should_initialize_with_model_name(self) -> None:
        adapter = SentenceEmbeddingAdapter(model_name="test-model")

        assert adapter.model_name == "test-model"
        assert adapter._model is None

    def test_should_encode_text(self) -> None:
        adapter = SentenceEmbeddingAdapter()
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1.0, 2.0, 3.0])
        adapter._model = mock_model

        result = adapter.encode("test text")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once_with("test text", convert_to_numpy=True)

    def test_should_encode_batch(self) -> None:
        adapter = SentenceEmbeddingAdapter()
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        adapter._model = mock_model

        texts = ["text1", "text2"]
        result = adapter.encode_batch(texts, batch_size=16, show_progress=True)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once_with(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True)

    def test_should_encode_document_sentences(self) -> None:
        adapter = SentenceEmbeddingAdapter()
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        adapter._model = mock_model

        document = "This is a sentence. This is another sentence!"
        result = adapter.encode_document_sentences(document, batch_size=16)

        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()

    def test_should_split_into_sentences(self) -> None:
        document = "First sentence. Second sentence! Third sentence? Remainder"

        result = SentenceEmbeddingAdapter._split_into_sentences(document)

        _assert_sentence_count(result)
        _assert_remaining_sentences(result)

    def test_should_handle_document_without_punctuation(self) -> None:
        document = "This is text without punctuation"

        result = SentenceEmbeddingAdapter._split_into_sentences(document)

        assert len(result) == 1
        assert result[0] == "This is text without punctuation"

    def test_should_filter_empty_sentences(self) -> None:
        document = "Sentence one.  Sentence two."

        result = SentenceEmbeddingAdapter._split_into_sentences(document)

        assert all(s for s in result)

    def test_should_get_embedding_dimension(self) -> None:
        adapter = SentenceEmbeddingAdapter()
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        adapter._model = mock_model

        result = adapter.embedding_dimension

        assert result == 384

    @patch("sentence_transformers.SentenceTransformer")
    def test_should_load_model_on_first_use(self, mock_sentence_transformer: Mock) -> None:
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        mock_model.encode.return_value = np.array([1.0, 2.0, 3.0])

        adapter = SentenceEmbeddingAdapter(model_name="test-model")
        adapter.encode("test")

        mock_sentence_transformer.assert_called_once_with("test-model")
        assert adapter._model is not None


class TestPickling:
    def test_should_drop_the_lazily_loaded_model_when_pickled(self) -> None:
        adapter = SentenceEmbeddingAdapter(model_name="test-model")
        adapter._model = Mock()

        assert adapter.__getstate__()["_model"] is None

    def test_should_keep_the_model_name_when_pickled(self) -> None:
        adapter = SentenceEmbeddingAdapter(model_name="test-model")
        adapter._model = Mock()

        assert pickle.loads(pickle.dumps(adapter)).model_name == "test-model"  # nosec B301  # nosemgrep
