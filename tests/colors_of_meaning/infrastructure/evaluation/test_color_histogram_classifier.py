from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_retrieval_core import (
    ColorHistogramRetrievalCore,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import WassersteinDistanceCalculator
from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
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


def _class_separated_dataset() -> Tuple[List[EvaluationSample], List[EvaluationSample], Dict[str, npt.NDArray]]:
    rng = np.random.default_rng(123)
    dimension = 16
    class_zero_axis = np.zeros(dimension, dtype=np.float32)
    class_zero_axis[0] = 4.0
    class_one_axis = np.zeros(dimension, dtype=np.float32)
    class_one_axis[1] = 4.0

    embedding_by_text: Dict[str, npt.NDArray] = {}
    train_samples: List[EvaluationSample] = []
    test_samples: List[EvaluationSample] = []
    for label, axis in [(0, class_zero_axis), (1, class_one_axis)]:
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


class TestColorHistogramClassifier:
    @pytest.fixture
    def mock_embedding_adapter(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_encode_use_case(self) -> Mock:
        mock_use_case = Mock()
        histogram = np.array([0.25, 0.25, 0.25, 0.25])
        mock_doc = ColoredDocument(histogram=histogram, document_id="doc_1")
        mock_use_case.execute.return_value = mock_doc
        return mock_use_case

    @pytest.fixture
    def mock_distance_calculator(self) -> Mock:
        mock_calc = Mock()
        mock_calc.compute_distance.return_value = 0.5
        return mock_calc

    @pytest.fixture
    def classifier(
        self,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
        mock_distance_calculator: Mock,
    ) -> ColorHistogramClassifier:
        return ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
            k=3,
        )

    @pytest.fixture
    def train_samples(self) -> list:
        return [
            EvaluationSample(text="the cat sat on the mat", label=0, split="train"),
            EvaluationSample(text="the dog ran in the park", label=1, split="train"),
            EvaluationSample(text="cats and dogs are pets", label=0, split="train"),
            EvaluationSample(text="dogs like to run and play", label=1, split="train"),
        ]

    @patch("hnswlib.Index")
    def test_should_fit_classifier_with_samples(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        assert len(classifier.core.training_samples) == 4
        assert len(classifier.core.training_docs) == 4
        assert mock_embedding_adapter.encode_document_sentences.call_count == 4
        assert mock_encode_use_case.execute.call_count == 4
        mock_index.add_items.assert_called_once()

    @patch("hnswlib.Index")
    def test_should_build_hnsw_index_with_cosine_space(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        mock_index_class.assert_called_once_with(space="cosine", dim=4)
        mock_index.init_index.assert_called_once()
        mock_index.set_ef.assert_called_once_with(100)

    @patch("hnswlib.Index")
    def test_should_set_ef_at_least_num_candidates_when_fitting(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        assert mock_index.set_ef.call_args[0][0] >= classifier.core.num_candidates

    @patch("hnswlib.Index")
    def test_should_pin_single_thread_when_building_index(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        mock_index.set_num_threads.assert_called_once_with(1)

    @patch("hnswlib.Index")
    def test_should_predict_labels_using_two_phase_retrieval(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 2, 1, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]
        mock_distance_calculator.compute_distance.side_effect = [0.1, 0.3, 0.2, 0.4]

        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="the cat is sleeping", label=0, split="test"),
        ]
        predictions = classifier.predict(test_samples)

        assert len(predictions) == 1
        assert predictions[0] == 0

    @patch("hnswlib.Index")
    def test_should_rerank_candidates_with_distance_calculator(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="test", label=0, split="test"),
        ]
        classifier.predict(test_samples)

        assert mock_distance_calculator.compute_distance.call_count == 4

    def test_should_raise_error_when_predicting_before_fit(self, classifier: ColorHistogramClassifier) -> None:
        test_samples = [
            EvaluationSample(text="test text", label=0, split="test"),
        ]

        with pytest.raises(RuntimeError, match="must be fitted before prediction"):
            classifier.predict(test_samples)

    @patch("hnswlib.Index")
    def test_should_predict_multiple_samples(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 2, 1]]), np.array([[0.1, 0.2, 0.3]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="cats are great", label=0, split="test"),
            EvaluationSample(text="dogs are fun", label=1, split="test"),
        ]
        predictions = classifier.predict(test_samples)

        assert len(predictions) == 2

    @patch("hnswlib.Index")
    def test_should_return_list_of_integers(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[1, 3, 0]]), np.array([[0.1, 0.2, 0.3]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="test", label=0, split="test"),
        ]
        predictions = classifier.predict(test_samples)

        assert isinstance(predictions[0], int)

    @patch("hnswlib.Index")
    def test_should_use_k_nearest_from_reranked_candidates(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, 1, 2, 3]]), np.array([[0.1, 0.2, 0.3, 0.4]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]
        mock_distance_calculator.compute_distance.side_effect = [0.9, 0.1, 0.5, 0.2]

        classifier.fit(train_samples)

        test_samples = [
            EvaluationSample(text="cats", label=0, split="test"),
        ]
        predictions = classifier.predict(test_samples)

        assert predictions[0] == 1

    @patch("hnswlib.Index")
    def test_should_handle_edge_case_with_no_neighbors(
        self,
        mock_index_class: Mock,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
            k=0,
        )

        train_samples = [
            EvaluationSample(text="train", label=1, split="train"),
        ]
        classifier.fit(train_samples)

        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        predictions = classifier.predict(test_samples)

        assert predictions[0] == 0

    @patch("hnswlib.Index")
    def test_should_handle_fewer_training_samples_than_num_candidates(
        self,
        mock_index_class: Mock,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0]]), np.array([[0.1]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
            k=3,
            num_candidates=100,
        )

        train_samples = [
            EvaluationSample(text="train", label=1, split="train"),
        ]
        classifier.fit(train_samples)

        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        classifier.predict(test_samples)

        call_args = mock_index.knn_query.call_args
        assert call_args[1]["k"] == 1

    def test_should_handle_majority_vote_with_empty_list(self, mock_embedding_adapter: Mock) -> None:
        mock_encode_use_case = Mock()
        mock_distance_calculator = Mock()
        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
        )

        result = classifier._majority_vote([])

        assert result == 0

    def test_should_handle_majority_vote_with_single_label(self, mock_embedding_adapter: Mock) -> None:
        mock_encode_use_case = Mock()
        mock_distance_calculator = Mock()
        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
        )

        result = classifier._majority_vote([1])

        assert result == 1

    def test_should_handle_majority_vote_with_tie(self, mock_embedding_adapter: Mock) -> None:
        mock_encode_use_case = Mock()
        mock_distance_calculator = Mock()
        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
        )

        result = classifier._majority_vote([0, 1])

        assert result in [0, 1]

    @patch("hnswlib.Index")
    def test_should_filter_negative_indices_in_nearest_labels(
        self,
        mock_index_class: Mock,
        classifier: ColorHistogramClassifier,
        train_samples: list,
        mock_embedding_adapter: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_index = Mock()
        mock_index_class.return_value = mock_index
        mock_index.knn_query.return_value = (np.array([[0, -1, 2]]), np.array([[0.1, 0.2, 0.3]]))
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier.fit(train_samples)

        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        predictions = classifier.predict(test_samples)

        assert mock_distance_calculator.compute_distance.call_count == 2
        assert predictions[0] == 0

    def test_should_return_default_label_when_no_training_data(
        self,
        mock_embedding_adapter: Mock,
        mock_encode_use_case: Mock,
        mock_distance_calculator: Mock,
    ) -> None:
        mock_embedding_adapter.encode_document_sentences.return_value = [[0.1, 0.2, 0.3]]

        classifier = ColorHistogramClassifier(
            embedding_adapter=mock_embedding_adapter,
            encode_use_case=mock_encode_use_case,
            distance_calculator=mock_distance_calculator,
            k=3,
        )
        classifier.core.index = Mock()
        classifier.core.training_samples = []

        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        predictions = classifier.predict(test_samples)

        assert predictions[0] == 0

    def test_should_classify_above_chance_when_pipeline_maps_separable_classes(self) -> None:
        train_samples, test_samples, embedding_by_text = _class_separated_dataset()
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)
        quantized_mapper = QuantizedColorMapper(color_mapper=_StructurePreservingColorMapper(), codebook=codebook)
        encode_use_case = EncodeDocumentUseCase(quantized_mapper=quantized_mapper)
        distance_calculator = WassersteinDistanceCalculator(codebook=codebook)
        adapter = Mock()
        adapter.encode_document_sentences.side_effect = lambda text: embedding_by_text[text]
        classifier = ColorHistogramClassifier(
            embedding_adapter=adapter,
            encode_use_case=encode_use_case,
            distance_calculator=distance_calculator,
            k=3,
        )

        classifier.fit(train_samples)
        predictions = classifier.predict(test_samples)

        accuracy = sum(
            int(prediction == sample.label) for prediction, sample in zip(predictions, test_samples, strict=True)
        ) / len(test_samples)
        assert accuracy >= 0.8, f"color pipeline accuracy {accuracy} below floor 0.8 (chance 0.5)"

    def test_should_delegate_to_a_shared_retrieval_core(
        self,
        classifier: ColorHistogramClassifier,
    ) -> None:
        assert isinstance(classifier.core, ColorHistogramRetrievalCore)
