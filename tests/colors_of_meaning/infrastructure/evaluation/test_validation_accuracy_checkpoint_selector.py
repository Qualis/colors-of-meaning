from collections import Counter
from pathlib import Path
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
)
from colors_of_meaning.infrastructure.evaluation.validation_accuracy_checkpoint_selector import (
    ValidationAccuracyCheckpointSelector,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)

SELECTOR_MODULE = "colors_of_meaning.infrastructure.evaluation.validation_accuracy_checkpoint_selector"
DOCUMENTS_DIR = Path("documents")
TIED_TRAIN_DOCUMENTS = 64
TIED_LEADING_LABELS = 5
DIFFERENTIAL_CAP = 60
DIFFERENTIAL_TRAIN = 250
DIFFERENTIAL_VALIDATION = 80
DIFFERENTIAL_FEATURES = 384
DIFFERENTIAL_NEIGHBOURS = 5
DIFFERENTIAL_EPOCHS = 80
DIFFERENTIAL_MINIMUM_OCCUPIED_BINS = 50


def _mode_switching_mapper() -> Mock:
    state = {"mode": "collapsing"}
    mapper = Mock()
    mapper.epoch_checkpoints.return_value = ["collapsing", "separating"]
    mapper.restore_checkpoint.side_effect = lambda checkpoint: state.update(mode=checkpoint)

    def embed_batch_to_lab(embeddings: np.ndarray) -> list:
        if state["mode"] == "separating":
            return [LabColor.from_unclamped(50.0, 120.0 if row[0] > 0.5 else -120.0, 0.0) for row in embeddings]
        return [LabColor.from_unclamped(50.0, 0.0, 0.0) for row in embeddings]

    mapper.embed_batch_to_lab.side_effect = embed_batch_to_lab
    return mapper


def _selector(mapper: Mock) -> ValidationAccuracyCheckpointSelector:
    encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(mapper, ColorCodebook.create_uniform_grid(2)))
    return ValidationAccuracyCheckpointSelector(
        encode_use_case=encode_use_case,
        distance_calculator=JensenShannonDistanceCalculator(smoothing_epsilon=1e-8),
        train_embeddings=np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32),
        train_labels=np.array([0, 0, 1, 1]),
        validation_embeddings=np.array([[0.0], [1.0]], dtype=np.float32),
        validation_labels=np.array([0, 1]),
        k=1,
    )


def _collapsing_mapper() -> Mock:
    mapper = Mock()
    mapper.epoch_checkpoints.return_value = ["only"]
    mapper.restore_checkpoint.side_effect = lambda checkpoint: None
    mapper.embed_batch_to_lab.side_effect = lambda embeddings: [
        LabColor.from_unclamped(50.0, 0.0, 0.0) for _row in embeddings
    ]
    return mapper


def _tie_breaking_selector(mapper: Mock) -> ValidationAccuracyCheckpointSelector:
    encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(mapper, ColorCodebook.create_uniform_grid(2)))
    labels = np.array([0] * TIED_LEADING_LABELS + [1] * (TIED_TRAIN_DOCUMENTS - TIED_LEADING_LABELS))
    return ValidationAccuracyCheckpointSelector(
        encode_use_case=encode_use_case,
        distance_calculator=JensenShannonDistanceCalculator(smoothing_epsilon=1e-8),
        train_embeddings=np.zeros((TIED_TRAIN_DOCUMENTS, 1), dtype=np.float32),
        train_labels=labels,
        validation_embeddings=np.zeros((1, 1), dtype=np.float32),
        validation_labels=np.array([0]),
        k=TIED_LEADING_LABELS,
    )


class BatchOnlyDistanceCalculator(DistanceCalculator):
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        raise NotImplementedError

    def compute_distance_matrix(self, queries: List[ColoredDocument], references: List[ColoredDocument]) -> np.ndarray:
        return self.matrix

    def metric_name(self) -> str:
        return "batch_only"


class TestValidationAccuracyCheckpointSelector:
    def test_should_select_the_checkpoint_with_higher_validation_accuracy(self) -> None:
        mapper = _mode_switching_mapper()

        selected_checkpoint, _accuracy = _selector(mapper)(mapper)

        assert selected_checkpoint == "separating"

    def test_should_report_perfect_validation_accuracy_for_the_separating_checkpoint(self) -> None:
        mapper = _mode_switching_mapper()

        _selected_checkpoint, accuracy = _selector(mapper)(mapper)

        assert accuracy == 1.0

    def test_should_rank_neighbours_from_the_batched_distance_matrix(self) -> None:
        mapper = _mode_switching_mapper()
        selector = _selector(mapper)
        selector.distance_calculator = BatchOnlyDistanceCalculator(np.array([[0.0, 9.0, 9.0, 9.0]] * 2))

        _selected_checkpoint, accuracy = selector(mapper)

        assert accuracy == 0.5

    def test_should_break_distance_ties_by_ascending_train_index_when_every_neighbour_is_equidistant(self) -> None:
        mapper = _collapsing_mapper()

        _selected_checkpoint, accuracy = _tie_breaking_selector(mapper)(mapper)

        assert accuracy == 1.0

    def test_should_score_identically_when_validation_documents_span_several_blocks(self, mocker) -> None:
        whole_mapper = _mode_switching_mapper()
        whole_accuracy = _selector(whole_mapper)(whole_mapper)[1]
        mocker.patch(f"{SELECTOR_MODULE}.VALIDATION_BLOCK_SIZE", 1)
        blocked_mapper = _mode_switching_mapper()

        blocked_accuracy = _selector(blocked_mapper)(blocked_mapper)[1]

        assert blocked_accuracy == whole_accuracy


def _real_split(split: str, limit: int) -> tuple:
    adapter = DocumentCorpusDatasetAdapter(
        documents_dir=str(DOCUMENTS_DIR),
        min_paragraph_chars=200,
        paragraphs_per_work=DIFFERENTIAL_CAP,
    )
    return adapter.get_samples(split=split, seed=0)[:limit], adapter.get_num_classes()


def _hashed_embeddings(texts: List[str]) -> np.ndarray:
    vectorizer = HashingVectorizer(n_features=DIFFERENTIAL_FEATURES, alternate_sign=False)
    return np.asarray(vectorizer.transform(texts).toarray(), dtype=np.float32)


def _trained_mapper_on_real_corpus(
    train_embeddings: np.ndarray, train_labels: np.ndarray, num_classes: int
) -> SupervisedPyTorchColorMapper:
    mapper = SupervisedPyTorchColorMapper(
        input_dim=DIFFERENTIAL_FEATURES, num_classes=num_classes, device="cpu", seed=0
    )
    mapper.set_training_labels(train_labels)
    mapper.train(train_embeddings, epochs=DIFFERENTIAL_EPOCHS, learning_rate=1e-3)
    return mapper


def _per_pair_matrix(validation_documents: List[ColoredDocument], train_documents: List[ColoredDocument]) -> np.ndarray:
    calculator = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8)
    return np.array(
        [
            [calculator.compute_distance(query, reference) for reference in train_documents]
            for query in validation_documents
        ]
    )


def _majority_label(neighbours: List[int]) -> int:
    return int(Counter(neighbours).most_common(1)[0][0])


def _predictions_from(matrix: np.ndarray, train_labels: np.ndarray, k: int) -> List[int]:
    return [
        _majority_label([int(train_labels[index]) for index in np.argsort(row, kind="stable")[:k]]) for row in matrix
    ]


def _rows_whose_neighbour_set_is_undetermined(matrix: np.ndarray, k: int) -> List[int]:
    ordered = np.sort(matrix, axis=1)
    return [row for row in range(len(matrix)) if ordered[row, k - 1] == ordered[row, k]]


def _differing_rows(left: List[int], right: List[int]) -> List[int]:
    return [row for row, (first, second) in enumerate(zip(left, right, strict=True)) if first != second]


@pytest.fixture(scope="module")
def real_corpus_differential() -> dict:
    train_samples, num_classes = _real_split("train", DIFFERENTIAL_TRAIN)
    validation_samples, _ = _real_split("validation", DIFFERENTIAL_VALIDATION)
    train_embeddings = _hashed_embeddings([sample.text for sample in train_samples])
    validation_embeddings = _hashed_embeddings([sample.text for sample in validation_samples])
    train_labels = np.array([sample.label for sample in train_samples])

    mapper = _trained_mapper_on_real_corpus(train_embeddings, train_labels, num_classes)
    encode_use_case = EncodeDocumentUseCase(
        QuantizedColorMapper(mapper, ColorCodebook.create_uniform_grid(bins_per_dimension=16))
    )
    mapper.restore_checkpoint(mapper.epoch_checkpoints()[-1])
    train_documents = encode_use_case.execute_per_embedding(train_embeddings, "train")
    validation_documents = encode_use_case.execute_per_embedding(validation_embeddings, "validation")

    legacy = _per_pair_matrix(validation_documents, train_documents)
    optimised = JensenShannonDistanceCalculator(smoothing_epsilon=1e-8).compute_distance_matrix(
        validation_documents, train_documents
    )
    legacy_predictions = _predictions_from(legacy, train_labels, DIFFERENTIAL_NEIGHBOURS)
    optimised_predictions = _predictions_from(optimised, train_labels, DIFFERENTIAL_NEIGHBOURS)
    return {
        "legacy": legacy,
        "optimised": optimised,
        "occupied_bins": len({int(np.argmax(document.histogram)) for document in train_documents}),
        "differing_rows": _differing_rows(legacy_predictions, optimised_predictions),
        "undetermined_rows": _rows_whose_neighbour_set_is_undetermined(optimised, DIFFERENTIAL_NEIGHBOURS),
    }


@pytest.mark.integration
@pytest.mark.skipif(not DOCUMENTS_DIR.exists(), reason="the authored document corpus is git-ignored and may be absent")
class TestRealCorpusDifferentialEquivalence:
    def test_should_reach_a_checkpoint_that_separates_documents_into_many_colour_bins(
        self, real_corpus_differential: dict
    ) -> None:
        assert real_corpus_differential["occupied_bins"] > DIFFERENTIAL_MINIMUM_OCCUPIED_BINS

    def test_should_match_the_per_pair_reference_distance_to_within_one_ulp_everywhere(
        self, real_corpus_differential: dict
    ) -> None:
        legacy = real_corpus_differential["legacy"]

        deviation = np.abs(real_corpus_differential["optimised"] - legacy)

        assert np.all(deviation <= np.spacing(np.abs(legacy)))

    def test_should_only_change_predictions_where_the_neighbour_set_is_not_determined_by_distance(
        self, real_corpus_differential: dict
    ) -> None:
        undetermined = set(real_corpus_differential["undetermined_rows"])

        assert set(real_corpus_differential["differing_rows"]) <= undetermined
