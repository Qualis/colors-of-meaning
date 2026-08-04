from collections import Counter
from typing import Any, List, Tuple

import numpy as np
import numpy.typing as npt

from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator

VALIDATION_BLOCK_SIZE = 1024


class ValidationAccuracyCheckpointSelector:
    def __init__(
        self,
        encode_use_case: EncodeDocumentUseCase,
        distance_calculator: DistanceCalculator,
        train_embeddings: npt.NDArray,
        train_labels: npt.NDArray,
        validation_embeddings: npt.NDArray,
        validation_labels: npt.NDArray,
        k: int = 5,
    ) -> None:
        self.encode_use_case = encode_use_case
        self.distance_calculator = distance_calculator
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
        self.validation_embeddings = validation_embeddings
        self.validation_labels = validation_labels
        self.k = k

    def __call__(self, color_mapper: ColorMapper) -> Tuple[Any, float]:
        scored = [
            (checkpoint, self._validation_accuracy(color_mapper, checkpoint))
            for checkpoint in color_mapper.epoch_checkpoints()
        ]
        return max(scored, key=lambda candidate: candidate[1])

    def _validation_accuracy(self, color_mapper: ColorMapper, checkpoint: Any) -> float:
        color_mapper.restore_checkpoint(checkpoint)
        train_documents = self.encode_use_case.execute_per_embedding(self.train_embeddings, "train")
        validation_documents = self.encode_use_case.execute_per_embedding(self.validation_embeddings, "validation")
        predictions = self._predict_all(validation_documents, train_documents)
        return self._accuracy(self.validation_labels, predictions)

    def _predict_all(
        self, validation_documents: List[ColoredDocument], train_documents: List[ColoredDocument]
    ) -> List[int]:
        predictions: List[int] = []
        for start in range(0, len(validation_documents), VALIDATION_BLOCK_SIZE):
            block = validation_documents[start : start + VALIDATION_BLOCK_SIZE]
            distances = self.distance_calculator.compute_distance_matrix(block, train_documents)
            predictions.extend(self._predict(row) for row in distances)
        return predictions

    def _predict(self, distances_to_train: npt.NDArray) -> int:
        ranked = np.argsort(distances_to_train, kind="stable")
        neighbours = [int(self.train_labels[index]) for index in ranked[: self.k]]
        return int(Counter(neighbours).most_common(1)[0][0])

    @staticmethod
    def _accuracy(true_labels: npt.NDArray, predictions: List[int]) -> float:
        correct = sum(
            1 for true_label, prediction in zip(true_labels, predictions, strict=True) if int(true_label) == prediction
        )
        return correct / len(predictions)
