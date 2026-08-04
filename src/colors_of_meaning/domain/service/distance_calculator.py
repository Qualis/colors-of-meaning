from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt

from colors_of_meaning.domain.model.colored_document import ColoredDocument


class DistanceCalculator(ABC):
    @abstractmethod
    def compute_distance(self, doc1: ColoredDocument, doc2: ColoredDocument) -> float:
        raise NotImplementedError

    @abstractmethod
    def metric_name(self) -> str:
        raise NotImplementedError

    def compute_distance_matrix(
        self, queries: List[ColoredDocument], references: List[ColoredDocument]
    ) -> npt.NDArray[np.float64]:
        distances = np.empty((len(queries), len(references)), dtype=np.float64)
        for row, query in enumerate(queries):
            for column, reference in enumerate(references):
                distances[row, column] = self.compute_distance(query, reference)
        return distances
