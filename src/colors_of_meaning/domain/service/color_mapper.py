from abc import ABC, abstractmethod
from typing import Any, List
import numpy.typing as npt

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook


class ColorMapper(ABC):
    @abstractmethod
    def embed_to_lab(self, embedding: npt.NDArray) -> LabColor:
        raise NotImplementedError

    @abstractmethod
    def embed_batch_to_lab(self, embeddings: npt.NDArray) -> List[LabColor]:
        raise NotImplementedError

    @abstractmethod
    def train(self, embeddings: npt.NDArray, epochs: int, learning_rate: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def epoch_checkpoints(self) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def restore_checkpoint(self, checkpoint: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_weights(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_weights(self, path: str) -> None:
        raise NotImplementedError


class QuantizedColorMapper:
    def __init__(self, color_mapper: ColorMapper, codebook: ColorCodebook) -> None:
        self.color_mapper = color_mapper
        self.codebook = codebook

    def embed_to_bin(self, embedding: npt.NDArray) -> int:
        lab_color = self.color_mapper.embed_to_lab(embedding)
        return self.codebook.quantize(lab_color)

    def embed_batch_to_bins(self, embeddings: npt.NDArray) -> List[int]:
        lab_colors = self.color_mapper.embed_batch_to_lab(embeddings)
        return [int(bin_index) for bin_index in self.codebook.quantize_batch(lab_colors)]
