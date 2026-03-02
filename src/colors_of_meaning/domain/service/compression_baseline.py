from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy.typing as npt


@dataclass(frozen=True)
class CompressedResult:
    compressed_size_bits: int
    original_size_bits: int
    reconstruction_error: Optional[float] = None

    @property
    def compression_ratio(self) -> float:
        if self.compressed_size_bits == 0:
            return 0.0
        return self.original_size_bits / self.compressed_size_bits

    @property
    def bits_per_dimension(self) -> float:
        return 0.0


class CompressionBaseline(ABC):
    @abstractmethod
    def compress(self, embeddings: npt.NDArray) -> CompressedResult:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
