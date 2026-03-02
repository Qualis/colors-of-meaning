from typing import Dict, List

import numpy.typing as npt

from colors_of_meaning.domain.service.compression_baseline import CompressionBaseline


class CompressionComparisonUseCase:
    def __init__(self, baselines: List[CompressionBaseline]) -> None:
        self.baselines = baselines

    def execute(self, embeddings: npt.NDArray) -> List[Dict[str, object]]:
        results = []

        for baseline in self.baselines:
            compressed = baseline.compress(embeddings)
            num_sentences = embeddings.shape[0]
            bits_per_token = compressed.compressed_size_bits / num_sentences if num_sentences > 0 else 0.0

            results.append(
                {
                    "method": baseline.name(),
                    "compressed_size_bits": compressed.compressed_size_bits,
                    "original_size_bits": compressed.original_size_bits,
                    "compression_ratio": compressed.compression_ratio,
                    "bits_per_token": bits_per_token,
                    "reconstruction_error": compressed.reconstruction_error,
                }
            )

        return results
