import gzip

import numpy as np
import numpy.typing as npt

from colors_of_meaning.domain.service.compression_baseline import (
    CompressionBaseline,
    CompressedResult,
)


class GzipCompressionBaseline(CompressionBaseline):
    def compress(self, embeddings: npt.NDArray) -> CompressedResult:
        raw_bytes = embeddings.astype(np.float32).tobytes()
        original_size_bits = len(raw_bytes) * 8

        compressed_bytes = gzip.compress(raw_bytes)
        compressed_size_bits = len(compressed_bytes) * 8

        decompressed_bytes = gzip.decompress(compressed_bytes)
        reconstructed = np.frombuffer(decompressed_bytes, dtype=np.float32).reshape(embeddings.shape)
        reconstruction_error = float(np.mean((embeddings - reconstructed) ** 2))

        return CompressedResult(
            compressed_size_bits=compressed_size_bits,
            original_size_bits=original_size_bits,
            reconstruction_error=reconstruction_error,
        )

    def name(self) -> str:
        return "gzip"
