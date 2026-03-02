import numpy as np

from colors_of_meaning.infrastructure.ml.gzip_compression_baseline import (
    GzipCompressionBaseline,
)
from colors_of_meaning.domain.service.compression_baseline import CompressedResult


class TestGzipCompressionBaseline:
    def test_should_compress_embeddings(self) -> None:
        baseline = GzipCompressionBaseline()
        embeddings = np.random.randn(10, 384).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_achieve_compression(self) -> None:
        baseline = GzipCompressionBaseline()
        embeddings = np.random.randn(10, 384).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.compressed_size_bits < result.original_size_bits

    def test_should_have_zero_reconstruction_error(self) -> None:
        baseline = GzipCompressionBaseline()
        embeddings = np.random.randn(10, 384).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.reconstruction_error == 0.0

    def test_should_compute_correct_original_size(self) -> None:
        baseline = GzipCompressionBaseline()
        embeddings = np.random.randn(5, 384).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.original_size_bits == 5 * 384 * 32

    def test_should_return_correct_name(self) -> None:
        baseline = GzipCompressionBaseline()

        assert baseline.name() == "gzip"
