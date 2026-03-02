import numpy as np

from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)
from colors_of_meaning.domain.service.compression_baseline import CompressedResult


class TestPQCompressionBaseline:
    def test_should_compress_embeddings(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_achieve_compression(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.compression_ratio > 1.0

    def test_should_have_positive_reconstruction_error(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=4)
        embeddings = np.random.randn(20, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.reconstruction_error is not None
        assert result.reconstruction_error >= 0.0

    def test_should_compute_correct_original_size(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=4, num_centroids=8)
        embeddings = np.random.randn(10, 16).astype(np.float32)

        result = baseline.compress(embeddings)

        assert result.original_size_bits == 10 * 16 * 32

    def test_should_return_correct_name(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=48, num_centroids=256)

        assert baseline.name() == "pq_m48_k256"

    def test_should_handle_small_embeddings(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=2, num_centroids=4)
        embeddings = np.random.randn(5, 4).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)

    def test_should_handle_more_subspaces_than_dims(self) -> None:
        baseline = PQCompressionBaseline(num_subspaces=20, num_centroids=4)
        embeddings = np.random.randn(10, 8).astype(np.float32)

        result = baseline.compress(embeddings)

        assert isinstance(result, CompressedResult)
