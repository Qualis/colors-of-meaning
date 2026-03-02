import numpy as np
from unittest.mock import Mock

from colors_of_meaning.application.use_case.compression_comparison_use_case import (
    CompressionComparisonUseCase,
)
from colors_of_meaning.domain.service.compression_baseline import CompressedResult


class TestCompressionComparisonUseCase:
    def test_should_compare_baselines(self) -> None:
        mock_baseline_1 = Mock()
        mock_baseline_1.name.return_value = "gzip"
        mock_baseline_1.compress.return_value = CompressedResult(
            compressed_size_bits=500,
            original_size_bits=1000,
            reconstruction_error=0.0,
        )

        mock_baseline_2 = Mock()
        mock_baseline_2.name.return_value = "pq"
        mock_baseline_2.compress.return_value = CompressedResult(
            compressed_size_bits=200,
            original_size_bits=1000,
            reconstruction_error=0.01,
        )

        use_case = CompressionComparisonUseCase(baselines=[mock_baseline_1, mock_baseline_2])
        embeddings = np.random.randn(10, 384).astype(np.float32)

        results = use_case.execute(embeddings)

        assert len(results) == 2

    def test_should_include_method_name(self) -> None:
        mock_baseline = Mock()
        mock_baseline.name.return_value = "gzip"
        mock_baseline.compress.return_value = CompressedResult(
            compressed_size_bits=500,
            original_size_bits=1000,
        )

        use_case = CompressionComparisonUseCase(baselines=[mock_baseline])
        embeddings = np.random.randn(10, 384).astype(np.float32)

        results = use_case.execute(embeddings)

        assert results[0]["method"] == "gzip"

    def test_should_include_compression_ratio(self) -> None:
        mock_baseline = Mock()
        mock_baseline.name.return_value = "gzip"
        mock_baseline.compress.return_value = CompressedResult(
            compressed_size_bits=500,
            original_size_bits=1000,
        )

        use_case = CompressionComparisonUseCase(baselines=[mock_baseline])
        embeddings = np.random.randn(10, 384).astype(np.float32)

        results = use_case.execute(embeddings)

        assert results[0]["compression_ratio"] == 2.0

    def test_should_compute_bits_per_token(self) -> None:
        mock_baseline = Mock()
        mock_baseline.name.return_value = "gzip"
        mock_baseline.compress.return_value = CompressedResult(
            compressed_size_bits=500,
            original_size_bits=1000,
        )

        use_case = CompressionComparisonUseCase(baselines=[mock_baseline])
        embeddings = np.random.randn(10, 384).astype(np.float32)

        results = use_case.execute(embeddings)

        assert results[0]["bits_per_token"] == 50.0

    def test_should_handle_empty_baselines(self) -> None:
        use_case = CompressionComparisonUseCase(baselines=[])
        embeddings = np.random.randn(10, 384).astype(np.float32)

        results = use_case.execute(embeddings)

        assert len(results) == 0
