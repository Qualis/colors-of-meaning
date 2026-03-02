import pytest

from colors_of_meaning.domain.service.compression_baseline import (
    CompressionBaseline,
    CompressedResult,
)


class TestCompressedResult:
    def test_should_create_compressed_result(self) -> None:
        result = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
        )

        assert result.compressed_size_bits == 100
        assert result.original_size_bits == 1000

    def test_should_compute_compression_ratio(self) -> None:
        result = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
        )

        assert result.compression_ratio == 10.0

    def test_should_return_zero_ratio_for_zero_compressed_size(self) -> None:
        result = CompressedResult(
            compressed_size_bits=0,
            original_size_bits=1000,
        )

        assert result.compression_ratio == 0.0

    def test_should_store_reconstruction_error(self) -> None:
        result = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
            reconstruction_error=0.001,
        )

        assert result.reconstruction_error == 0.001

    def test_should_default_reconstruction_error_to_none(self) -> None:
        result = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
        )

        assert result.reconstruction_error is None

    def test_should_return_zero_bits_per_dimension(self) -> None:
        result = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
        )

        assert result.bits_per_dimension == 0.0


class TestCompressionBaselineAbstract:
    def test_should_be_abstract(self) -> None:
        with pytest.raises(TypeError):
            CompressionBaseline()  # type: ignore
