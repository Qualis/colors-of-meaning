from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.interface.cli.compress import main, CompressArgs
from colors_of_meaning.domain.service.compression_baseline import CompressedResult


def _patched_config() -> Mock:
    config = Mock()
    config.training.seed = 42
    return config


class TestCompressCLI:
    @patch("colors_of_meaning.interface.cli.compress.np")
    @patch("colors_of_meaning.interface.cli.compress.ColorVqCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.compress.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.compress.SynestheticConfig")
    @patch("builtins.print")
    def test_should_construct_color_vq_baseline_with_loaded_codebook(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_repo_class: Mock,
        mock_mapper_class: Mock,
        mock_color_vq_class: Mock,
        mock_np: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config_class.from_yaml.return_value = _patched_config()
        codebook = Mock()
        mock_repo_class.return_value.load.return_value = codebook
        mock_np.load.return_value = np.random.randn(5, 8).astype(np.float32)
        mock_color_vq_class.return_value.compress.return_value = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
            reconstruction_error=2.5,
        )
        mock_color_vq_class.return_value.codec.shared_palette_overhead_bits.return_value = 393216

        main(CompressArgs(embeddings_path=str(tmp_path / "embeddings.npy")))

        mock_color_vq_class.assert_called_once_with(codebook=codebook, color_mapper=mock_mapper_class.return_value)

    @patch("colors_of_meaning.interface.cli.compress.np")
    @patch("colors_of_meaning.interface.cli.compress.ColorVqCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.compress.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.compress.SynestheticConfig")
    @patch("builtins.print")
    def test_should_compress_embeddings_in_vq_analysis_mode(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_repo_class: Mock,
        mock_mapper_class: Mock,
        mock_color_vq_class: Mock,
        mock_np: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config_class.from_yaml.return_value = _patched_config()
        mock_repo_class.return_value.load.return_value = Mock()
        embeddings = np.random.randn(5, 8).astype(np.float32)
        mock_np.load.return_value = embeddings
        mock_color_vq_class.return_value.compress.return_value = CompressedResult(
            compressed_size_bits=100,
            original_size_bits=1000,
            reconstruction_error=2.5,
        )
        mock_color_vq_class.return_value.codec.shared_palette_overhead_bits.return_value = 393216

        main(CompressArgs(embeddings_path=str(tmp_path / "embeddings.npy")))

        mock_color_vq_class.return_value.compress.assert_called_once_with(embeddings)

    @patch("colors_of_meaning.interface.cli.compress.np")
    @patch("colors_of_meaning.interface.cli.compress.CompressionComparisonUseCase")
    @patch("colors_of_meaning.interface.cli.compress.ColorVqCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.PQCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.GzipCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.compress.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.compress.SynestheticConfig")
    @patch("builtins.print")
    def test_should_include_color_vq_in_baseline_comparison_list(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_repo_class: Mock,
        mock_mapper_class: Mock,
        mock_gzip_class: Mock,
        mock_pq_class: Mock,
        mock_color_vq_class: Mock,
        mock_comparison_class: Mock,
        mock_np: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config_class.from_yaml.return_value = _patched_config()
        mock_repo_class.return_value.load.return_value = Mock()
        mock_np.load.return_value = np.random.randn(10, 8).astype(np.float32)
        mock_comparison_class.return_value.execute.return_value = [
            {
                "method": "color_vq",
                "compressed_size_bits": 100,
                "original_size_bits": 1000,
                "compression_ratio": 10.0,
                "bits_per_token": 10.0,
                "reconstruction_error": 4.2,
            },
        ]

        main(CompressArgs(compare_baselines=True, embeddings_path=str(tmp_path / "embeddings.npy")))

        baselines = mock_comparison_class.call_args.kwargs["baselines"]
        assert baselines[2] is mock_color_vq_class.return_value

    @patch("colors_of_meaning.interface.cli.compress.np")
    @patch("colors_of_meaning.interface.cli.compress.CompressionComparisonUseCase")
    @patch("colors_of_meaning.interface.cli.compress.ColorVqCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.PQCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.GzipCompressionBaseline")
    @patch("colors_of_meaning.interface.cli.compress.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.compress.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.compress.SynestheticConfig")
    @patch("builtins.print")
    def test_should_handle_none_reconstruction_error_in_comparison(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_repo_class: Mock,
        mock_mapper_class: Mock,
        mock_gzip_class: Mock,
        mock_pq_class: Mock,
        mock_color_vq_class: Mock,
        mock_comparison_class: Mock,
        mock_np: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config_class.from_yaml.return_value = _patched_config()
        mock_repo_class.return_value.load.return_value = Mock()
        mock_np.load.return_value = np.random.randn(10, 8).astype(np.float32)
        mock_comparison_class.return_value.execute.return_value = [
            {
                "method": "gzip",
                "compressed_size_bits": 500,
                "original_size_bits": 1000,
                "compression_ratio": 2.0,
                "bits_per_token": 50.0,
                "reconstruction_error": None,
            },
        ]

        main(CompressArgs(compare_baselines=True, embeddings_path=str(tmp_path / "embeddings.npy")))

        assert mock_print.call_count > 0

    @patch("colors_of_meaning.interface.cli.compress.np")
    @patch("colors_of_meaning.interface.cli.compress.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.compress.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.compress.SynestheticConfig")
    @patch("builtins.print")
    def test_should_raise_when_codebook_is_missing(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_repo_class: Mock,
        mock_mapper_class: Mock,
        mock_np: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config_class.from_yaml.return_value = _patched_config()
        mock_repo_class.return_value.load.return_value = None
        mock_np.load.return_value = np.random.randn(5, 8).astype(np.float32)

        with pytest.raises(FileNotFoundError, match="not found"):
            main(CompressArgs(embeddings_path=str(tmp_path / "embeddings.npy")))
