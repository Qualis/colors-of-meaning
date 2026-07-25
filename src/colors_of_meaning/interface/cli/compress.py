import tyro
from dataclasses import dataclass

import numpy as np

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.application.use_case.compression_comparison_use_case import (
    CompressionComparisonUseCase,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.gzip_compression_baseline import (
    GzipCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.color_vq_compression_baseline import (
    ColorVqCompressionBaseline,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook

DELTA_E_METHOD = "color_vq"


@dataclass
class CompressArgs:
    config: str = "configs/base.yaml"
    embeddings_path: str = "artifacts/encoded/test_embeddings.npy"
    model_path: str = "artifacts/models/projector.pth"
    codebook_name: str = "codebook_4096"
    compare_baselines: bool = False


def _setup_color_mapper(config: SynestheticConfig, model_path: str) -> PyTorchColorMapper:
    color_mapper = PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        device=config.training.device,
    )
    color_mapper.load_weights(model_path)
    return color_mapper


def _build_color_vq_baseline(
    config: SynestheticConfig, args: CompressArgs, codebook: ColorCodebook
) -> ColorVqCompressionBaseline:
    color_mapper = _setup_color_mapper(config, args.model_path)
    return ColorVqCompressionBaseline(codebook=codebook, color_mapper=color_mapper)


def _distortion_unit(method: str) -> str:
    return "ΔE" if method == DELTA_E_METHOD else "MSE"


def _original_basis(method: str) -> str:
    return "Lab-triple" if method == DELTA_E_METHOD else "embedding"


def _run_vq_analysis(args: CompressArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    codebook = load_codebook(args.codebook_name)
    baseline = _build_color_vq_baseline(config, args, codebook)

    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = np.load(args.embeddings_path)
    result = baseline.compress(embeddings)

    print("\n" + "=" * 60)
    print("COLOR-VQ COMPRESSION")
    print("=" * 60)
    print(f"Original size (bits): {result.original_size_bits:,}")
    print(f"Compressed size (bits): {result.compressed_size_bits:,}")
    print(f"Compression ratio: {result.compression_ratio:.2f}x")
    print(f"Reconstruction error (ΔE): {result.reconstruction_error:.4f}")
    print(
        f"Shared palette overhead (one-time, excluded from rate): {baseline.codec.shared_palette_overhead_bits():,} bits"
    )
    print("=" * 60)


def _run_baseline_comparison(args: CompressArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    codebook = load_codebook(args.codebook_name)

    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = np.load(args.embeddings_path)

    baselines = [
        GzipCompressionBaseline(),
        PQCompressionBaseline(num_subspaces=48, num_centroids=256, seed=config.training.seed),
        _build_color_vq_baseline(config, args, codebook),
    ]

    results = CompressionComparisonUseCase(baselines=baselines).execute(embeddings)

    print("\n" + "=" * 84)
    print("COMPRESSION BASELINE COMPARISON")
    print("=" * 84)
    print(f"{'Method':<22} {'Compresses':>12} {'Ratio':>9} {'Bits/Token':>11} {'Distortion':>13} {'Unit':>5}")
    print("-" * 84)

    for result in results:
        method = str(result["method"])
        distortion = result["reconstruction_error"]
        distortion_str = f"{distortion:.6f}" if distortion is not None else "N/A"
        print(
            f"{method:<22} "
            f"{_original_basis(method):>12} "
            f"{result['compression_ratio']:>9.2f}x "
            f"{result['bits_per_token']:>11.2f} "
            f"{distortion_str:>13} "
            f"{_distortion_unit(method):>5}"
        )

    print("=" * 84)
    print("Originals differ: gzip/PQ compress the high-dim embedding; color_vq compresses the 3-float Lab color")
    print(
        "(after the projector's lossy map). PQ and color_vq exclude their shared trained codebook; gzip is self-contained."
    )


def main(args: CompressArgs) -> None:
    if args.compare_baselines:
        _run_baseline_comparison(args)
    else:
        _run_vq_analysis(args)


if __name__ == "__main__":
    main(tyro.cli(CompressArgs))
