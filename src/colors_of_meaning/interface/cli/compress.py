import tyro
import pickle  # nosec B403
from dataclasses import dataclass
from typing import List

import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.application.use_case.compress_document_use_case import (
    CompressDocumentUseCase,
)
from colors_of_meaning.application.use_case.compression_comparison_use_case import (
    CompressionComparisonUseCase,
)
from colors_of_meaning.infrastructure.ml.gzip_compression_baseline import (
    GzipCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)


@dataclass
class CompressArgs:
    config: str = "configs/base.yaml"
    encoded_documents: str = "artifacts/encoded/test_documents.pkl"
    method: str = "vq"
    compare_baselines: bool = False
    embeddings_path: str = "artifacts/encoded/test_embeddings.npy"


def _run_vq_analysis(args: CompressArgs) -> None:
    print(f"Loading encoded documents from {args.encoded_documents}...")
    with open(args.encoded_documents, "rb") as f:
        documents: List[ColoredDocument] = pickle.load(f)  # nosec B301 nosemgrep

    use_case = CompressDocumentUseCase()

    print(f"\nAnalyzing compression for {len(documents)} documents...")
    batch_results = use_case.execute_batch(documents)

    print("\n" + "=" * 60)
    print("COMPRESSION ANALYSIS")
    print("=" * 60)
    print(f"Total bits: {batch_results['total_bits']:,}")
    print(f"Total tokens: {batch_results['total_tokens']:,}")
    print(f"Average bits per token: {batch_results['average_bits_per_token']:.2f}")
    print("=" * 60)

    print("\nSample individual results:")
    for i, result in enumerate(batch_results["individual_results"][:5]):
        print(f"\nDocument {i}:")
        print(f"  Tokens: {result['num_tokens']}")
        print(f"  Total bits: {result['total_bits']}")
        print(f"  Bits per token: {result['bits_per_token']:.2f}")
        print(f"  Compression ratio: {result['compression_ratio']:.2f}x")


def _run_baseline_comparison(args: CompressArgs) -> None:
    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings = np.load(args.embeddings_path)

    baselines = [
        GzipCompressionBaseline(),
        PQCompressionBaseline(num_subspaces=48, num_centroids=256),
    ]

    use_case = CompressionComparisonUseCase(baselines=baselines)
    results = use_case.execute(embeddings)

    print("\n" + "=" * 70)
    print("COMPRESSION BASELINE COMPARISON")
    print("=" * 70)
    print(f"{'Method':<25} {'Ratio':>10} {'Bits/Token':>12} {'MSE':>12}")
    print("-" * 70)

    for result in results:
        mse_str = f"{result['reconstruction_error']:.6f}" if result["reconstruction_error"] is not None else "N/A"
        print(
            f"{result['method']:<25} "
            f"{result['compression_ratio']:>10.2f}x "
            f"{result['bits_per_token']:>12.2f} "
            f"{mse_str:>12}"
        )

    print("=" * 70)


def main(args: CompressArgs) -> None:
    if args.compare_baselines:
        _run_baseline_comparison(args)
    else:
        _run_vq_analysis(args)


if __name__ == "__main__":
    main(tyro.cli(CompressArgs))
