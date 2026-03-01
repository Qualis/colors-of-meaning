import tyro
from dataclasses import dataclass
from typing import Literal

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import (
    AGNewsDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter import (
    IMDBDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter import (
    NewsgroupsDatasetAdapter,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.tfidf_classifier import (
    TFIDFClassifier,
)
from colors_of_meaning.infrastructure.evaluation.hnsw_classifier import (
    HNSWClassifier,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import (
    EvaluateUseCase,
)
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


@dataclass
class EvalArgs:
    config: str = "configs/base.yaml"
    dataset: Literal["ag_news", "imdb", "newsgroups"] = "ag_news"
    method: Literal["color", "tfidf", "hnsw"] = "color"
    model_path: str = "artifacts/models/projector.pth"
    codebook_path: str = "codebook_4096"
    k_neighbors: int = 5


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    dataset_adapters = {
        "ag_news": (AGNewsDatasetAdapter, "Loading AG News dataset..."),
        "imdb": (IMDBDatasetAdapter, "Loading IMDB dataset..."),
        "newsgroups": (NewsgroupsDatasetAdapter, "Loading 20 Newsgroups dataset..."),
    }
    adapter_class, message = dataset_adapters[dataset_name]
    print(message)
    return adapter_class()


def _create_color_classifier(args: EvalArgs, config: SynestheticConfig) -> tuple:
    print("Using color histogram classifier...")
    from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper

    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
    )
    color_mapper.load_weights(args.model_path)
    codebook = FileColorCodebookRepository().load(args.codebook_path)
    if codebook is None:
        raise FileNotFoundError(f"Codebook not found at {args.codebook_path}")
    quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
    encode_use_case = EncodeDocumentUseCase(quantized_mapper)
    classifier = ColorHistogramClassifier(
        embedding_adapter, encode_use_case, WassersteinDistanceCalculator(), k=args.k_neighbors
    )
    return classifier, 12.0


def _create_classifier(args: EvalArgs, config: SynestheticConfig) -> tuple:
    if args.method == "color":
        return _create_color_classifier(args, config)
    elif args.method == "tfidf":
        print("Using TF-IDF baseline...")
        return TFIDFClassifier(), None
    elif args.method == "hnsw":
        print("Using HNSW baseline...")
        return HNSWClassifier(SentenceEmbeddingAdapter(), k=args.k_neighbors), None
    else:
        raise ValueError(f"Unknown method: {args.method}")


def _print_results(args: EvalArgs, result: EvaluationResult) -> None:
    print("\n=== Evaluation Results ===")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    print(f"MRR: {result.mrr:.4f}")
    if result.recall_at_k:
        for k, recall in sorted(result.recall_at_k.items()):
            print(f"Recall@{k}: {recall:.4f}")
    if result.bits_per_token is not None:
        print(f"Bits per token: {result.bits_per_token:.2f}")


def main(args: EvalArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    dataset_repo = _setup_dataset(args.dataset)
    classifier, bits_per_token = _create_classifier(args, config)
    evaluate_use_case = EvaluateUseCase(classifier, SklearnMetricsCalculator(), dataset_repo)
    max_samples = config.dataset.max_samples if hasattr(config.dataset, "max_samples") else None
    limit_msg = f" (limited to {max_samples} samples per split)" if max_samples else ""
    print(f"Evaluating on {args.dataset} with {args.method} method{limit_msg}...")
    result = evaluate_use_case.execute(bits_per_token=bits_per_token, max_samples=max_samples)
    _print_results(args, result)


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
