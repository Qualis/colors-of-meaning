import tyro
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
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
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_retriever import (
    ColorHistogramRetriever,
)
from colors_of_meaning.infrastructure.evaluation.embedding_retriever import (
    EmbeddingRetriever,
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
from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import (
    RetrievalEvaluateUseCase,
)
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.retrieval_evaluation import RetrievalEvaluation
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.retriever import Retriever

DistanceChoice = Literal["wasserstein", "sliced", "sinkhorn", "jensen_shannon"]
DEFAULT_SINKHORN_REG = 1.0


@dataclass
class EvalArgs:
    config: str = "configs/base.yaml"
    dataset: Literal["ag_news", "imdb", "newsgroups"] = "ag_news"
    method: Literal["color", "tfidf", "hnsw"] = "color"
    task: Literal["classification", "retrieval"] = "classification"
    distance: DistanceChoice = "wasserstein"
    model_path: str = "artifacts/models/projector.pth"
    codebook_path: str = "codebook_4096"
    k_neighbors: int = 5
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    mapper_type: str = "unconstrained"
    max_samples: Optional[int] = None
    source: Literal["dataset", "documents"] = "dataset"
    documents_dir: str = "./documents"
    min_paragraph_chars: int = 200
    paragraphs_per_work: Optional[int] = None
    split_strategy: Literal["work", "paragraph"] = "work"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    dataset_adapters = {
        "ag_news": (AGNewsDatasetAdapter, "Loading AG News dataset..."),
        "imdb": (IMDBDatasetAdapter, "Loading IMDB dataset..."),
        "newsgroups": (NewsgroupsDatasetAdapter, "Loading 20 Newsgroups dataset..."),
    }
    adapter_class, message = dataset_adapters[dataset_name]
    print(message)
    return adapter_class()


def _build_dataset_repository(args: EvalArgs, config: SynestheticConfig) -> DatasetRepository:
    if args.source == "documents":
        return _build_document_corpus(args, config)
    return _setup_dataset(args.dataset)


def _resolve_paragraphs_per_work(args: EvalArgs, config: SynestheticConfig) -> int:
    if args.paragraphs_per_work is not None:
        return args.paragraphs_per_work
    return config.dataset.paragraphs_per_work


def _build_document_corpus(args: EvalArgs, config: SynestheticConfig) -> DatasetRepository:
    print(f"Loading document corpus from {args.documents_dir}...")
    return DocumentCorpusDatasetAdapter(
        documents_dir=args.documents_dir,
        min_paragraph_chars=args.min_paragraph_chars,
        paragraphs_per_work=_resolve_paragraphs_per_work(args, config),
        split_strategy=args.split_strategy,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )


def _create_distance_calculator(
    distance: str, codebook: ColorCodebook, config: SynestheticConfig
) -> DistanceCalculator:
    if distance == "wasserstein":
        return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)
    if distance == "sliced":
        return SlicedWassersteinDistanceCalculator(codebook=codebook, seed=config.training.seed)
    if distance == "sinkhorn":
        return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=_resolved_sinkhorn_reg(config))
    if distance == "jensen_shannon":
        return JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    raise ValueError(f"Unknown distance: {distance}")


def _resolved_sinkhorn_reg(config: SynestheticConfig) -> float:
    return config.distance.sinkhorn_reg if config.distance.sinkhorn_reg else DEFAULT_SINKHORN_REG


def _build_color_components(args: EvalArgs, config: SynestheticConfig) -> tuple:
    from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper

    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    codebook = FileColorCodebookRepository().load(args.codebook_path)
    if codebook is None:
        raise FileNotFoundError(f"Codebook not found at {args.codebook_path}")
    quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
    encode_use_case = EncodeDocumentUseCase(quantized_mapper)
    distance_calculator = _create_distance_calculator(args.distance, codebook, config)
    return embedding_adapter, encode_use_case, distance_calculator


def _create_color_classifier(args: EvalArgs, config: SynestheticConfig) -> tuple:
    print("Using color histogram classifier...")
    embedding_adapter, encode_use_case, distance_calculator = _build_color_components(args, config)
    classifier = ColorHistogramClassifier(embedding_adapter, encode_use_case, distance_calculator, k=args.k_neighbors)
    return classifier, 12.0


def _create_color_retriever(args: EvalArgs, config: SynestheticConfig) -> tuple:
    print("Using color histogram retriever...")
    embedding_adapter, encode_use_case, distance_calculator = _build_color_components(args, config)
    retriever = ColorHistogramRetriever(embedding_adapter, encode_use_case, distance_calculator)
    return retriever, 12.0


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


def _create_retriever(args: EvalArgs, config: SynestheticConfig) -> tuple:
    if args.method == "color":
        return _create_color_retriever(args, config)
    if args.method == "hnsw":
        print("Using embedding retriever...")
        return EmbeddingRetriever(SentenceEmbeddingAdapter()), None
    if args.method == "tfidf":
        raise ValueError("Retrieval not supported for method 'tfidf' (classification-only baseline)")
    raise ValueError(f"Unknown method: {args.method}")


def _print_results(args: EvalArgs, result: EvaluationResult) -> None:
    print("\n=== Evaluation Results ===")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    if args.method == "color":
        print(f"Distance: {args.distance}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Macro F1: {result.macro_f1:.4f}")
    if result.recall_at_k:
        for k, recall in sorted(result.recall_at_k.items()):
            print(f"Recall@{k}: {recall:.4f}")
    if result.bits_per_token is not None:
        print(f"Bits per token: {result.bits_per_token:.2f}")


def _print_retrieval_results(args: EvalArgs, evaluation: RetrievalEvaluation) -> None:
    result = evaluation.result
    print("\n=== Retrieval Evaluation Results ===")
    print(f"Dataset: {args.dataset}")
    print(f"Method: {args.method}")
    if args.method == "color":
        print(f"Distance: {args.distance}")
    print("Label-based retrieval (relevance = shared class label)")
    print(f"MRR: {result.mrr:.4f}")
    for k, recall in sorted(result.recall_at_k.items()):
        print(f"Recall@{k}: {recall:.4f}")
    for k, ndcg in sorted(evaluation.ndcg_at_k.items()):
        print(f"NDCG@{k} (graded, label relevance): {ndcg:.4f}")
    if result.bits_per_token is not None:
        print(f"Bits per token: {result.bits_per_token:.2f}")


def _resolve_max_samples(args: EvalArgs, config: SynestheticConfig) -> Optional[int]:
    if args.max_samples is not None:
        return args.max_samples
    return config.dataset.max_samples if hasattr(config.dataset, "max_samples") else None


def _run_classification(args: EvalArgs, config: SynestheticConfig, dataset_repo: DatasetRepository) -> None:
    classifier, bits_per_token = _create_classifier(args, config)
    evaluate_use_case = EvaluateUseCase(classifier, SklearnMetricsCalculator(), dataset_repo)
    max_samples = _resolve_max_samples(args, config)
    limit_msg = f" (limited to {max_samples} samples per split)" if max_samples else ""
    print(f"Evaluating on {args.dataset} with {args.method} method{limit_msg}...")
    result = evaluate_use_case.execute(
        bits_per_token=bits_per_token, max_samples=max_samples, seed=config.training.seed
    )
    _print_results(args, result)


def _run_retrieval(args: EvalArgs, config: SynestheticConfig, dataset_repo: DatasetRepository) -> None:
    retriever: Retriever
    retriever, bits_per_token = _create_retriever(args, config)
    use_case = RetrievalEvaluateUseCase(retriever, SklearnMetricsCalculator(), dataset_repo, args.k_values)
    max_samples = _resolve_max_samples(args, config)
    limit_msg = f" (limited to {max_samples} samples per split)" if max_samples else ""
    print(f"Evaluating retrieval on {args.dataset} with {args.method} method{limit_msg}...")
    evaluation = use_case.execute(bits_per_token=bits_per_token, max_samples=max_samples, seed=config.training.seed)
    _print_retrieval_results(args, evaluation)


def main(args: EvalArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    dataset_repo = _build_dataset_repository(args, config)
    if args.task == "retrieval":
        _run_retrieval(args, config, dataset_repo)
        return
    _run_classification(args, config, dataset_repo)


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
