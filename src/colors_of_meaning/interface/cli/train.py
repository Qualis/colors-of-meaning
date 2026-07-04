import logging
import uuid
import numpy as np
import numpy.typing as npt
import tyro
from dataclasses import dataclass
from typing import Literal, Optional, cast

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig, StructuredMapperConfig
from colors_of_meaning.shared.determinism import seed_everything
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.evaluation.validation_accuracy_checkpoint_selector import (
    ValidationAccuracyCheckpointSelector,
)
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.color_codebook_factory import ColorCodebookFactory
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.learned_color_codebook_factory import (
    LearnedColorCodebookFactory,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator import (
    SpearmanStructurePreservationEvaluator,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
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
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository

logger = logging.getLogger(__name__)

SUPPORTED_SENTIMENT_SOURCES = {"labels", "none"}


@dataclass
class TrainArgs:
    config: str = "configs/base.yaml"
    dataset_path: str = "data/train.txt"
    output_model: str = "artifacts/models/projector.pth"
    output_codebook: str = "codebook_4096"
    mapper_type: str = "unconstrained"
    codebook_mode: str = "uniform"
    deterministic: bool = False
    source: Literal["dataset", "documents"] = "dataset"
    documents_dir: str = "./documents"
    min_paragraph_chars: int = 200
    paragraphs_per_work: Optional[int] = None
    split_strategy: Literal["work", "paragraph"] = "work"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    select_on: Literal["structure", "validation"] = "structure"
    selection_train_samples: int = 300
    selection_validation_samples: int = 150
    selection_k: int = 5


def _create_codebook_factory(color_mapper: ColorMapper) -> ColorCodebookFactory:
    return LearnedColorCodebookFactory(color_mapper=color_mapper)


def _create_dataset_adapter(dataset_name: str) -> DatasetRepository:
    adapters: dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "20newsgroups": NewsgroupsDatasetAdapter,
    }
    adapter_class = adapters.get(dataset_name)
    if adapter_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(adapters.keys())}")
    return adapter_class()


def _create_color_mapper(args: TrainArgs, config: SynestheticConfig) -> ColorMapper:
    color_mapper = create_color_mapper(args.mapper_type, config)
    if args.mapper_type == "structured":
        _log_structured_setup(config)
    return color_mapper


def _structured_config(config: SynestheticConfig) -> StructuredMapperConfig:
    return cast(StructuredMapperConfig, config.structured_mapper)


def _log_structured_setup(config: SynestheticConfig) -> None:
    structured_config = _structured_config(config)
    logger.info(
        "Configured structured mapper honest axes",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "sentiment_source": structured_config.sentiment_source,
            "concreteness_resource": structured_config.concreteness_resource,
        },
    )


def _load_supervised_data(config: SynestheticConfig) -> tuple:
    dataset_adapter = _create_dataset_adapter(config.dataset.name)
    samples = dataset_adapter.get_samples(
        split=config.dataset.train_split,
        max_samples=config.dataset.max_samples,
        seed=config.training.seed,
    )
    texts = [sample.text for sample in samples]
    labels = np.array([sample.label for sample in samples])
    return texts, labels


def _load_texts_from_file(dataset_path: str) -> list:
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _configure_supervised_mapper(
    color_mapper: ColorMapper,
    labels: npt.NDArray,
) -> None:
    if isinstance(color_mapper, SupervisedPyTorchColorMapper):
        color_mapper.set_training_labels(labels)


def _configure_structured_mapper(
    color_mapper: ColorMapper,
    texts: list,
    sentiment_scores: Optional[npt.NDArray],
) -> None:
    if not isinstance(color_mapper, StructuredPyTorchColorMapper):
        return
    color_mapper.set_training_texts(texts)
    if sentiment_scores is not None:
        color_mapper.set_sentiment_scores(sentiment_scores)


def _uses_label_sentiment(args: TrainArgs, config: SynestheticConfig) -> bool:
    return args.mapper_type == "structured" and _structured_config(config).sentiment_source == "labels"


def _validate_sentiment_source(args: TrainArgs, config: SynestheticConfig) -> None:
    if args.mapper_type != "structured":
        return
    source = _structured_config(config).sentiment_source
    if source not in SUPPORTED_SENTIMENT_SOURCES:
        raise ValueError(f"Unknown sentiment_source: {source}. Supported: {sorted(SUPPORTED_SENTIMENT_SOURCES)}")


def _resolve_paragraphs_per_work(args: TrainArgs, config: SynestheticConfig) -> int:
    if args.paragraphs_per_work is not None:
        return args.paragraphs_per_work
    return config.dataset.paragraphs_per_work


def _build_document_corpus(args: TrainArgs, config: SynestheticConfig) -> DocumentCorpusDatasetAdapter:
    return DocumentCorpusDatasetAdapter(
        documents_dir=args.documents_dir,
        min_paragraph_chars=args.min_paragraph_chars,
        paragraphs_per_work=_resolve_paragraphs_per_work(args, config),
        split_strategy=args.split_strategy,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )


def _load_documents_data(args: TrainArgs, config: SynestheticConfig) -> tuple:
    print(f"Loading document corpus train split from {args.documents_dir}...")
    samples = _build_document_corpus(args, config).get_samples(
        split=config.dataset.train_split, max_samples=config.dataset.max_samples, seed=config.training.seed
    )
    texts = [sample.text for sample in samples]
    if args.mapper_type == "supervised":
        return texts, np.array([sample.label for sample in samples]), None
    return texts, None, None


def _load_training_data(args: TrainArgs, config: SynestheticConfig) -> tuple:
    _validate_sentiment_source(args, config)
    if args.source == "documents":
        return _load_documents_data(args, config)
    if args.mapper_type == "supervised":
        print(f"Loading labeled dataset: {config.dataset.name}...")
        texts, labels = _load_supervised_data(config)
        return texts, labels, None
    if _uses_label_sentiment(args, config):
        print(f"Loading sentiment-labeled dataset: {config.dataset.name}...")
        texts, sentiment_scores = _load_supervised_data(config)
        return texts, None, sentiment_scores
    print(f"Loading dataset from {args.dataset_path}...")
    return _load_texts_from_file(args.dataset_path), None, None


def _apply_determinism(args: TrainArgs, config: SynestheticConfig) -> None:
    seed_everything(config.training.seed, deterministic=args.deterministic)
    logger.info(
        "Applied training determinism settings",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "seed": config.training.seed,
            "deterministic": args.deterministic,
        },
    )


def _select_evaluation_embeddings(
    embeddings: npt.NDArray,
    seed: int,
    max_evaluation_samples: int = 256,
) -> npt.NDArray:
    generator = np.random.default_rng(seed)
    sample_count = min(len(embeddings), max_evaluation_samples)
    selection = generator.permutation(len(embeddings))[:sample_count]
    return np.asarray(embeddings[selection])


def main(args: TrainArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    _apply_determinism(args, config)

    texts, labels, sentiment_scores = _load_training_data(args, config)

    print(f"Encoding {len(texts)} texts with sentence embeddings...")
    embedding_adapter = SentenceEmbeddingAdapter()
    embeddings = embedding_adapter.encode_batch(texts, batch_size=config.training.batch_size, show_progress=True)

    color_mapper = _create_color_mapper(args, config)

    if labels is not None:
        _configure_supervised_mapper(color_mapper, labels)
    _configure_structured_mapper(color_mapper, texts, sentiment_scores)

    checkpoint_selector = _maybe_build_validation_selector(args, config, color_mapper, embeddings, labels)
    _execute_training(args, config, color_mapper, embeddings, checkpoint_selector)


def _maybe_build_validation_selector(
    args: TrainArgs,
    config: SynestheticConfig,
    color_mapper: ColorMapper,
    train_embeddings: npt.NDArray,
    train_labels: Optional[npt.NDArray],
) -> Optional[ValidationAccuracyCheckpointSelector]:
    if args.select_on != "validation":
        return None
    if args.source != "documents":
        raise ValueError("--select-on validation requires --source documents")
    if train_labels is None:
        raise ValueError("--select-on validation requires --mapper-type supervised")
    return _build_validation_selector(args, config, color_mapper, train_embeddings, train_labels)


def _build_validation_selector(
    args: TrainArgs,
    config: SynestheticConfig,
    color_mapper: ColorMapper,
    train_embeddings: npt.NDArray,
    train_labels: npt.NDArray,
) -> ValidationAccuracyCheckpointSelector:
    print("Encoding validation split for checkpoint selection...")
    samples = _build_document_corpus(args, config).get_samples(
        split="validation", max_samples=args.selection_validation_samples, seed=config.training.seed
    )
    validation_embeddings = SentenceEmbeddingAdapter().encode_batch(
        [sample.text for sample in samples], batch_size=config.training.batch_size
    )
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=config.codebook.bins_per_dimension)
    encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook))
    reference = min(len(train_embeddings), args.selection_train_samples)
    return ValidationAccuracyCheckpointSelector(
        encode_use_case=encode_use_case,
        distance_calculator=JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon),
        train_embeddings=train_embeddings[:reference],
        train_labels=train_labels[:reference],
        validation_embeddings=validation_embeddings,
        validation_labels=np.array([sample.label for sample in samples]),
        k=args.selection_k,
    )


def _execute_training(
    args: TrainArgs,
    config: SynestheticConfig,
    color_mapper: ColorMapper,
    embeddings: npt.NDArray,
    checkpoint_selector: Optional[ValidationAccuracyCheckpointSelector] = None,
) -> None:
    print(f"Training {args.mapper_type} color projector for {config.training.epochs} epochs...")
    codebook_repo = FileColorCodebookRepository()
    evaluator = SpearmanStructurePreservationEvaluator(seed=config.training.seed)
    evaluation_embeddings = _select_evaluation_embeddings(embeddings, config.training.seed)
    use_case = TrainColorMappingUseCase(
        color_mapper=color_mapper,
        structure_preservation_evaluator=evaluator,
        codebook_repository=codebook_repo,
        codebook_factory=_create_codebook_factory(color_mapper),
        checkpoint_selector=checkpoint_selector,
    )

    score = use_case.execute(
        embeddings=embeddings,
        evaluation_embeddings=evaluation_embeddings,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        bins_per_dimension=config.codebook.bins_per_dimension,
        model_name=args.output_model,
        codebook_name=args.output_codebook,
        codebook_mode=args.codebook_mode,
        num_bins=config.codebook.num_bins,
        seed=config.training.seed,
    )

    print(f"Model saved to {args.output_model}")
    print(f"Codebook saved to artifacts/codebooks/{args.output_codebook}.pkl ({args.codebook_mode} mode)")
    _print_selection_score(args, score)


def _print_selection_score(args: TrainArgs, score: float) -> None:
    if args.select_on == "validation":
        print(f"Validation authorship accuracy (selected checkpoint): {score:.4f}")
    else:
        print(f"Structure-preservation correlation: {score:.4f}")


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
