import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Tuple

import numpy.typing as npt
import tyro

from colors_of_meaning.application.use_case.ablation_sweep_use_case import (
    AblationSweepUseCase,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.domain.model.ablation_result import AblationResult
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import (
    AGNewsDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter import (
    IMDBDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter import (
    NewsgroupsDatasetAdapter,
)
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator import (
    SpearmanStructurePreservationEvaluator,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.cosine_histogram_distance_calculator import (
    CosineHistogramDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

DEFAULT_CODEBOOKS = ["grid1024=codebook_1024", "grid4096=codebook_4096", "learned=codebook_learned"]
DEFAULT_METRICS = ["wasserstein", "jensen_shannon", "cosine"]
COLOR_HISTOGRAM_BITS_PER_TOKEN = 12.0
MAX_STRUCTURE_SAMPLES = 256


@dataclass
class AblateArgs:
    config: str = "configs/base.yaml"
    dataset: Literal["ag_news", "imdb", "newsgroups"] = "ag_news"
    codebooks: List[str] = field(default_factory=lambda: list(DEFAULT_CODEBOOKS))
    metrics: List[str] = field(default_factory=lambda: list(DEFAULT_METRICS))
    model_path: str = "artifacts/models/projector.pth"
    mapper_type: str = "unconstrained"
    k_neighbors: int = 5
    output_path: str = "artifacts/ablations/sweep.json"


def _create_distance_calculator(
    metric_name: str, codebook: ColorCodebook, config: SynestheticConfig
) -> DistanceCalculator:
    if metric_name == "wasserstein":
        return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)
    if metric_name == "jensen_shannon":
        return JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    if metric_name == "cosine":
        return CosineHistogramDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    raise ValueError(f"Unknown metric: {metric_name}")


def _parse_codebook_specification(specification: str) -> Tuple[str, ColorCodebook]:
    label, separator, path = specification.partition("=")
    if not separator:
        raise ValueError(f"Codebook specification must be label=path, got {specification}")
    return label, load_codebook(path)


def _build_classifier_factory(
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    config: SynestheticConfig,
    k_neighbors: int,
) -> Callable[[ColorCodebook, str], Classifier]:
    def build(codebook: ColorCodebook, metric_name: str) -> Classifier:
        quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
        encode_use_case = EncodeDocumentUseCase(quantized_mapper)
        distance_calculator = _create_distance_calculator(metric_name, codebook, config)
        return ColorHistogramClassifier(embedding_adapter, encode_use_case, distance_calculator, k=k_neighbors)

    return build


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    adapters: Dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "newsgroups": NewsgroupsDatasetAdapter,
    }
    return adapters[dataset_name]()


def _structure_sample_budget(config: SynestheticConfig) -> int:
    if config.dataset.max_samples is None:
        return MAX_STRUCTURE_SAMPLES
    return min(config.dataset.max_samples, MAX_STRUCTURE_SAMPLES)


def _encode_evaluation_embeddings(
    dataset_repository: DatasetRepository,
    embedding_adapter: SentenceEmbeddingAdapter,
    config: SynestheticConfig,
) -> npt.NDArray:
    samples = dataset_repository.get_samples(
        split=config.dataset.test_split,
        max_samples=_structure_sample_budget(config),
        seed=config.training.seed,
    )
    texts = [sample.text for sample in samples]
    return embedding_adapter.encode_batch(texts, batch_size=config.training.batch_size)


def _result_to_row(result: AblationResult) -> Dict[str, object]:
    return {
        "codebook": result.codebook_label,
        "metric": result.metric_name,
        "accuracy": result.result.accuracy,
        "macro_f1": result.result.macro_f1,
        "structure_correlation": result.structure_correlation,
    }


def _write_artifact(results: List[AblationResult], output_path: str) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [_result_to_row(result) for result in results]
    with open(destination, "w") as artifact:
        json.dump(rows, artifact, indent=2)


def _print_table(results: List[AblationResult]) -> None:
    print("\n=== Ablation Sweep ===")
    print("codebook | metric | accuracy | macro_f1 | structure_correlation")
    for result in results:
        print(
            f"{result.codebook_label} | {result.metric_name} | "
            f"{result.result.accuracy:.4f} | {result.result.macro_f1:.4f} | {result.structure_correlation:.4f}"
        )


def _log_startup(args: AblateArgs, config: SynestheticConfig) -> None:
    logger.info(
        "Starting ablation sweep",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "dataset": args.dataset,
            "codebooks": args.codebooks,
            "metrics": args.metrics,
            "seed": config.training.seed,
        },
    )


def main(args: AblateArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    dataset_repository = _setup_dataset(args.dataset)
    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    codebooks = [_parse_codebook_specification(specification) for specification in args.codebooks]
    evaluation_embeddings = _encode_evaluation_embeddings(dataset_repository, embedding_adapter, config)
    _log_startup(args, config)
    use_case = AblationSweepUseCase(
        classifier_factory=_build_classifier_factory(embedding_adapter, color_mapper, config, args.k_neighbors),
        metrics_calculator=SklearnMetricsCalculator(),
        dataset_repository=dataset_repository,
        color_mapper=color_mapper,
        structure_preservation_evaluator=SpearmanStructurePreservationEvaluator(seed=config.training.seed),
        codebooks=codebooks,
        metric_names=args.metrics,
    )
    results = use_case.execute(
        evaluation_embeddings,
        bits_per_token=COLOR_HISTOGRAM_BITS_PER_TOKEN,
        max_samples=config.dataset.max_samples,
        seed=config.training.seed,
    )
    _print_table(results)
    _write_artifact(results, args.output_path)


if __name__ == "__main__":
    main(tyro.cli(AblateArgs))
