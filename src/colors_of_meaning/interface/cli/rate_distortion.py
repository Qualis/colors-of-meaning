import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy.typing as npt
import tyro

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.rate_distortion_sweep_use_case import (
    BaselineFactory,
    EvaluateUseCaseFactory,
    RateDistortionSweepUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.rate_distortion_point import (
    RateDistortionFrontier,
    RateDistortionPoint,
)
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.domain.service.compression_baseline import CompressionBaseline
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import (
    AGNewsDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
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
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.color_vq_compression_baseline import (
    ColorVqCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.gzip_compression_baseline import (
    GzipCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

COLOR_VQ = "color_vq"
GZIP = "gzip"
PQ = "pq"
DEFAULT_BUDGETS = [2, 4, 8, 16]
DEFAULT_METHODS = [COLOR_VQ, GZIP, PQ]
PQ_BITS_PER_SUBQUANTIZER = 3


@dataclass
class RateDistortionArgs:
    config: str = "configs/base.yaml"
    dataset: Literal["ag_news", "imdb", "newsgroups"] = "ag_news"
    source: Literal["dataset", "documents"] = "dataset"
    documents_dir: str = "./documents"
    min_paragraph_chars: int = 200
    paragraphs_per_work: int = 60
    split_strategy: Literal["work", "paragraph"] = "work"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    budgets: List[int] = field(default_factory=lambda: list(DEFAULT_BUDGETS))
    methods: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    model_path: str = "artifacts/models/projector.pth"
    mapper_type: str = "unconstrained"
    distance: str = "wasserstein"
    k_neighbors: int = 5
    with_accuracy: bool = False
    max_samples: Optional[int] = 400
    output_path: str = "reports/rate_distortion.md"
    figure_path: str = "reports/figures/rate_distortion.png"


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    adapters: Dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "newsgroups": NewsgroupsDatasetAdapter,
    }
    return adapters[dataset_name]()


def _build_dataset_repository(args: RateDistortionArgs) -> DatasetRepository:
    if args.source == "documents":
        return DocumentCorpusDatasetAdapter(
            documents_dir=args.documents_dir,
            min_paragraph_chars=args.min_paragraph_chars,
            paragraphs_per_work=args.paragraphs_per_work,
            split_strategy=args.split_strategy,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
    return _setup_dataset(args.dataset)


def _create_distance_calculator(
    distance: str, codebook: ColorCodebook, config: SynestheticConfig
) -> DistanceCalculator:
    if distance == "wasserstein":
        return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)
    if distance == "jensen_shannon":
        return JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    raise ValueError(f"Unknown distance: {distance}")


def _pq_subquantizers(budget: int) -> int:
    return max(1, int(round(math.log2(budget))))


def _build_baseline_factory(
    color_mapper: ColorMapper, config: SynestheticConfig, primary_budget: int
) -> BaselineFactory:
    def build(method: str, budget: int) -> Optional[CompressionBaseline]:
        if method == COLOR_VQ:
            codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=budget)
            return ColorVqCompressionBaseline(codebook=codebook, color_mapper=color_mapper)
        if method == PQ:
            return PQCompressionBaseline(
                num_subspaces=_pq_subquantizers(budget),
                num_centroids=2**PQ_BITS_PER_SUBQUANTIZER,
                seed=config.training.seed,
            )
        if method == GZIP:
            return GzipCompressionBaseline() if budget == primary_budget else None
        raise ValueError(f"Unknown method: {method}")

    return build


def _build_evaluate_factory(
    args: RateDistortionArgs,
    config: SynestheticConfig,
    dataset_repository: DatasetRepository,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
) -> EvaluateUseCaseFactory:
    def build(method: str, budget: int) -> Optional[EvaluateUseCase]:
        if method != COLOR_VQ:
            return None
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=budget)
        encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook))
        distance_calculator = _create_distance_calculator(args.distance, codebook, config)
        classifier = ColorHistogramClassifier(
            embedding_adapter, encode_use_case, distance_calculator, k=args.k_neighbors
        )
        return EvaluateUseCase(classifier, SklearnMetricsCalculator(), dataset_repository)

    return build


def _encode_evaluation_embeddings(
    dataset_repository: DatasetRepository,
    embedding_adapter: SentenceEmbeddingAdapter,
    args: RateDistortionArgs,
    config: SynestheticConfig,
) -> npt.NDArray:
    samples = dataset_repository.get_samples(
        split=config.dataset.test_split, max_samples=args.max_samples, seed=config.training.seed
    )
    texts = [sample.text for sample in samples]
    return embedding_adapter.encode_batch(texts, batch_size=config.training.batch_size)


def _run_sweep(
    args: RateDistortionArgs,
    config: SynestheticConfig,
    dataset_repository: DatasetRepository,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    embeddings: npt.NDArray,
    correlation_id: str,
) -> RateDistortionFrontier:
    baseline_factory = _build_baseline_factory(color_mapper, config, args.budgets[0])
    evaluate_factory = _build_evaluate_factory(args, config, dataset_repository, embedding_adapter, color_mapper)
    use_case = RateDistortionSweepUseCase(baseline_factory, evaluate_use_case_factory=evaluate_factory)
    return use_case.execute(
        embeddings,
        budgets=args.budgets,
        methods=args.methods,
        with_accuracy=args.with_accuracy,
        max_samples=args.max_samples,
        seed=config.training.seed,
        correlation_id=correlation_id,
    )


def _distortion_unit(method: str) -> str:
    return "ΔE" if method == COLOR_VQ else "MSE"


def _format_accuracy(accuracy: Optional[float]) -> str:
    return "n/a" if accuracy is None else f"{accuracy:.4f}"


def _group_by_budget(frontier: RateDistortionFrontier) -> Dict[float, List[RateDistortionPoint]]:
    groups: Dict[float, List[RateDistortionPoint]] = {}
    for point in sorted(frontier.points, key=lambda candidate: (candidate.bits_per_token, candidate.method)):
        groups.setdefault(point.bits_per_token, []).append(point)
    return groups


def _matched_budget_groups(frontier: RateDistortionFrontier) -> List[List[RateDistortionPoint]]:
    groups = _group_by_budget(frontier)
    return [points for points in groups.values() if len({point.method for point in points}) >= 2]


def _point_rows(frontier: RateDistortionFrontier) -> List[str]:
    rows = ["| method | bits/token | distortion | metric | accuracy |", "|---|---|---|---|---|"]
    for point in sorted(frontier.points, key=lambda candidate: (candidate.method, candidate.bits_per_token)):
        rows.append(
            f"| {point.method} | {point.bits_per_token:.2f} | {point.reconstruction_error:.6f} | "
            f"{_distortion_unit(point.method)} | {_format_accuracy(point.accuracy)} |"
        )
    return rows


def _matched_budget_rows(frontier: RateDistortionFrontier) -> List[str]:
    rows = ["| bits/token | method | distortion | metric |", "|---|---|---|---|"]
    for points in _matched_budget_groups(frontier):
        for point in points:
            rows.append(
                f"| {point.bits_per_token:.2f} | {point.method} | {point.reconstruction_error:.6f} | "
                f"{_distortion_unit(point.method)} |"
            )
    return rows


def _pareto_rows(frontier: RateDistortionFrontier) -> List[str]:
    rows = ["| method | bits/token | distortion | metric |", "|---|---|---|---|"]
    for point in sorted(frontier.pareto_envelope(), key=lambda candidate: candidate.bits_per_token):
        rows.append(
            f"| {point.method} | {point.bits_per_token:.2f} | {point.reconstruction_error:.6f} | "
            f"{_distortion_unit(point.method)} |"
        )
    return rows


def _provenance_line() -> str:
    import numpy
    import sklearn  # type: ignore

    return f"Library versions: numpy {numpy.__version__}, scikit-learn {sklearn.__version__}."


def _source_flags(args: RateDistortionArgs) -> str:
    if args.source == "documents":
        return (
            f"--source documents --documents-dir {args.documents_dir} "
            f"--split-strategy {args.split_strategy} --min-paragraph-chars {args.min_paragraph_chars} "
            f"--paragraphs-per-work {args.paragraphs_per_work} "
            f"--validation-fraction {args.validation_fraction} --test-fraction {args.test_fraction}"
        )
    return f"--dataset {args.dataset}"


def _reproduce_command(args: RateDistortionArgs) -> str:
    budgets = " ".join(str(budget) for budget in args.budgets)
    methods = " ".join(args.methods)
    accuracy_flag = " --with-accuracy" if args.with_accuracy else ""
    return (
        f"tox -e rate_distortion -- {_source_flags(args)} --budgets {budgets} "
        f"--methods {methods}{accuracy_flag} --distance {args.distance} "
        f"--max-samples {args.max_samples} --config {args.config} "
        f"--output-path {args.output_path} --figure-path {args.figure_path}"
    )


def _report_lines(args: RateDistortionArgs, frontier: RateDistortionFrontier) -> List[str]:
    return [
        "# Rate-distortion frontier for semantic color compression",
        "",
        "The ~1024:1 headline is one operating point; this report measures the whole frontier.",
        "Each codec is swept across bit budgets and its native distortion recorded: color-VQ over",
        "grid resolutions (bits = log2(bins)), Product Quantization over subquantizers matched to the",
        "same bits, and gzip as a single data-dependent point. The color codec additionally records a",
        "downstream retrieval accuracy at each budget, so the cost of compression is shown in both",
        "perceptual distortion (ΔE for color-VQ, MSE for gzip/PQ) and task accuracy at matched budgets.",
        "",
        _provenance_line(),
        "",
        "## Rate-distortion points",
        "",
        *_point_rows(frontier),
        "",
        "## Matched-budget comparison",
        "",
        *_matched_budget_rows(frontier),
        "",
        "## Pareto frontier",
        "",
        "The envelope is the geometric lower-left set over (bits, native distortion). Distortion",
        "metrics differ across codecs (ΔE for color-VQ, MSE for gzip/PQ), so cross-codec domination",
        "is not directly comparable; read each codec's own curve in the figure rather than comparing",
        "ΔE against MSE as if they were one number.",
        "",
        *_pareto_rows(frontier),
        "",
        "## Reproduce",
        "",
        "```bash",
        _reproduce_command(args),
        "```",
        "",
    ]


def _print_table(frontier: RateDistortionFrontier) -> None:
    print("\n=== Rate-Distortion Frontier ===")
    for line in _point_rows(frontier):
        print(line)
    print("\n=== Matched-Budget Comparison ===")
    for line in _matched_budget_rows(frontier):
        print(line)


def _write_report(args: RateDistortionArgs, frontier: RateDistortionFrontier) -> None:
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, frontier)), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _render_figure(frontier: RateDistortionFrontier, figure_path: str) -> None:
    MatplotlibFigureRenderer().render_rate_distortion(frontier, figure_path)
    print(f"Saved {figure_path}")


def _log_startup(args: RateDistortionArgs, config: SynestheticConfig, correlation_id: str) -> None:
    logger.info(
        "Starting rate-distortion sweep",
        extra={
            "correlation_id": correlation_id,
            "source": args.source,
            "dataset": args.dataset,
            "budgets": args.budgets,
            "methods": args.methods,
            "with_accuracy": args.with_accuracy,
            "seed": config.training.seed,
        },
    )


def main(args: RateDistortionArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    correlation_id = str(uuid.uuid4())
    dataset_repository = _build_dataset_repository(args)
    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    _log_startup(args, config, correlation_id)
    embeddings = _encode_evaluation_embeddings(dataset_repository, embedding_adapter, args, config)
    frontier = _run_sweep(args, config, dataset_repository, embedding_adapter, color_mapper, embeddings, correlation_id)
    _print_table(frontier)
    _write_report(args, frontier)
    _render_figure(frontier, args.figure_path)


if __name__ == "__main__":
    main(tyro.cli(RateDistortionArgs))
