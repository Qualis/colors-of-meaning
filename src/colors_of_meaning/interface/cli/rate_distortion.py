import logging
import math
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

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
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
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
RateDistortionDistance = Literal["wasserstein", "sliced", "jensen_shannon"]
DEFAULT_DISTANCES: List[RateDistortionDistance] = ["wasserstein"]
NO_INVERSION_SUMMARY = (
    "No distance inverts: accuracy is highest at the widest budget, so the earlier inversion was a metric artifact."
)
UNIVERSAL_INVERSION_SUMMARY = (
    "Every distance inverts, so the drop past the peak budget is a property of the bit budget, not of the metric."
)
SINGLE_DISTANCE_SUMMARY = (
    "Only one distance was measured, so this run cannot separate a property of the bit budget from a property of "
    "the metric; re-run with a second --distance to attribute the shape."
)
PQ_BITS_PER_SUBQUANTIZER = 3


@dataclass(frozen=True)
class SweepRun:
    distance: str
    seed: int
    frontier: RateDistortionFrontier


@dataclass(frozen=True)
class RateAccuracyPoint:
    distance: str
    bits_per_token: float
    mean_accuracy: float
    stdev_accuracy: float
    seeds: int


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
    distance: List[RateDistortionDistance] = field(default_factory=lambda: list(DEFAULT_DISTANCES))
    seeds: Optional[List[int]] = None
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
    if distance == "sliced":
        return SlicedWassersteinDistanceCalculator(codebook=codebook, seed=config.training.seed)
    if distance == "jensen_shannon":
        return JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    raise ValueError(f"Unknown distance: {distance}")


def _pq_subquantizers(budget: int) -> int:
    return max(1, int(round(math.log2(budget))))


def _build_baseline_factory(color_mapper: ColorMapper, seed: int, primary_budget: int) -> BaselineFactory:
    def build(method: str, budget: int) -> Optional[CompressionBaseline]:
        if method == COLOR_VQ:
            codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=budget)
            return ColorVqCompressionBaseline(codebook=codebook, color_mapper=color_mapper)
        if method == PQ:
            return PQCompressionBaseline(
                num_subspaces=_pq_subquantizers(budget),
                num_centroids=2**PQ_BITS_PER_SUBQUANTIZER,
                seed=seed,
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
    distance: str,
) -> EvaluateUseCaseFactory:
    def build(method: str, budget: int) -> Optional[EvaluateUseCase]:
        if method != COLOR_VQ:
            return None
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=budget)
        encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook))
        distance_calculator = _create_distance_calculator(distance, codebook, config)
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


@dataclass(frozen=True, eq=False)
class SweepInputs:
    args: RateDistortionArgs
    config: SynestheticConfig
    dataset_repository: DatasetRepository
    embedding_adapter: SentenceEmbeddingAdapter
    color_mapper: ColorMapper
    embeddings: npt.NDArray
    correlation_id: str


def _resolved_seeds(args: RateDistortionArgs, config: SynestheticConfig) -> List[int]:
    return list(args.seeds) if args.seeds else [config.training.seed]


def _reject_repeats(name: str, values: Sequence[object]) -> None:
    labels = [str(value) for value in values]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{name} must not repeat a value, got {labels}")


def _reject_unknown_methods(methods: Sequence[str]) -> None:
    unknown = sorted(set(methods) - {COLOR_VQ, GZIP, PQ})
    if unknown:
        raise ValueError(f"Unknown method: {' '.join(unknown)}. Supported: {sorted([COLOR_VQ, GZIP, PQ])}")


def _reject_unusable_budgets(budgets: Sequence[int]) -> None:
    unusable = [budget for budget in budgets if budget < 2]
    if unusable:
        raise ValueError(f"budgets must be at least 2 bins per dimension, got {unusable}")


def _reject_empty_axes(args: RateDistortionArgs) -> None:
    axes = (
        ("distance", args.distance),
        ("budgets", args.budgets),
        ("methods", args.methods),
        ("seeds", args.seeds if args.seeds is not None else DEFAULT_DISTANCES),
    )
    for name, values in axes:
        if not values:
            raise ValueError(f"{name} must name at least one value")


def _reject_empty_sweep_axes(args: RateDistortionArgs) -> None:
    _reject_empty_axes(args)
    _reject_repeats("distance", args.distance)
    _reject_repeats("seeds", args.seeds or [])
    _reject_unknown_methods(args.methods)
    _reject_unusable_budgets(args.budgets)


def _sweep_grid(args: RateDistortionArgs, config: SynestheticConfig) -> List[Tuple[str, int]]:
    seeds = _resolved_seeds(args, config)
    if not args.with_accuracy:
        return [(args.distance[0], seeds[0])]
    return [(distance, seed) for distance in args.distance for seed in seeds]


def _run_sweep(inputs: SweepInputs, distance: str, seed: int, methods: Sequence[str]) -> RateDistortionFrontier:
    args = inputs.args
    baseline_factory = _build_baseline_factory(inputs.color_mapper, seed, args.budgets[0])
    evaluate_factory = _build_evaluate_factory(
        args, inputs.config, inputs.dataset_repository, inputs.embedding_adapter, inputs.color_mapper, distance
    )
    use_case = RateDistortionSweepUseCase(baseline_factory, evaluate_use_case_factory=evaluate_factory)
    return use_case.execute(
        inputs.embeddings,
        budgets=args.budgets,
        methods=list(methods),
        with_accuracy=args.with_accuracy,
        max_samples=args.max_samples,
        seed=seed,
        correlation_id=inputs.correlation_id,
    )


def _run_all_sweeps(inputs: SweepInputs) -> List[SweepRun]:
    grid = _sweep_grid(inputs.args, inputs.config)
    return [
        SweepRun(distance, seed, _run_sweep(inputs, distance, seed, _run_methods(inputs.args, index)))
        for index, (distance, seed) in enumerate(grid)
    ]


def _run_methods(args: RateDistortionArgs, index: int) -> Sequence[str]:
    return args.methods if index == 0 else [COLOR_VQ]


def _accuracy_by_budget(run: SweepRun) -> Dict[float, float]:
    return {
        point.bits_per_token: point.accuracy
        for point in run.frontier.points
        if point.method == COLOR_VQ and point.accuracy is not None
    }


def _rate_accuracy_points(runs: Sequence[SweepRun], distance: str) -> List[RateAccuracyPoint]:
    measurements: Dict[float, List[float]] = {}
    for run in runs:
        if run.distance != distance:
            continue
        for bits, accuracy in _accuracy_by_budget(run).items():
            measurements.setdefault(bits, []).append(accuracy)
    return [_rate_accuracy_point(distance, bits, values) for bits, values in sorted(measurements.items())]


def _rate_accuracy_point(distance: str, bits: float, values: List[float]) -> RateAccuracyPoint:
    return RateAccuracyPoint(
        distance=distance,
        bits_per_token=bits,
        mean_accuracy=statistics.fmean(values),
        stdev_accuracy=statistics.stdev(values) if len(values) > 1 else 0.0,
        seeds=len(values),
    )


def _diagnosis(args: RateDistortionArgs, runs: Sequence[SweepRun]) -> List[RateAccuracyPoint]:
    return [point for distance in args.distance for point in _rate_accuracy_points(runs, distance)]


def _diagnosis_rows(diagnosis: Sequence[RateAccuracyPoint]) -> List[str]:
    rows = ["| distance | bits/token | mean accuracy | sd | seeds |", "|---|---|---|---|---|"]
    for point in diagnosis:
        rows.append(
            f"| {point.distance} | {point.bits_per_token:.2f} | {point.mean_accuracy:.4f} | "
            f"{point.stdev_accuracy:.4f} | {point.seeds} |"
        )
    return rows


def _peak_budget(points: Sequence[RateAccuracyPoint]) -> RateAccuracyPoint:
    return max(points, key=lambda point: point.mean_accuracy)


def _is_inverted(points: Sequence[RateAccuracyPoint]) -> bool:
    return _peak_budget(points).bits_per_token < max(point.bits_per_token for point in points)


def _distance_verdict(distance: str, points: Sequence[RateAccuracyPoint]) -> str:
    peak = _peak_budget(points)
    widest = max(points, key=lambda point: point.bits_per_token)
    return (
        f"Under `{distance}` accuracy peaks at {peak.bits_per_token:.2f} bits ({peak.mean_accuracy:.4f}) and "
        f"reads {widest.mean_accuracy:.4f} at {widest.bits_per_token:.2f} bits."
    )


def _inverted_distances(inverted: Dict[str, bool]) -> List[str]:
    return [distance for distance, flag in inverted.items() if flag]


def _partial_inversion_summary(inverted_distances: Sequence[str]) -> str:
    named = ", ".join(f"`{name}`" for name in inverted_distances)
    return (
        f"The inversion appears only under {named}, so it is a metric artifact rather than a property of the "
        "bit budget."
    )


def _inversion_summary(inverted: Dict[str, bool]) -> str:
    if len(inverted) < 2:
        return SINGLE_DISTANCE_SUMMARY
    inverted_distances = _inverted_distances(inverted)
    if not inverted_distances:
        return NO_INVERSION_SUMMARY
    if len(inverted_distances) == len(inverted):
        return UNIVERSAL_INVERSION_SUMMARY
    return _partial_inversion_summary(inverted_distances)


def _measured_by_distance(
    args: RateDistortionArgs, diagnosis: Sequence[RateAccuracyPoint]
) -> Dict[str, List[RateAccuracyPoint]]:
    measured: Dict[str, List[RateAccuracyPoint]] = {}
    for distance in args.distance:
        points = _rate_accuracy_points_for(diagnosis, distance)
        if points:
            measured[distance] = points
    return measured


def _verdict_lines(measured: Dict[str, List[RateAccuracyPoint]]) -> List[str]:
    return [_distance_verdict(distance, points) for distance, points in measured.items()]


def _inversion_flags(measured: Dict[str, List[RateAccuracyPoint]]) -> Dict[str, bool]:
    return {distance: _is_inverted(points) for distance, points in measured.items()}


def _diagnosis_lines(args: RateDistortionArgs, diagnosis: Sequence[RateAccuracyPoint]) -> List[str]:
    if not diagnosis:
        return []
    measured = _measured_by_distance(args, diagnosis)
    return [
        "## Rate-accuracy diagnosis",
        "",
        "The accuracy column above is one distance at one seed. This section re-runs the rate-accuracy axis under",
        "every requested distance and seed at a fixed projector, so a peak that moves with the distance can be told",
        "apart from a peak that belongs to the bit budget. Seeds vary the evaluation sample draw.",
        "",
        *_diagnosis_rows(diagnosis),
        "",
        *_verdict_lines(measured),
        "",
        _inversion_summary(_inversion_flags(measured)),
        "",
    ]


def _rate_accuracy_points_for(diagnosis: Sequence[RateAccuracyPoint], distance: str) -> List[RateAccuracyPoint]:
    return [point for point in diagnosis if point.distance == distance]


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


def _seed_flag(args: RateDistortionArgs) -> str:
    return "" if not args.seeds else f" --seeds {' '.join(str(seed) for seed in args.seeds)}"


def _reproduce_command(args: RateDistortionArgs) -> str:
    budgets = " ".join(str(budget) for budget in args.budgets)
    methods = " ".join(args.methods)
    accuracy_flag = " --with-accuracy" if args.with_accuracy else ""
    return (
        f"tox -e rate_distortion -- {_source_flags(args)} --budgets {budgets} "
        f"--methods {methods}{accuracy_flag} --distance {' '.join(args.distance)}{_seed_flag(args)} "
        f"--max-samples {args.max_samples} --config {args.config} "
        f"--output-path {args.output_path} --figure-path {args.figure_path}"
    )


def _report_lines(
    args: RateDistortionArgs, frontier: RateDistortionFrontier, diagnosis: Sequence[RateAccuracyPoint] = ()
) -> List[str]:
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
        *_diagnosis_lines(args, diagnosis),
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


def _write_report(
    args: RateDistortionArgs, frontier: RateDistortionFrontier, diagnosis: Sequence[RateAccuracyPoint] = ()
) -> None:
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, frontier, diagnosis)), encoding="utf-8")
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
            "distances": args.distance,
            "seeds": _resolved_seeds(args, config),
        },
    )


def _build_sweep_inputs(args: RateDistortionArgs, config: SynestheticConfig, correlation_id: str) -> SweepInputs:
    dataset_repository = _build_dataset_repository(args)
    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    _log_startup(args, config, correlation_id)
    return SweepInputs(
        args=args,
        config=config,
        dataset_repository=dataset_repository,
        embedding_adapter=embedding_adapter,
        color_mapper=color_mapper,
        embeddings=_encode_evaluation_embeddings(dataset_repository, embedding_adapter, args, config),
        correlation_id=correlation_id,
    )


def _comparable_diagnosis(args: RateDistortionArgs, runs: Sequence[SweepRun]) -> List[RateAccuracyPoint]:
    if len(runs) < 2:
        return []
    return _diagnosis(args, runs)


def main(args: RateDistortionArgs) -> None:
    _reject_empty_sweep_axes(args)
    config = SynestheticConfig.from_yaml(args.config)
    inputs = _build_sweep_inputs(args, config, str(uuid.uuid4()))
    runs = _run_all_sweeps(inputs)
    frontier = runs[0].frontier
    diagnosis = _comparable_diagnosis(args, runs)
    _print_table(frontier)
    _write_report(args, frontier, diagnosis)
    _render_figure(frontier, args.figure_path)


if __name__ == "__main__":
    main(tyro.cli(RateDistortionArgs))
