import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import tyro

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_distance_fidelity_use_case import (
    EvaluateDistanceFidelityUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.evaluation_suite_use_case import (
    EvaluatedCell,
    EvaluationCell,
    EvaluationSuiteUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
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
from colors_of_meaning.infrastructure.evaluation.spearman_rank_correlation_calculator import (
    SpearmanRankCorrelationCalculator,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.interface.cli.eval import DistanceChoice, _create_distance_calculator
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

DEFAULT_DATASETS = ["ag_news", "imdb", "newsgroups"]
COLOR_BITS_PER_TOKEN = 12.0


@dataclass
class EvalSuiteArgs:
    config: str = "configs/agnews_full.yaml"
    datasets: List[str] = field(default_factory=lambda: list(DEFAULT_DATASETS))
    distance: DistanceChoice = "sliced"
    budget: int = 4000
    budgets: Optional[List[int]] = None
    fidelity_dataset: str = "ag_news"
    fidelity_samples: int = 1000
    fidelity_pairs: int = 1500
    fidelity_accuracy_delta: float = 0.0
    threshold_spearman: float = 0.95
    max_accuracy_delta: float = 1.0
    model_path: str = "artifacts/models/projector.pth"
    codebook_path: str = "codebook_4096"
    mapper_type: str = "unconstrained"
    k_neighbors: int = 5
    output_path: str = "reports/eval_results.md"


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    adapters: Dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "newsgroups": NewsgroupsDatasetAdapter,
    }
    return adapters[dataset_name]()


def _load_codebook(codebook_path: str) -> ColorCodebook:
    codebook = FileColorCodebookRepository().load(codebook_path)
    if codebook is None:
        raise FileNotFoundError(f"Codebook not found at {codebook_path}")
    return codebook


def _encode_documents(
    embedding_adapter: SentenceEmbeddingAdapter,
    encode_use_case: EncodeDocumentUseCase,
    samples: Sequence[EvaluationSample],
) -> List[ColoredDocument]:
    return [
        encode_use_case.execute(
            embedding_adapter.encode_document_sentences(sample.text), document_id=f"fidelity_{index}"
        )
        for index, sample in enumerate(samples)
    ]


def _build_color_classifier(
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    codebook: ColorCodebook,
    config: SynestheticConfig,
    distance: str,
    k_neighbors: int,
) -> ColorHistogramClassifier:
    quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
    encode_use_case = EncodeDocumentUseCase(quantized_mapper)
    distance_calculator = _create_distance_calculator(distance, codebook, config)
    return ColorHistogramClassifier(embedding_adapter, encode_use_case, distance_calculator, k=k_neighbors)


def _build_evaluate_use_case_factory(
    args: EvalSuiteArgs,
    config: SynestheticConfig,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    codebook: ColorCodebook,
) -> Callable[[EvaluationCell], EvaluateUseCase]:
    def build(cell: EvaluationCell) -> EvaluateUseCase:
        dataset_repository = _setup_dataset(cell.dataset)
        classifier = _build_color_classifier(
            embedding_adapter, color_mapper, codebook, config, cell.distance, args.k_neighbors
        )
        return EvaluateUseCase(classifier, SklearnMetricsCalculator(), dataset_repository)

    return build


def _cell_budgets(args: EvalSuiteArgs) -> List[int]:
    if args.budgets is None:
        return [args.budget for _ in args.datasets]
    if len(args.budgets) != len(args.datasets):
        raise ValueError("budgets must provide one budget per dataset")
    return list(args.budgets)


def _build_cells(args: EvalSuiteArgs) -> List[EvaluationCell]:
    return [
        EvaluationCell(
            dataset=dataset_name,
            method="color",
            distance=args.distance,
            budget=budget,
            requires_fidelity=args.distance == "sliced",
            bits_per_token=COLOR_BITS_PER_TOKEN,
        )
        for dataset_name, budget in zip(args.datasets, _cell_budgets(args))
    ]


def _run_fidelity_gate(
    args: EvalSuiteArgs,
    config: SynestheticConfig,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    codebook: ColorCodebook,
) -> DistanceFidelity:
    dataset_repository = _setup_dataset(args.fidelity_dataset)
    samples = dataset_repository.get_samples(
        split=config.dataset.test_split, max_samples=args.fidelity_samples, seed=config.training.seed
    )
    encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook))
    documents = _encode_documents(embedding_adapter, encode_use_case, samples)
    fidelity_use_case = EvaluateDistanceFidelityUseCase(
        proxy_calculator=SlicedWassersteinDistanceCalculator(codebook=codebook, seed=config.training.seed),
        exact_calculator=WassersteinDistanceCalculator(codebook=codebook),
        rank_correlation_calculator=SpearmanRankCorrelationCalculator(),
    )
    return fidelity_use_case.execute(
        documents,
        pair_count=args.fidelity_pairs,
        seed=config.training.seed,
        accuracy_delta=args.fidelity_accuracy_delta,
        threshold_spearman=args.threshold_spearman,
        max_accuracy_delta=args.max_accuracy_delta,
    )


def _fidelity_rows(fidelity: DistanceFidelity) -> List[str]:
    faithful = "yes" if fidelity.is_faithful else "no"
    return [
        "| proxy | exact | spearman | accuracy_delta (pts) | pairs | threshold | max_delta | faithful |",
        "|-------|-------|----------|----------------------|-------|-----------|-----------|----------|",
        f"| sliced_wasserstein | wasserstein | {fidelity.spearman:.4f} | {fidelity.accuracy_delta:.4f} | "
        f"{fidelity.pair_count} | {fidelity.threshold_spearman} | {fidelity.max_accuracy_delta} | {faithful} |",
    ]


def _result_rows(evaluated_cells: Sequence[EvaluatedCell]) -> List[str]:
    rows = [
        "| dataset | method | distance | budget | accuracy | macro_f1 | bits/token | seconds |",
        "|---------|--------|----------|--------|----------|----------|------------|---------|",
    ]
    for evaluated in evaluated_cells:
        rows.append(_result_row(evaluated))
    return rows


def _result_row(evaluated: EvaluatedCell) -> str:
    cell = evaluated.cell
    result = evaluated.result
    budget = "full" if cell.budget is None else str(cell.budget)
    bits = "n/a" if cell.bits_per_token is None else f"{cell.bits_per_token:.2f}"
    return (
        f"| {cell.dataset} | {cell.method} | {cell.distance} | {budget} | {result.accuracy:.4f} | "
        f"{result.macro_f1:.4f} | {bits} | {evaluated.seconds:.1f} |"
    )


def _provenance_line() -> str:
    import numpy
    import ot  # type: ignore
    import scipy  # type: ignore

    return f"Library versions: numpy {numpy.__version__}, scipy {scipy.__version__}, POT {ot.__version__}."


def _reproduce_command(args: EvalSuiteArgs) -> str:
    budget_flag = (
        f"--budgets {' '.join(str(budget) for budget in args.budgets)}"
        if args.budgets is not None
        else f"--budget {args.budget}"
    )
    return (
        f"tox -e eval_suite -- --datasets {' '.join(args.datasets)} --distance {args.distance} "
        f"{budget_flag} --fidelity-accuracy-delta {args.fidelity_accuracy_delta} "
        f"--config {args.config} --mapper-type {args.mapper_type}"
    )


def _report_lines(
    fidelity: DistanceFidelity, evaluated_cells: Sequence[EvaluatedCell], args: EvalSuiteArgs
) -> List[str]:
    command = _reproduce_command(args)
    return [
        "# Scaled, multi-dataset evaluation of the color method",
        "",
        "Committed evidence for the color method beyond the 400-sample AG News budget. Every row is produced",
        "by the command below; the sliced-Wasserstein proxy is only trusted once it clears the fidelity gate.",
        "",
        _provenance_line(),
        "",
        "## Distance proxy fidelity gate",
        "",
        *_fidelity_rows(fidelity),
        "",
        "## Results",
        "",
        "Classification metrics only. Retrieval quality (label-based MRR and recall@k) is measured "
        "separately via `eval --task retrieval` and is not reported as a constant here.",
        "",
        *_result_rows(evaluated_cells),
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
    ]


def _write_report(
    output_path: str, fidelity: DistanceFidelity, evaluated_cells: Sequence[EvaluatedCell], args: EvalSuiteArgs
) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(fidelity, evaluated_cells, args)), encoding="utf-8")


def _print_table(fidelity: DistanceFidelity, evaluated_cells: Sequence[EvaluatedCell]) -> None:
    print("\n=== Distance Proxy Fidelity ===")
    for line in _fidelity_rows(fidelity):
        print(line)
    print("\n=== Scaled Evaluation Results ===")
    for line in _result_rows(evaluated_cells):
        print(line)


def _log_startup(args: EvalSuiteArgs, config: SynestheticConfig) -> None:
    logger.info(
        "Starting scaled evaluation suite",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "datasets": args.datasets,
            "distance": args.distance,
            "budget": args.budget,
            "seed": config.training.seed,
        },
    )


def main(args: EvalSuiteArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    codebook = _load_codebook(args.codebook_path)
    _log_startup(args, config)
    fidelity = _run_fidelity_gate(args, config, embedding_adapter, color_mapper, codebook)
    factory = _build_evaluate_use_case_factory(args, config, embedding_adapter, color_mapper, codebook)
    suite = EvaluationSuiteUseCase(factory, seed=config.training.seed)
    evaluated_cells = suite.execute(_build_cells(args), fidelity)
    _print_table(fidelity, evaluated_cells)
    _write_report(args.output_path, fidelity, evaluated_cells, args)


if __name__ == "__main__":
    main(tyro.cli(EvalSuiteArgs))
