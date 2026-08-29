import logging
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy.typing as npt
import tyro

from colors_of_meaning.application.use_case.compare_structure_objectives_use_case import (
    ArmRequest,
    CompareStructureObjectivesUseCase,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import (
    RetrievalEvaluateUseCase,
)
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.objective_comparison import (
    ObjectiveArmResult,
    ObjectiveComparison,
)
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
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
from colors_of_meaning.infrastructure.evaluation.color_histogram_retriever import (
    ColorHistogramRetriever,
)
from colors_of_meaning.infrastructure.evaluation.pca_projection_control import (
    PcaProjectionControl,
    rescale_preserving_ranks,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator import (
    SpearmanStructurePreservationEvaluator,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structure_objectives import (
    StructureObjective,
    cosine_centred,
    delta_e_correlation,
    margin_ranking,
)
from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_color_codebook_repository import (
    InMemoryColorCodebookRepository,
)
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook
from colors_of_meaning.interface.cli.eval import DistanceChoice, _create_distance_calculator
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

BASELINE_ARM = "cosine_centred"
NOISE_CONTROL = "noise"
PCA_CONTROL = "pca3"
UNCONSTRAINED_HEAD = "unconstrained_head"
UNCONSTRAINED_HEAD_PRECLAMP = "unconstrained_head_preclamp"
COMMITTED_CONTROL = "committed"

OBJECTIVE_ARMS: Dict[str, StructureObjective] = {
    BASELINE_ARM: cosine_centred,
    "delta_e_correlation": delta_e_correlation,
    "margin_ranking": margin_ranking,
}
UNTRAINED_ARMS = frozenset({NOISE_CONTROL, COMMITTED_CONTROL})
UNCONSTRAINED_HEAD_ARMS = frozenset({UNCONSTRAINED_HEAD, UNCONSTRAINED_HEAD_PRECLAMP})

DEFAULT_ARMS = list(OBJECTIVE_ARMS)
DEFAULT_CONTROLS = [
    NOISE_CONTROL,
    PCA_CONTROL,
    UNCONSTRAINED_HEAD,
    UNCONSTRAINED_HEAD_PRECLAMP,
    COMMITTED_CONTROL,
]
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49]
DEFAULT_DOWNSTREAM_SEEDS = [42, 43, 44]
KNOWN_ARMS = frozenset(OBJECTIVE_ARMS) | UNTRAINED_ARMS | UNCONSTRAINED_HEAD_ARMS | {PCA_CONTROL}
MAPPER_BACKED_ARMS = KNOWN_ARMS - {PCA_CONTROL, UNCONSTRAINED_HEAD_PRECLAMP}


@dataclass
class CompareObjectivesArgs:
    config: str = "configs/base.yaml"
    dataset: Literal["ag_news", "imdb", "newsgroups"] = "ag_news"
    arms: List[str] = field(default_factory=lambda: list(DEFAULT_ARMS))
    controls: List[str] = field(default_factory=lambda: list(DEFAULT_CONTROLS))
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    downstream_arms: List[str] = field(default_factory=list)
    downstream_controls: List[str] = field(default_factory=lambda: [COMMITTED_CONTROL])
    downstream_top_k: int = 2
    downstream_seeds: List[int] = field(default_factory=lambda: list(DEFAULT_DOWNSTREAM_SEEDS))
    budget: int = 4000
    distance: DistanceChoice = "sliced"
    codebook_path: str = "codebook_4096"
    k_neighbors: int = 5
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    train_samples: Optional[int] = None
    selection_samples: int = 256
    structure_samples: int = 256
    adoption_threshold_sigma: float = 2.0
    max_accuracy_drop: float = 0.01
    model_dir: str = "artifacts/objectives"
    committed_model_path: str = "artifacts/models/projector.pth"
    output_path: str = "reports/structure_objective.md"
    figure_path: str = "reports/figures/structure_objective.png"


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    adapters: Dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "newsgroups": NewsgroupsDatasetAdapter,
    }
    return adapters[dataset_name]()


def _training_arm(arm: str) -> str:
    return UNCONSTRAINED_HEAD if arm == UNCONSTRAINED_HEAD_PRECLAMP else arm


def _arm_objective(arm: str) -> StructureObjective:
    if arm not in KNOWN_ARMS:
        raise ValueError(f"Unknown arm: {arm}. Supported: {sorted(KNOWN_ARMS)}")
    return OBJECTIVE_ARMS.get(arm, cosine_centred)


def _reject_unknown_names(unknown: Sequence[str], label: str, supported: Sequence[str]) -> None:
    if unknown:
        raise ValueError(f"Unknown {label}: {' '.join(unknown)}. Supported: {sorted(supported)}")


def _reject_empty_axes(args: CompareObjectivesArgs) -> None:
    axes = (
        ("arms", args.arms),
        ("seeds", args.seeds),
        ("downstream-seeds", args.downstream_seeds),
        ("k-values", args.k_values),
    )
    for name, values in axes:
        if not values:
            raise ValueError(f"{name} must name at least one value")


def _reject_repeats(name: str, values: Sequence[object]) -> None:
    labels = [str(value) for value in values]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{name} must not repeat a value, got {labels}")


def _reject_repeated_axes(args: CompareObjectivesArgs) -> None:
    _reject_repeats("arms", args.arms)
    _reject_repeats("controls", args.controls)
    _reject_repeats("seeds", args.seeds)
    _reject_repeats("downstream-seeds", args.downstream_seeds)
    overlap = sorted(set(args.arms) & set(args.controls))
    if overlap:
        raise ValueError(f"an arm cannot also be a control, got {' '.join(overlap)}")


def _reject_missing_artifacts(args: CompareObjectivesArgs) -> None:
    if COMMITTED_CONTROL in args.controls and not Path(args.committed_model_path).is_file():
        raise FileNotFoundError(f"Committed projector not found: {args.committed_model_path}")


def _reject_unrunnable_request(args: CompareObjectivesArgs) -> None:
    _reject_empty_axes(args)
    _reject_repeated_axes(args)
    _reject_unknown_names(sorted((set(args.arms) | set(args.controls)) - KNOWN_ARMS), "arm", sorted(KNOWN_ARMS))
    _reject_unknown_names(
        sorted(set(args.downstream_arms) - MAPPER_BACKED_ARMS), "downstream arm", sorted(MAPPER_BACKED_ARMS)
    )
    if BASELINE_ARM not in args.arms:
        raise ValueError(f"arms must include the baseline arm {BASELINE_ARM}, got {args.arms}")
    if args.downstream_arms and BASELINE_ARM not in args.downstream_arms:
        raise ValueError(f"downstream arms must include the baseline arm {BASELINE_ARM}, got {args.downstream_arms}")
    _reject_missing_artifacts(args)


def _bits_per_token(codebook: ColorCodebook) -> float:
    return math.log2(codebook.num_bins)


def _build_mapper(arm: str, seed: int, config: SynestheticConfig) -> PyTorchColorMapper:
    return PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
        seed=seed,
        structure_objective=_arm_objective(arm),
        constrain_to_lab=arm not in UNCONSTRAINED_HEAD_ARMS,
    )


class TrainedMapperCache:
    def __init__(self, args: CompareObjectivesArgs, config: SynestheticConfig, selection_embeddings: npt.NDArray):
        self._args = args
        self._config = config
        self._selection_embeddings = selection_embeddings
        self._mappers: Dict[Tuple[str, int], PyTorchColorMapper] = {}

    def mapper(self, request: ArmRequest) -> PyTorchColorMapper:
        key = (_training_arm(request.arm), request.seed)
        if key not in self._mappers:
            self._mappers[key] = self._train(key[0], request)
        return self._mappers[key]

    def _train(self, arm: str, request: ArmRequest) -> PyTorchColorMapper:
        mapper = _build_mapper(arm, request.seed, self._config)
        if arm == COMMITTED_CONTROL:
            mapper.load_weights(self._args.committed_model_path)
            return mapper
        if arm in UNTRAINED_ARMS:
            return mapper
        self._fit(mapper, arm, request)
        mapper.epoch_checkpoints().clear()
        return mapper

    def _fit(self, mapper: PyTorchColorMapper, arm: str, request: ArmRequest) -> None:
        TrainColorMappingUseCase(
            color_mapper=mapper,
            structure_preservation_evaluator=SpearmanStructurePreservationEvaluator(seed=request.seed),
            codebook_repository=InMemoryColorCodebookRepository(),
        ).execute(
            embeddings=request.train_embeddings,
            evaluation_embeddings=self._selection_embeddings,
            epochs=self._config.training.epochs,
            learning_rate=self._config.training.learning_rate,
            bins_per_dimension=self._config.codebook.bins_per_dimension,
            model_name=str(Path(self._args.model_dir) / f"{arm}_seed{request.seed}.pth"),
            codebook_name=f"{arm}_seed{request.seed}",
            seed=request.seed,
        )


def _build_lab_colors_factory(cache: TrainedMapperCache) -> Callable[[ArmRequest], List[LabColor]]:
    def build(request: ArmRequest) -> List[LabColor]:
        if request.arm == PCA_CONTROL:
            control = PcaProjectionControl(seed=request.seed).fit(request.train_embeddings)
            return control.transform(request.eval_embeddings)
        mapper = cache.mapper(request)
        if request.arm == UNCONSTRAINED_HEAD_PRECLAMP:
            return rescale_preserving_ranks(mapper.embed_batch_to_coordinates(request.eval_embeddings))
        return mapper.embed_batch_to_lab(request.eval_embeddings)

    return build


def _build_evaluate_factory(
    cache: TrainedMapperCache,
    args: CompareObjectivesArgs,
    config: SynestheticConfig,
    embedding_adapter: SentenceEmbeddingAdapter,
    codebook: ColorCodebook,
) -> Callable[[ArmRequest], EvaluateUseCase]:
    def build(request: ArmRequest) -> EvaluateUseCase:
        encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(cache.mapper(request), codebook))
        classifier = ColorHistogramClassifier(
            embedding_adapter,
            encode_use_case,
            _create_distance_calculator(args.distance, codebook, config),
            k=args.k_neighbors,
        )
        return EvaluateUseCase(classifier, SklearnMetricsCalculator(), _setup_dataset(args.dataset))

    return build


def _build_retrieval_factory(
    cache: TrainedMapperCache,
    args: CompareObjectivesArgs,
    config: SynestheticConfig,
    embedding_adapter: SentenceEmbeddingAdapter,
    codebook: ColorCodebook,
) -> Callable[[ArmRequest], RetrievalEvaluateUseCase]:
    def build(request: ArmRequest) -> RetrievalEvaluateUseCase:
        encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(cache.mapper(request), codebook))
        retriever = ColorHistogramRetriever(
            embedding_adapter, encode_use_case, _create_distance_calculator(args.distance, codebook, config)
        )
        return RetrievalEvaluateUseCase(
            retriever, SklearnMetricsCalculator(), _setup_dataset(args.dataset), args.k_values
        )

    return build


def _encode_split(
    dataset_repository: DatasetRepository,
    embedding_adapter: SentenceEmbeddingAdapter,
    config: SynestheticConfig,
    split: str,
    max_samples: Optional[int],
) -> npt.NDArray:
    samples = dataset_repository.get_samples(split=split, max_samples=max_samples, seed=config.training.seed)
    return embedding_adapter.encode_batch([sample.text for sample in samples], batch_size=config.training.batch_size)


def _held_out_embeddings(
    dataset_repository: DatasetRepository,
    embedding_adapter: SentenceEmbeddingAdapter,
    args: CompareObjectivesArgs,
    config: SynestheticConfig,
) -> Tuple[npt.NDArray, npt.NDArray]:
    encoded = _encode_split(
        dataset_repository,
        embedding_adapter,
        config,
        config.dataset.test_split,
        args.selection_samples + args.structure_samples,
    )
    return encoded[: args.selection_samples], encoded[args.selection_samples :]


def _nominated_arms(args: CompareObjectivesArgs, comparison: ObjectiveComparison) -> List[str]:
    if args.downstream_arms:
        return list(args.downstream_arms)
    if args.downstream_top_k <= 0:
        return []
    ranked = sorted(comparison.challengers(), key=lambda arm: arm.strength(), reverse=True)
    return [comparison.baseline_arm] + [arm.arm for arm in ranked[: max(0, args.downstream_top_k - 1)]]


def _measurable_controls(args: CompareObjectivesArgs) -> List[str]:
    return [control for control in args.downstream_controls if control in args.controls]


def _downstream_arms(args: CompareObjectivesArgs, comparison: ObjectiveComparison) -> List[str]:
    nominated = [arm for arm in _nominated_arms(args, comparison) if arm in MAPPER_BACKED_ARMS]
    if not nominated:
        return []
    return nominated + _measurable_controls(args)


def _control_named(comparison: ObjectiveComparison, arm: str) -> Optional[ObjectiveArmResult]:
    for control in comparison.controls:
        if control.arm == arm:
            return control
    return None


def _adopted_result(comparison: ObjectiveComparison) -> ObjectiveArmResult:
    adopted = comparison.adopted_arm()
    return next(result for result in comparison.results if result.arm == adopted)


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _format_recall(recall_at_k: Optional[Dict[int, float]]) -> str:
    if not recall_at_k:
        return "n/a"
    return " ".join(f"{depth}:{value:.4f}" for depth, value in sorted(recall_at_k.items()))


def _arm_row(result: ObjectiveArmResult) -> str:
    return (
        f"| {result.arm} | {result.mean_rho:.4f} | {result.stdev_rho:.4f} | {result.seeds} | "
        f"{_format_metric(result.accuracy)} | {_format_metric(result.macro_f1)} | "
        f"{_format_metric(result.mrr)} | {_format_recall(result.recall_at_k)} |"
    )


def _arm_rows(results: Sequence[ObjectiveArmResult]) -> List[str]:
    header = [
        "| arm | mean rho | sd rho | seeds | accuracy | macro_f1 | mrr | recall@k |",
        "|---|---|---|---|---|---|---|---|",
    ]
    return header + [_arm_row(result) for result in results]


def _format_margin(comparison: ObjectiveComparison, challenger: ObjectiveArmResult) -> str:
    if not comparison.has_measurable_spread(challenger):
        return "n/a"
    return f"{comparison.margin_in_sigma(challenger):.2f}"


def _adoption_verdict(comparison: ObjectiveComparison, challenger: ObjectiveArmResult) -> str:
    if comparison.clears_adoption_rule(challenger):
        return "yes"
    if not comparison.has_measurable_spread(challenger):
        return "no - seed spread is unmeasurable"
    if comparison.margin_in_sigma(challenger) <= comparison.adoption_threshold_sigma:
        return "no - margin under threshold"
    if challenger.accuracy is None:
        return "no - accuracy not measured"
    return "no - accuracy guard"


def _margin_rows(comparison: ObjectiveComparison) -> List[str]:
    rows = ["| challenger | mean rho | margin over baseline (pooled sd) | clears rule |", "|---|---|---|---|"]
    for challenger in comparison.challengers():
        rows.append(
            f"| {challenger.arm} | {challenger.mean_rho:.4f} | "
            f"{_format_margin(comparison, challenger)} | {_adoption_verdict(comparison, challenger)} |"
        )
    return rows


def _provenance_line() -> str:
    import numpy
    import sklearn  # type: ignore
    import torch

    return (
        f"Library versions: numpy {numpy.__version__}, scikit-learn {sklearn.__version__}, torch {torch.__version__}."
    )


def _reproduce_command(args: CompareObjectivesArgs) -> str:
    return (
        f"tox -e compare_objectives -- --dataset {args.dataset} --arms {' '.join(args.arms)} "
        f"--controls {' '.join(args.controls)} --seeds {' '.join(str(seed) for seed in args.seeds)} "
        f"--downstream-top-k {args.downstream_top_k} "
        f"--downstream-controls {' '.join(args.downstream_controls)} "
        f"--downstream-seeds {' '.join(str(seed) for seed in args.downstream_seeds)} "
        f"--budget {args.budget} --distance {args.distance} --k-neighbors {args.k_neighbors} "
        f"--k-values {' '.join(str(depth) for depth in args.k_values)} "
        f"--adoption-threshold-sigma {args.adoption_threshold_sigma} "
        f"--max-accuracy-drop {args.max_accuracy_drop} --config {args.config} "
        f"--codebook-path {args.codebook_path} --committed-model-path {args.committed_model_path} "
        f"--selection-samples {args.selection_samples} --structure-samples {args.structure_samples} "
        f"--output-path {args.output_path} --figure-path {args.figure_path}"
    )


def _committed_advantage(comparison: ObjectiveComparison) -> Optional[Tuple[float, float]]:
    committed = _control_named(comparison, COMMITTED_CONTROL)
    adopted_accuracy = _adopted_result(comparison).accuracy
    if committed is None or committed.accuracy is None or adopted_accuracy is None:
        return None
    if committed.accuracy <= adopted_accuracy:
        return None
    return committed.accuracy, adopted_accuracy


def _artifact_held_sentence(comparison: ObjectiveComparison, advantage: Tuple[float, float]) -> str:
    adopted = _adopted_result(comparison)
    committed_accuracy, adopted_accuracy = advantage
    return (
        f"**Adopted arm: `{adopted.arm}`.** It clears the pre-registered rule against "
        f"`{comparison.baseline_arm}` by {comparison.margin_in_sigma(adopted):.2f} pooled seed sd, so it is the "
        "objective this family should be trained on. The shipped artifact is nevertheless left in place: it is "
        "produced by the supervised mapper, which owns a different loss and is out of scope here, and at the same "
        f"budget it scores {committed_accuracy:.4f} against the adopted arm's {adopted_accuracy:.4f}. Replacing it "
        "would trade a better correlation for a worse task result, which is the outcome the accuracy guard exists "
        "to refuse."
    )


def _artifact_replaced_sentence(comparison: ObjectiveComparison) -> str:
    adopted = _adopted_result(comparison)
    return (
        f"**Adopted arm: `{adopted.arm}`.** It clears the pre-registered rule against "
        f"`{comparison.baseline_arm}` by {comparison.margin_in_sigma(adopted):.2f} pooled seed sd without losing "
        "downstream accuracy, so the committed projector is retrained under that objective."
    )


def _adoption_sentence(comparison: ObjectiveComparison) -> str:
    if comparison.adopted_arm() == comparison.baseline_arm:
        return (
            f"**Adopted arm: `{comparison.baseline_arm}` (unchanged).** No challenger cleared the pre-registered "
            "rule, so the committed projector artifact is not replaced and the negative result stands."
        )
    advantage = _committed_advantage(comparison)
    if advantage is None:
        return _artifact_replaced_sentence(comparison)
    return _artifact_held_sentence(comparison, advantage)


def _report_lines(args: CompareObjectivesArgs, comparison: ObjectiveComparison) -> List[str]:
    return [
        "# Structure objective alignment",
        "",
        "The projector is scored on Spearman rho between embedding cosine similarity and Lab Euclidean distance,",
        "but it has always been trained on a different quantity. This report measures each candidate training",
        "objective on the metric it is judged by, and bounds what three dimensions can hold at all with untrained,",
        "linear and unconstrained-head controls, so the residual can be attributed rather than assumed.",
        "",
        "Every arm is trained on the training split, its checkpoint selected on a held-out selection slice, and",
        "its rho reported on a disjoint held-out slice. rho is negative by design: closer meanings, closer colors.",
        "",
        _provenance_line(),
        "",
        "## Objective arms",
        "",
        "The `seeds` column counts the seeds the correlation was averaged over; the downstream columns are averaged",
        "over the smaller downstream seed set named in the command below, and are measured only for the nominated",
        "arms, so an unnominated arm reads n/a rather than zero.",
        "",
        *_arm_rows(comparison.results),
        "",
        "![Structure preservation by training objective](figures/structure_objective.png)",
        "",
        "## Ceiling controls",
        "",
        "`noise` is an untrained projector (floor), `pca3` an untrained linear projection rescaled per axis into the",
        "Lab ranges, and `unconstrained_head` the same architecture with the Lab sigmoid/tanh head removed, reported",
        "both pre-clamp (roadmap R-2, the gamut constraint isolated) and post-clamp (what the pipeline receives).",
        "`committed` is the shipped projector artifact read straight off disk and scored on the same held-out slice,",
        "so every number above can be read against the one the repository already publishes. It is a single fixed",
        "artifact and the PCA fit is deterministic, so the zero seed spread of `committed` and `pca3` is a property",
        "of those controls rather than a measurement. Two further caveats: the pre-clamp figure is a lower bound on",
        "the unconstrained-head ceiling, because the checkpoint it reads was selected on the post-clamp score; and",
        "`pca3` is rescaled per axis as the specification pre-registers, which is not rank-preserving, so read it as",
        "a same-recipe comparator rather than as a bound on what any linear map could reach.",
        "",
        *_arm_rows(comparison.controls),
        "",
        "## Pre-registered adoption rule",
        "",
        f"An arm replaces the committed projector only if its mean held-out |rho| exceeds `{comparison.baseline_arm}`'s",
        f"by more than {comparison.adoption_threshold_sigma:g} times the pooled seed standard deviation *and* its",
        f"accuracy is no more than {comparison.max_accuracy_drop * 100:.1f} points below it. The rule was fixed",
        "before the run. A margin needs a measurable seed spread: when both arms have zero spread the margin is",
        "reported as n/a and no arm is adopted, because a single seed cannot separate a gain from noise.",
        "",
        *_margin_rows(comparison),
        "",
        _adoption_sentence(comparison),
        "",
        "## Reproduce",
        "",
        "```bash",
        _reproduce_command(args),
        "```",
        "",
    ]


def _write_report(args: CompareObjectivesArgs, comparison: ObjectiveComparison) -> None:
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, comparison)), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _print_table(comparison: ObjectiveComparison) -> None:
    print("\n=== Structure Objective Arms ===")
    for line in _arm_rows(comparison.results):
        print(line)
    print("\n=== Ceiling Controls ===")
    for line in _arm_rows(comparison.controls):
        print(line)
    print(f"\nAdopted arm: {comparison.adopted_arm()}")


def _log_startup(args: CompareObjectivesArgs, correlation_id: str) -> None:
    logger.info(
        "Starting the structure objective comparison",
        extra={
            "correlation_id": correlation_id,
            "dataset": args.dataset,
            "arms": args.arms,
            "controls": args.controls,
            "seeds": args.seeds,
            "budget": args.budget,
        },
    )


def _build_use_case(
    args: CompareObjectivesArgs,
    config: SynestheticConfig,
    cache: TrainedMapperCache,
    embedding_adapter: SentenceEmbeddingAdapter,
    codebook: ColorCodebook,
) -> CompareStructureObjectivesUseCase:
    return CompareStructureObjectivesUseCase(
        lab_colors_factory=_build_lab_colors_factory(cache),
        structure_preservation_evaluator=SpearmanStructurePreservationEvaluator(seed=config.training.seed),
        evaluate_use_case_factory=_build_evaluate_factory(cache, args, config, embedding_adapter, codebook),
        retrieval_use_case_factory=_build_retrieval_factory(cache, args, config, embedding_adapter, codebook),
        adoption_threshold_sigma=args.adoption_threshold_sigma,
        max_accuracy_drop=args.max_accuracy_drop,
        downstream_budget=args.budget,
        downstream_bits_per_token=_bits_per_token(codebook),
    )


def _run_comparison(
    use_case: CompareStructureObjectivesUseCase,
    args: CompareObjectivesArgs,
    train_embeddings: npt.NDArray,
    structure_embeddings: npt.NDArray,
    correlation_id: str,
) -> ObjectiveComparison:
    comparison = use_case.execute(
        train_embeddings=train_embeddings,
        eval_embeddings=structure_embeddings,
        arms=args.arms,
        seeds=args.seeds,
        baseline_arm=BASELINE_ARM,
        controls=args.controls,
        log_decision=False,
        correlation_id=correlation_id,
    )
    nominated = _downstream_arms(args, comparison)
    if not nominated:
        use_case.log_decision(comparison, correlation_id)
        return comparison
    return use_case.execute(
        train_embeddings=train_embeddings,
        eval_embeddings=structure_embeddings,
        arms=args.arms,
        seeds=args.seeds,
        baseline_arm=BASELINE_ARM,
        controls=args.controls,
        downstream_arms=nominated,
        downstream_seeds=args.downstream_seeds,
        correlation_id=correlation_id,
    )


def main(args: CompareObjectivesArgs) -> None:
    _reject_unrunnable_request(args)
    config = SynestheticConfig.from_yaml(args.config)
    correlation_id = str(uuid.uuid4())
    _log_startup(args, correlation_id)
    dataset_repository = _setup_dataset(args.dataset)
    embedding_adapter = SentenceEmbeddingAdapter()
    codebook = load_codebook(args.codebook_path)
    train_embeddings = _encode_split(
        dataset_repository,
        embedding_adapter,
        config,
        config.dataset.train_split,
        args.train_samples if args.train_samples is not None else config.dataset.max_samples,
    )
    selection_embeddings, structure_embeddings = _held_out_embeddings(
        dataset_repository, embedding_adapter, args, config
    )
    cache = TrainedMapperCache(args, config, selection_embeddings)
    use_case = _build_use_case(args, config, cache, embedding_adapter, codebook)
    comparison = _run_comparison(use_case, args, train_embeddings, structure_embeddings, correlation_id)
    _print_table(comparison)
    _write_report(args, comparison)
    MatplotlibFigureRenderer().render_objective_comparison(comparison, args.figure_path)
    print(f"Saved {args.figure_path}")


if __name__ == "__main__":
    main(tyro.cli(CompareObjectivesArgs))
