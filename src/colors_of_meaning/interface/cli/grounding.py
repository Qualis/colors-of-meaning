import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import tyro
import yaml  # type: ignore

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_grounding_use_case import (
    EvaluateGroundingUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.grounding_report import GroundingReport
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

_MAPPER_TYPES = {"structured": "structured", "plain": "unconstrained"}


@dataclass(frozen=True)
class CandidateAnswer:
    answer_id: str
    text: str


@dataclass(frozen=True)
class RagSample:
    question: str
    passages: List[str]
    answers: List[CandidateAnswer]


@dataclass(frozen=True)
class AuditedAnswer:
    answer: CandidateAnswer
    report: GroundingReport


@dataclass
class GroundingPipeline:
    embedder: SentenceEmbeddingAdapter
    encode: EncodeDocumentUseCase
    use_case: EvaluateGroundingUseCase


@dataclass
class GroundingArgs:
    sample: str = "samples/rag_grounding.yaml"
    config: str = "configs/base.yaml"
    model_path: str = "artifacts/models/projector.pth"
    codebook_name: str = "codebook_4096"
    mapper: Literal["structured", "plain"] = "plain"
    threshold: float = 25.0
    metric: Literal["sliced", "jensen_shannon", "exact"] = "sliced"
    output_path: str = "reports/grounding_audit.md"


def _load_sample(sample_path: str) -> RagSample:
    raw = yaml.safe_load(Path(sample_path).read_text(encoding="utf-8"))
    answers = [CandidateAnswer(answer_id=entry["id"], text=entry["text"]) for entry in raw["answers"]]
    return RagSample(question=raw["question"], passages=list(raw["passages"]), answers=answers)


def _build_mapper(args: GroundingArgs, config: SynestheticConfig) -> ColorMapper:
    mapper = create_color_mapper(_MAPPER_TYPES[args.mapper], config)
    mapper.load_weights(args.model_path)
    return mapper


def _create_distance_calculator(metric: str, codebook: ColorCodebook, config: SynestheticConfig) -> DistanceCalculator:
    if metric == "sliced":
        return SlicedWassersteinDistanceCalculator(codebook=codebook, seed=config.training.seed)
    if metric == "jensen_shannon":
        return JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    if metric == "exact":
        return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)
    raise ValueError(f"Unknown metric: {metric}")


def _build_pipeline(args: GroundingArgs, config: SynestheticConfig) -> GroundingPipeline:
    codebook = load_codebook(args.codebook_name)
    encode = EncodeDocumentUseCase(QuantizedColorMapper(_build_mapper(args, config), codebook))
    use_case = EvaluateGroundingUseCase(_create_distance_calculator(args.metric, codebook, config))
    return GroundingPipeline(embedder=SentenceEmbeddingAdapter(), encode=encode, use_case=use_case)


def _encode(pipeline: GroundingPipeline, text: str, document_id: str) -> ColoredDocument:
    return pipeline.encode.execute(pipeline.embedder.encode_document_sentences(text), document_id=document_id)


def _audit(pipeline: GroundingPipeline, sample: RagSample, threshold: float) -> List[AuditedAnswer]:
    contexts = [_encode(pipeline, text, f"passage_{index}") for index, text in enumerate(sample.passages)]
    audited = []
    for answer in sample.answers:
        histogram = _encode(pipeline, answer.text, answer.answer_id)
        report = pipeline.use_case.execute_against_contexts(histogram, contexts, threshold)
        audited.append(AuditedAnswer(answer=answer, report=report))
    return audited


def _verdict(report: GroundingReport) -> str:
    return "grounded" if report.is_grounded else "off-context"


def _verdict_rows(audited: List[AuditedAnswer]) -> List[str]:
    rows = ["| answer | divergence | threshold | verdict |", "|---|---|---|---|"]
    for entry in audited:
        rows.append(
            f"| {entry.answer.answer_id} | {entry.report.divergence:.4f} | "
            f"{entry.report.threshold:.4f} | {_verdict(entry.report)} |"
        )
    return rows


def _provenance_line() -> str:
    import numpy

    return f"Library versions: numpy {numpy.__version__}."


def _reproduce_command(args: GroundingArgs) -> str:
    return (
        f"tox -e grounding -- --sample {args.sample} --config {args.config} "
        f"--model-path {args.model_path} --codebook-name {args.codebook_name} "
        f"--mapper {args.mapper} --metric {args.metric} --threshold {args.threshold}"
    )


def _flagged_count(audited: List[AuditedAnswer]) -> int:
    return sum(1 for entry in audited if not entry.report.is_grounded)


def _report_lines(args: GroundingArgs, sample: RagSample, audited: List[AuditedAnswer]) -> List[str]:
    return [
        "# Answer grounding audit",
        "",
        "Each candidate answer is encoded to a colour distribution over the 4,096-colour codebook and",
        "measured against the uniform **mixture** of the retrieved passages' colour distributions (never a",
        "chroma-cancelling mean of Lab colours). The grounding divergence is the perceptual distance between",
        f"the answer's colour mass and the retrieved evidence ({args.metric}); answers above the threshold",
        f"({args.threshold:g}) are flagged as off-context.",
        "",
        "**This is a distributional off-context signal, not a factuality or citation checker.** A large",
        "divergence is a strong smell that the answer is topically unsupported by the retrieved passages. A",
        "small divergence does **not** prove the answer is factually correct -- only that its colour",
        "distribution lies within the context's. It inherits the 384->3 lossy bottleneck, so it is a coarse",
        "triage instrument, and the threshold is corpus- and metric-dependent, not a universal constant.",
        "",
        _provenance_line(),
        "",
        f"Question: {sample.question}",
        "",
        f"Retrieved passages: {len(sample.passages)}. Candidate answers: {len(sample.answers)}. "
        f"Flagged off-context: {_flagged_count(audited)}.",
        "",
        "## Per-answer grounding",
        "",
        *_verdict_rows(audited),
        "",
        "## Reproduce",
        "",
        "```bash",
        _reproduce_command(args),
        "```",
        "",
    ]


def _write_report(args: GroundingArgs, sample: RagSample, audited: List[AuditedAnswer]) -> None:
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, sample, audited)), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _print_table(audited: List[AuditedAnswer]) -> None:
    print("\n=== Answer grounding audit ===")
    for line in _verdict_rows(audited):
        print(line)


def _log_startup(args: GroundingArgs, config: SynestheticConfig, sample: RagSample) -> None:
    logger.info(
        "Starting grounding audit",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "sample": args.sample,
            "passages": len(sample.passages),
            "answers": len(sample.answers),
            "metric": args.metric,
            "threshold": args.threshold,
            "seed": config.training.seed,
        },
    )


def _audited_sample(args: GroundingArgs, config: SynestheticConfig, sample: RagSample) -> List[AuditedAnswer]:
    return _audit(_build_pipeline(args, config), sample, args.threshold)


def main(args: GroundingArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    sample = _load_sample(args.sample)
    if not sample.answers:
        raise ValueError(f"No candidate answers found in sample: {args.sample}")
    _log_startup(args, config, sample)
    audited = _audited_sample(args, config, sample)
    _print_table(audited)
    _write_report(args, sample, audited)


if __name__ == "__main__":
    main(tyro.cli(GroundingArgs))
