import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import tyro

from colors_of_meaning.application.use_case.analyze_narrative_arc_use_case import (
    AnalyzeNarrativeArcUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.narrative_arc import NarrativeArc, NarrativeBeat
from colors_of_meaning.domain.service.color_mapper import ColorMapper
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
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook
from colors_of_meaning.shared.outline_parser import parse_outline
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

_MAPPER_TYPES = {"structured": "structured", "plain": "unconstrained"}


@dataclass
class CompassArgs:
    outline: str = "outlines/sample_story.md"
    config: str = "configs/structured.yaml"
    model_path: str = "artifacts/models/structured_projector.pth"
    codebook_name: str = "codebook_4096"
    min_beat_chars: int = 200
    drift_threshold: float = 25.0
    mapper: Literal["structured", "plain"] = "structured"
    metric: Literal["sliced", "jensen_shannon", "exact"] = "sliced"
    output_path: str = "reports/story_compass.md"
    figure_path: str = "reports/figures/story_compass.png"


def _read_outline(outline_path: str, min_beat_chars: int) -> List[NarrativeBeat]:
    text = Path(outline_path).read_text(encoding="utf-8")
    return parse_outline(text, min_beat_chars)


def _build_mapper(args: CompassArgs, config: SynestheticConfig) -> ColorMapper:
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


def _build_use_case(args: CompassArgs, config: SynestheticConfig) -> AnalyzeNarrativeArcUseCase:
    codebook = load_codebook(args.codebook_name)
    return AnalyzeNarrativeArcUseCase(
        embedding=SentenceEmbeddingAdapter(),
        structured_mapper=_build_mapper(args, config),
        codebook=codebook,
        distance_calculator=_create_distance_calculator(args.metric, codebook, config),
    )


def _title_or_placeholder(beat: NarrativeBeat) -> str:
    return beat.title if beat.title else f"Beat {beat.index + 1}"


def _drift_cell(arc: NarrativeArc, index: int) -> str:
    if index < len(arc.drift_series):
        return f"{arc.drift_series[index]:.4f}"
    return "n/a"


def _beat_rows(arc: NarrativeArc, threshold: float) -> List[str]:
    flagged_indices = {flag.index for flag in arc.flagged_beats(threshold)}
    rows = [
        "| beat | title | lightness | chroma | hue_deg | coherence | drift_to_next | flagged |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for index, point in enumerate(arc.points):
        marker = "yes" if point.beat.index in flagged_indices else "no"
        rows.append(
            f"| {point.beat.index} | {_title_or_placeholder(point.beat)} | {point.lightness:.2f} | "
            f"{point.chroma:.2f} | {point.hue_degrees:.1f} | {arc.coherence_series[index]:.4f} | "
            f"{_drift_cell(arc, index)} | {marker} |"
        )
    return rows


def _provenance_line() -> str:
    import numpy

    return f"Library versions: numpy {numpy.__version__}."


def _reproduce_command(args: CompassArgs) -> str:
    return (
        f"tox -e compass -- --outline {args.outline} --config {args.config} "
        f"--model-path {args.model_path} --codebook-name {args.codebook_name} "
        f"--mapper {args.mapper} --metric {args.metric} --min-beat-chars {args.min_beat_chars} "
        f"--drift-threshold {args.drift_threshold}"
    )


def _report_lines(args: CompassArgs, arc: NarrativeArc) -> List[str]:
    flagged = arc.flagged_beats(args.drift_threshold)
    return [
        "# Narrative color compass",
        "",
        "An ordered outline read as a measured colour trajectory. Each beat's colour is read from its",
        "whole-text (document-level) embedding through the structured mapper -- never the mean of its",
        "sentence colours, which would cancel chroma -- so lightness tracks sentiment, chroma tracks",
        "concreteness, and hue tracks topic. Drift is the perceptual distance between consecutive beat",
        "colour histograms; coherence is each beat's distance to the whole-book palette. Beats whose",
        f"coherence exceeds the threshold ({args.drift_threshold:g}, metric: {args.metric}) are flagged.",
        "",
        _provenance_line(),
        "",
        f"Beats: {arc.beat_count}. Flagged: {len(flagged)}.",
        "",
        "## Per-beat arc",
        "",
        *_beat_rows(arc, args.drift_threshold),
        "",
        "## Reproduce",
        "",
        "```bash",
        _reproduce_command(args),
        "```",
        "",
    ]


def _write_report(args: CompassArgs, arc: NarrativeArc) -> None:
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, arc)), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _render_figure(arc: NarrativeArc, figure_path: str) -> None:
    MatplotlibFigureRenderer().render_narrative_arc(arc, figure_path)
    print(f"Saved {figure_path}")


def _print_table(args: CompassArgs, arc: NarrativeArc) -> None:
    print("\n=== Narrative Color Compass ===")
    for line in _beat_rows(arc, args.drift_threshold):
        print(line)


def _log_startup(args: CompassArgs, config: SynestheticConfig, correlation_id: str, beat_count: int) -> None:
    logger.info(
        "Starting narrative compass",
        extra={
            "correlation_id": correlation_id,
            "outline": args.outline,
            "beats": beat_count,
            "mapper": args.mapper,
            "metric": args.metric,
            "drift_threshold": args.drift_threshold,
            "seed": config.training.seed,
        },
    )


def main(args: CompassArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    correlation_id = str(uuid.uuid4())
    beats = _read_outline(args.outline, args.min_beat_chars)
    if not beats:
        raise ValueError(f"No beats parsed from outline: {args.outline}")
    _log_startup(args, config, correlation_id, len(beats))
    arc = _build_use_case(args, config).execute(beats)
    _print_table(args, arc)
    _write_report(args, arc)
    _render_figure(arc, args.figure_path)


if __name__ == "__main__":
    main(tyro.cli(CompassArgs))
