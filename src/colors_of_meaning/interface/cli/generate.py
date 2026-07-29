import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Set

import tyro

from colors_of_meaning.application.use_case.analyze_narrative_arc_use_case import (
    AnalyzeNarrativeArcUseCase,
)
from colors_of_meaning.application.use_case.generate_book_use_case import GenerateBookUseCase
from colors_of_meaning.domain.model.book_specification import BookSpecification
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.generated_book import GeneratedBook, GeneratedChapter
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.generation.anthropic_text_generator_adapter import (
    AnthropicTextGeneratorAdapter,
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
from colors_of_meaning.shared.synesthetic_config import GenerationConfig, SynestheticConfig

logger = logging.getLogger(__name__)

_MAPPER_TYPES = {"structured": "structured", "plain": "unconstrained"}


@dataclass
class GenerateArgs:
    summary: str = ""
    summary_file: str = ""
    config: str = "configs/documents.yaml"
    model_path: str = "artifacts/models/structured_projector.pth"
    codebook_name: str = "codebook_4096"
    num_beats: int = 8
    words_per_beat: int = 400
    tone: str = ""
    retry_budget: int = 2
    drift_threshold: float = 25.0
    model: str = "claude-opus-4-8"
    effort: str = "high"
    mapper: Literal["structured", "plain"] = "structured"
    metric: Literal["sliced", "jensen_shannon", "exact"] = "sliced"
    output_dir: str = "reports/book"
    figure_path: str = "reports/figures/story_book.png"


def _read_summary(args: GenerateArgs) -> str:
    if args.summary_file:
        return Path(args.summary_file).read_text(encoding="utf-8")
    if args.summary:
        return args.summary
    raise ValueError("provide --summary or --summary-file")


def _build_specification(args: GenerateArgs) -> BookSpecification:
    return BookSpecification(
        summary=_read_summary(args),
        num_beats=args.num_beats,
        words_per_beat=args.words_per_beat,
        tone=args.tone or None,
    )


def _generation_config(args: GenerateArgs, config: SynestheticConfig) -> GenerationConfig:
    base = config.generation if config.generation is not None else GenerationConfig()
    return GenerationConfig(
        model=args.model,
        effort=args.effort,
        outline_max_tokens=base.outline_max_tokens,
        prose_max_tokens=base.prose_max_tokens,
        num_beats=args.num_beats,
        words_per_beat=args.words_per_beat,
        retry_budget=args.retry_budget,
        drift_threshold=args.drift_threshold,
    )


def _build_mapper(args: GenerateArgs, config: SynestheticConfig) -> ColorMapper:
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


def _build_narrative_arc_analyzer(args: GenerateArgs, config: SynestheticConfig) -> AnalyzeNarrativeArcUseCase:
    codebook = load_codebook(args.codebook_name)
    return AnalyzeNarrativeArcUseCase(
        embedding=SentenceEmbeddingAdapter(),
        structured_mapper=_build_mapper(args, config),
        codebook=codebook,
        distance_calculator=_create_distance_calculator(args.metric, codebook, config),
    )


def _build_use_case(args: GenerateArgs, config: SynestheticConfig) -> GenerateBookUseCase:
    return GenerateBookUseCase(
        text_generator=AnthropicTextGeneratorAdapter(_generation_config(args, config)),
        narrative_arc_analyzer=_build_narrative_arc_analyzer(args, config),
        drift_threshold=args.drift_threshold,
        retry_budget=args.retry_budget,
    )


def _slugify(title: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in title.lower())
    return slug.strip("_") or "chapter"


def _strip_leading_heading(prose: str) -> str:
    body = prose.lstrip()
    if not body.startswith("# "):
        return body
    _, _, remainder = body.partition("\n")
    return remainder.lstrip()


def _write_chapter(output_dir: Path, chapter: GeneratedChapter) -> None:
    filename = f"{chapter.brief.index + 1:02d}_{_slugify(chapter.brief.title)}.md"
    destination = output_dir / filename
    body = _strip_leading_heading(chapter.prose)
    destination.write_text(f"# {chapter.brief.title}\n\n{body}\n", encoding="utf-8")


def _arc_rows(book: GeneratedBook, flagged: Set[int]) -> List[str]:
    rows = []
    for chapter, point, coherence in zip(book.chapters, book.arc.points, book.arc.coherence_series, strict=True):
        marker = "yes" if chapter.brief.index in flagged else "no"
        rows.append(
            f"| {chapter.brief.index} | {chapter.brief.title} | {point.lightness:.2f} | "
            f"{point.chroma:.2f} | {point.hue_degrees:.1f} | {coherence:.4f} | {marker} |"
        )
    return rows


def _arc_report_lines(args: GenerateArgs, book: GeneratedBook) -> List[str]:
    flagged = set(book.metadata.flagged_beat_indices)
    tokens = book.metadata.total_tokens
    return [
        "# Generated book — colour compass audit",
        "",
        "An LLM wrote each chapter; the narrative colour compass measured every beat against the",
        "emerging book palette and flagged the ones whose coherence drifted beyond the threshold.",
        "",
        f"Model: {args.model} (effort {args.effort}). Chapters: {book.chapter_count}. "
        f"Retries: {book.metadata.total_retries}. Flagged: {len(flagged)}.",
        f"Tokens: {tokens.input_tokens} in / {tokens.output_tokens} out.",
        "",
        "| beat | title | lightness | chroma | hue_deg | coherence | flagged |",
        "|---|---|---|---|---|---|---|",
        *_arc_rows(book, flagged),
        "",
    ]


def _write_book(args: GenerateArgs, book: GeneratedBook) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for chapter in book.chapters:
        _write_chapter(output_dir, chapter)
    (output_dir / "arc.md").write_text("\n".join(_arc_report_lines(args, book)), encoding="utf-8")
    print(f"Saved {book.chapter_count} chapters and arc.md to {args.output_dir}")


def _render_figure(book: GeneratedBook, figure_path: str) -> None:
    MatplotlibFigureRenderer().render_narrative_arc(book.arc, figure_path)
    print(f"Saved {figure_path}")


def _print_summary(book: GeneratedBook) -> None:
    print(f"\n=== Generated book: {book.chapter_count} chapters ===")
    print(f"Retries: {book.metadata.total_retries}. Flagged beats: {book.metadata.flagged_beat_indices}.")


def _log_startup(args: GenerateArgs, correlation_id: str) -> None:
    logger.info(
        "Starting book generation",
        extra={
            "correlation_id": correlation_id,
            "model": args.model,
            "effort": args.effort,
            "num_beats": args.num_beats,
            "words_per_beat": args.words_per_beat,
            "retry_budget": args.retry_budget,
            "drift_threshold": args.drift_threshold,
        },
    )


def main(args: GenerateArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    correlation_id = str(uuid.uuid4())
    specification = _build_specification(args)
    _log_startup(args, correlation_id)
    book = _build_use_case(args, config).execute(specification)
    _write_book(args, book)
    _render_figure(book, args.figure_path)
    _print_summary(book)


if __name__ == "__main__":
    main(tyro.cli(GenerateArgs))
