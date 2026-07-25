from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.domain.model.book_specification import BeatBrief, BookSpecification
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.generated_book import (
    GeneratedBook,
    GeneratedChapter,
    GenerationMetadata,
    TokenUsage,
)
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.narrative_arc import ArcPoint, NarrativeArc, NarrativeBeat
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.interface.cli.generate import (
    GenerateArgs,
    _arc_report_lines,
    _arc_rows,
    _build_mapper,
    _build_narrative_arc_analyzer,
    _build_use_case,
    _create_distance_calculator,
    _generation_config,
    _read_summary,
    _slugify,
    _strip_leading_heading,
    _write_book,
    _write_chapter,
    main,
)
from colors_of_meaning.shared.synesthetic_config import GenerationConfig

MODULE = "colors_of_meaning.interface.cli.generate"


def _config() -> Mock:
    config = Mock()
    config.training.seed = 42
    config.distance.smoothing_epsilon = 1e-8
    config.distance.sinkhorn_reg = None
    return config


def _arc(beat_count: int) -> NarrativeArc:
    points = [
        ArcPoint(
            beat=NarrativeBeat(index=index, title=f"Beat {index}", text="prose"),
            colour=LabColor(l=50.0, a=10.0, b=5.0),
            histogram=ColoredDocument(histogram=np.full(4, 0.25, dtype=np.float64)),
        )
        for index in range(beat_count)
    ]
    return NarrativeArc(points=points, drift_series=[0.5] * (beat_count - 1), coherence_series=[1.0] * beat_count)


def _book(flagged: Optional[List[int]] = None) -> GeneratedBook:
    chapters = [
        GeneratedChapter(brief=BeatBrief(index=index, title=f"Beat {index}", synopsis="s"), prose=f"Prose {index}.")
        for index in range(2)
    ]
    metadata = GenerationMetadata(total_tokens=TokenUsage(10, 20), total_retries=1, flagged_beat_indices=flagged or [])
    specification = BookSpecification(summary="A town.", num_beats=2, words_per_beat=200)
    return GeneratedBook(specification=specification, chapters=chapters, arc=_arc(2), metadata=metadata)


class TestReadSummary:
    def test_should_read_the_summary_from_a_file(self, tmp_path: Path) -> None:
        summary_file = tmp_path / "summary.txt"
        summary_file.write_text("A lighthouse keeper.")

        assert _read_summary(GenerateArgs(summary_file=str(summary_file))) == "A lighthouse keeper."

    def test_should_use_the_inline_summary(self) -> None:
        assert _read_summary(GenerateArgs(summary="A quiet town.")) == "A quiet town."

    def test_should_raise_when_no_summary_is_provided(self) -> None:
        with pytest.raises(ValueError):
            _read_summary(GenerateArgs())


class TestGenerationConfig:
    def test_should_override_the_model_from_args(self) -> None:
        config = Mock()
        config.generation = GenerationConfig()

        assert _generation_config(GenerateArgs(model="claude-custom"), config).model == "claude-custom"

    def test_should_keep_the_configured_token_budgets(self) -> None:
        config = Mock()
        config.generation = GenerationConfig(outline_max_tokens=999)

        assert _generation_config(GenerateArgs(), config).outline_max_tokens == 999

    def test_should_fall_back_to_defaults_when_generation_is_absent(self) -> None:
        config = Mock()
        config.generation = None

        assert _generation_config(GenerateArgs(), config).outline_max_tokens == 4096


class TestBuildMapper:
    def test_should_load_weights_from_the_model_path(self) -> None:
        with patch(f"{MODULE}.create_color_mapper"):
            mapper = _build_mapper(GenerateArgs(model_path="m.pth"), _config())

        mapper.load_weights.assert_called_once_with("m.pth")

    def test_should_select_the_unconstrained_mapper_for_plain(self) -> None:
        with patch(f"{MODULE}.create_color_mapper") as factory:
            _build_mapper(GenerateArgs(mapper="plain"), _config())

        assert factory.call_args.args[0] == "unconstrained"


class TestCreateDistanceCalculator:
    def test_should_build_sliced_wasserstein(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("sliced", codebook, _config())

        assert isinstance(calculator, SlicedWassersteinDistanceCalculator)

    def test_should_build_jensen_shannon(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("jensen_shannon", codebook, _config())

        assert isinstance(calculator, JensenShannonDistanceCalculator)

    def test_should_build_exact_wasserstein(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("exact", codebook, _config())

        assert isinstance(calculator, WassersteinDistanceCalculator)

    def test_should_raise_for_unknown_metric(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        with pytest.raises(ValueError, match="Unknown metric"):
            _create_distance_calculator("cosine", codebook, _config())


class TestBuildUseCase:
    def test_should_assemble_the_narrative_arc_analyzer(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.SentenceEmbeddingAdapter"))
            stack.enter_context(patch(f"{MODULE}._build_mapper"))
            stack.enter_context(patch(f"{MODULE}.load_codebook"))
            stack.enter_context(patch(f"{MODULE}._create_distance_calculator"))
            analyzer_class = stack.enter_context(patch(f"{MODULE}.AnalyzeNarrativeArcUseCase"))
            analyzer = _build_narrative_arc_analyzer(GenerateArgs(), _config())

        assert analyzer is analyzer_class.return_value

    def test_should_assemble_the_generate_book_use_case(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.AnthropicTextGeneratorAdapter"))
            stack.enter_context(patch(f"{MODULE}._build_narrative_arc_analyzer"))
            use_case_class = stack.enter_context(patch(f"{MODULE}.GenerateBookUseCase"))
            config = Mock()
            config.generation = GenerationConfig()
            use_case = _build_use_case(GenerateArgs(), config)

        assert use_case is use_case_class.return_value


class TestReporting:
    def test_should_slugify_a_title(self) -> None:
        assert _slugify("The Storm!") == "the_storm"

    def test_should_fall_back_to_chapter_for_an_empty_slug(self) -> None:
        assert _slugify("!!!") == "chapter"

    def test_should_drop_a_leading_title_heading_from_prose(self) -> None:
        assert _strip_leading_heading("# The Storm\n\nRain fell hard.") == "Rain fell hard."

    def test_should_keep_prose_without_a_leading_heading(self) -> None:
        assert _strip_leading_heading("Rain fell hard.") == "Rain fell hard."

    def test_should_write_a_single_title_heading_when_prose_repeats_it(self, tmp_path: Path) -> None:
        chapter = GeneratedChapter(
            brief=BeatBrief(index=0, title="The Storm", synopsis="s"),
            prose="# The Storm\n\nRain fell hard.",
        )

        _write_chapter(tmp_path, chapter)

        assert (tmp_path / "01_the_storm.md").read_text().count("# ") == 1

    def test_should_mark_a_flagged_beat_in_the_arc_rows(self) -> None:
        rows = _arc_rows(_book(flagged=[1]), {1})

        assert rows[1].endswith("| yes |")

    def test_should_mark_an_unflagged_beat_in_the_arc_rows(self) -> None:
        rows = _arc_rows(_book(flagged=[1]), {1})

        assert rows[0].endswith("| no |")

    def test_should_title_the_arc_report(self) -> None:
        lines = _arc_report_lines(GenerateArgs(), _book())

        assert lines[0] == "# Generated book — colour compass audit"

    def test_should_write_the_arc_report_file(self, tmp_path: Path) -> None:
        with patch("builtins.print"):
            _write_book(GenerateArgs(output_dir=str(tmp_path)), _book())

        assert (tmp_path / "arc.md").exists()

    def test_should_write_a_chapter_file(self, tmp_path: Path) -> None:
        with patch("builtins.print"):
            _write_book(GenerateArgs(output_dir=str(tmp_path)), _book())

        assert (tmp_path / "01_beat_0.md").exists()


def _run_main(tmp_path: Path) -> Path:
    output_dir = tmp_path / "book"
    args = GenerateArgs(summary="A quiet town.", output_dir=str(output_dir), figure_path=str(tmp_path / "fig.png"))
    with ExitStack() as stack:
        stack.enter_context(patch(f"{MODULE}.SynestheticConfig")).from_yaml.return_value = Mock()
        use_case = Mock()
        use_case.execute.return_value = _book()
        stack.enter_context(patch(f"{MODULE}._build_use_case")).return_value = use_case
        stack.enter_context(patch(f"{MODULE}._render_figure"))
        stack.enter_context(patch("builtins.print"))
        main(args)
    return output_dir


class TestGenerateMain:
    def test_should_write_the_arc_report(self, tmp_path: Path) -> None:
        output_dir = _run_main(tmp_path)

        assert (output_dir / "arc.md").exists()

    def test_should_render_the_arc_figure(self, tmp_path: Path) -> None:
        args = GenerateArgs(
            summary="A quiet town.",
            output_dir=str(tmp_path / "book"),
            figure_path=str(tmp_path / "fig.png"),
        )
        book = _book()
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.SynestheticConfig")).from_yaml.return_value = Mock()
            use_case = Mock()
            use_case.execute.return_value = book
            stack.enter_context(patch(f"{MODULE}._build_use_case")).return_value = use_case
            renderer = stack.enter_context(patch(f"{MODULE}.MatplotlibFigureRenderer"))
            stack.enter_context(patch("builtins.print"))
            main(args)

        renderer.return_value.render_narrative_arc.assert_called_once_with(book.arc, str(tmp_path / "fig.png"))
