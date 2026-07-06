from contextlib import ExitStack
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
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
from colors_of_meaning.interface.cli.compass import (
    CompassArgs,
    _beat_rows,
    _build_mapper,
    _build_use_case,
    _create_distance_calculator,
    _drift_cell,
    _load_codebook,
    _provenance_line,
    _print_table,
    _read_outline,
    _reproduce_command,
    _title_or_placeholder,
    _write_report,
    main,
)

MODULE = "colors_of_meaning.interface.cli.compass"
_LONG_BLOCK = "The travellers cross the river at first light and the mountains close in behind them. " * 3


def _config() -> Mock:
    config = Mock()
    config.training.seed = 42
    config.distance.smoothing_epsilon = 1e-8
    config.distance.sinkhorn_reg = None
    return config


def _arc(coherence: List[float]) -> NarrativeArc:
    points = [
        ArcPoint(
            beat=NarrativeBeat(index=index, title="" if index == 0 else f"Chapter {index}", text="prose"),
            colour=LabColor(l=50.0, a=10.0, b=5.0),
            histogram=ColoredDocument(histogram=np.full(4, 0.25, dtype=np.float64)),
        )
        for index in range(len(coherence))
    ]
    return NarrativeArc(points=points, drift_series=[0.5] * (len(points) - 1), coherence_series=coherence)


def _printed_lines(print_mock: Mock) -> List[str]:
    return [str(call.args[0]) for call in print_mock.call_args_list if call.args]


class TestReadOutline:
    def test_should_parse_beats_from_an_outline_file(self, tmp_path: Path) -> None:
        outline = tmp_path / "story.md"
        outline.write_text(f"## One\n\n{_LONG_BLOCK}\n\n## Two\n\n{_LONG_BLOCK}\n")

        beats = _read_outline(str(outline), min_beat_chars=50)

        assert len(beats) == 2


class TestLoadCodebook:
    def test_should_return_the_loaded_codebook(self) -> None:
        with patch(f"{MODULE}.FileColorCodebookRepository") as repo:
            repo.return_value.load.return_value = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

            codebook = _load_codebook("codebook_4096")

        assert codebook.num_bins == 8

    def test_should_raise_when_codebook_is_missing(self) -> None:
        with patch(f"{MODULE}.FileColorCodebookRepository") as repo:
            repo.return_value.load.return_value = None

            with pytest.raises(FileNotFoundError, match="Codebook not found"):
                _load_codebook("absent")


class TestBuildMapper:
    def test_should_load_weights_from_the_model_path(self) -> None:
        with patch(f"{MODULE}.create_color_mapper"):
            mapper = _build_mapper(CompassArgs(model_path="m.pth"), _config())

        mapper.load_weights.assert_called_once_with("m.pth")

    def test_should_select_the_unconstrained_mapper_for_plain(self) -> None:
        with patch(f"{MODULE}.create_color_mapper") as factory:
            _build_mapper(CompassArgs(mapper="plain"), _config())

        assert factory.call_args.args[0] == "unconstrained"

    def test_should_select_the_structured_mapper_for_structured(self) -> None:
        with patch(f"{MODULE}.create_color_mapper") as factory:
            _build_mapper(CompassArgs(mapper="structured"), _config())

        assert factory.call_args.args[0] == "structured"


class TestCreateDistanceCalculator:
    def test_should_build_sliced_wasserstein_by_default_metric(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("sliced", codebook, _config())

        assert isinstance(calculator, SlicedWassersteinDistanceCalculator)

    def test_should_build_jensen_shannon_calculator(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("jensen_shannon", codebook, _config())

        assert isinstance(calculator, JensenShannonDistanceCalculator)

    def test_should_build_exact_wasserstein_calculator(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("exact", codebook, _config())

        assert isinstance(calculator, WassersteinDistanceCalculator)

    def test_should_raise_for_unknown_metric(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        with pytest.raises(ValueError, match="Unknown metric"):
            _create_distance_calculator("cosine", codebook, _config())


class TestBuildUseCase:
    def test_should_assemble_the_use_case_from_collaborators(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.SentenceEmbeddingAdapter"))
            stack.enter_context(patch(f"{MODULE}._build_mapper"))
            stack.enter_context(patch(f"{MODULE}._load_codebook"))
            stack.enter_context(patch(f"{MODULE}._create_distance_calculator"))
            use_case_class = stack.enter_context(patch(f"{MODULE}.AnalyzeNarrativeArcUseCase"))
            use_case = _build_use_case(CompassArgs(), _config())

        assert use_case is use_case_class.return_value


class TestReporting:
    def test_should_fall_back_to_a_placeholder_title_when_empty(self) -> None:
        assert _title_or_placeholder(NarrativeBeat(index=0, title="", text="x")) == "Beat 1"

    def test_should_use_the_beat_title_when_present(self) -> None:
        assert _title_or_placeholder(NarrativeBeat(index=1, title="Climax", text="x")) == "Climax"

    def test_should_render_a_drift_value_for_non_final_beats(self) -> None:
        assert _drift_cell(_arc([1.0, 2.0]), 0) == "0.5000"

    def test_should_render_not_available_for_the_final_beat_drift(self) -> None:
        assert _drift_cell(_arc([1.0, 2.0]), 1) == "n/a"

    def test_should_mark_a_beat_above_threshold_as_flagged(self) -> None:
        rows = _beat_rows(_arc([10.0, 40.0, 5.0]), threshold=25.0)

        assert rows[3].endswith("| yes |")

    def test_should_mark_a_beat_below_threshold_as_not_flagged(self) -> None:
        rows = _beat_rows(_arc([10.0, 40.0, 5.0]), threshold=25.0)

        assert rows[2].endswith("| no |")

    def test_should_record_numpy_in_provenance(self) -> None:
        assert "numpy" in _provenance_line()

    def test_should_emit_the_outline_in_the_reproduce_command(self) -> None:
        assert "--outline story.md" in _reproduce_command(CompassArgs(outline="story.md"))

    def test_should_write_the_report_with_the_compass_heading(self, tmp_path: Path) -> None:
        output_path = tmp_path / "story_compass.md"

        _write_report(CompassArgs(output_path=str(output_path)), _arc([1.0, 2.0]))

        assert "Narrative color compass" in output_path.read_text()

    def test_should_print_beat_rows(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_table(CompassArgs(), _arc([1.0, 2.0]))

        assert any(line.startswith("| beat | title") for line in _printed_lines(print_mock))


def _run_main(tmp_path: Path, arc: NarrativeArc, outline_text: str) -> Path:
    outline = tmp_path / "story.md"
    outline.write_text(outline_text)
    output_path = tmp_path / "story_compass.md"
    figure_path = tmp_path / "story_compass.png"
    args = CompassArgs(outline=str(outline), output_path=str(output_path), figure_path=str(figure_path))
    with ExitStack() as stack:
        config = Mock()
        config.training.seed = 42
        stack.enter_context(patch(f"{MODULE}.SynestheticConfig")).from_yaml.return_value = config
        use_case = Mock()
        use_case.execute.return_value = arc
        stack.enter_context(patch(f"{MODULE}._build_use_case")).return_value = use_case
        stack.enter_context(patch(f"{MODULE}._render_figure"))
        stack.enter_context(patch("builtins.print"))
        main(args)
    return output_path


class TestCompassMain:
    def test_should_write_the_report_file(self, tmp_path: Path) -> None:
        output_path = _run_main(tmp_path, _arc([1.0, 2.0]), f"## One\n\n{_LONG_BLOCK}\n\n## Two\n\n{_LONG_BLOCK}\n")

        assert output_path.exists()

    def test_should_render_the_figure_to_the_figure_path(self, tmp_path: Path) -> None:
        outline = tmp_path / "story.md"
        outline.write_text(f"## One\n\n{_LONG_BLOCK}\n\n## Two\n\n{_LONG_BLOCK}\n")
        arc = _arc([1.0, 2.0])
        args = CompassArgs(
            outline=str(outline),
            output_path=str(tmp_path / "r.md"),
            figure_path=str(tmp_path / "r.png"),
        )
        with ExitStack() as stack:
            config = Mock()
            config.training.seed = 42
            stack.enter_context(patch(f"{MODULE}.SynestheticConfig")).from_yaml.return_value = config
            use_case = Mock()
            use_case.execute.return_value = arc
            stack.enter_context(patch(f"{MODULE}._build_use_case")).return_value = use_case
            renderer = stack.enter_context(patch(f"{MODULE}.MatplotlibFigureRenderer"))
            stack.enter_context(patch("builtins.print"))
            main(args)

        renderer.return_value.render_narrative_arc.assert_called_once_with(arc, str(tmp_path / "r.png"))

    def test_should_raise_when_the_outline_has_no_qualifying_beats(self, tmp_path: Path) -> None:
        outline = tmp_path / "empty.md"
        outline.write_text("## Tiny\n\ntoo short\n")
        args = CompassArgs(outline=str(outline), min_beat_chars=200)
        with patch(f"{MODULE}.SynestheticConfig"), patch("builtins.print"):
            with pytest.raises(ValueError, match="No beats parsed"):
                main(args)
