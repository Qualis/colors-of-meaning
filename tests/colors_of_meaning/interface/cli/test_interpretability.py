from contextlib import ExitStack
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

from colors_of_meaning.domain.model.interpretability_report import (
    InterpretabilityReport,
    InterpretabilityScores,
    InterpretabilityThresholds,
)
from colors_of_meaning.interface.cli.interpretability import (
    InterpretabilityArgs,
    _axis_rows,
    _build_control_mapper,
    _build_structured_mapper,
    _build_thresholds,
    _build_use_case,
    _falsified_summary,
    _provenance_line,
    _reproduce_command,
    _setup_dataset,
    _print_table,
    _verdict,
    _write_report,
    main,
)


def _report(validated: bool = True) -> InterpretabilityReport:
    control = InterpretabilityScores(0.10, 0.10, 0.10) if validated else InterpretabilityScores(0.59, 0.10, 0.10)
    return InterpretabilityReport(
        structured=InterpretabilityScores(0.60, 0.50, 0.40),
        control=control,
        thresholds=InterpretabilityThresholds(),
    )


def _printed_lines(print_mock: Mock) -> List[str]:
    return [str(call.args[0]) for call in print_mock.call_args_list if call.args]


class TestInterpretabilityHelpers:
    def test_should_return_dataset_adapter_instance(self) -> None:
        with patch("colors_of_meaning.interface.cli.interpretability.IMDBDatasetAdapter") as adapter_class:
            adapter = _setup_dataset("imdb")

        assert adapter is adapter_class.return_value

    def test_should_load_structured_weights(self) -> None:
        with patch("colors_of_meaning.interface.cli.interpretability.create_color_mapper"):
            mapper = _build_structured_mapper(InterpretabilityArgs(structured_model="s.pth"), Mock())

        mapper.load_weights.assert_called_once_with("s.pth")

    def test_should_load_control_weights_when_control_is_unconstrained(self) -> None:
        with patch("colors_of_meaning.interface.cli.interpretability.create_color_mapper"):
            mapper = _build_control_mapper(InterpretabilityArgs(control="unconstrained", control_model="c.pth"), Mock())

        mapper.load_weights.assert_called_once_with("c.pth")

    def test_should_not_load_control_weights_when_control_is_noise(self) -> None:
        with patch("colors_of_meaning.interface.cli.interpretability.create_color_mapper"):
            mapper = _build_control_mapper(InterpretabilityArgs(control="noise"), Mock())

        mapper.load_weights.assert_not_called()

    def test_should_build_thresholds_from_args(self) -> None:
        thresholds = _build_thresholds(InterpretabilityArgs(hue_topic_margin=0.11))

        assert thresholds.hue_topic_margin == 0.11

    def test_should_build_use_case_from_collaborators(self) -> None:
        config = Mock()
        config.structured_mapper.num_clusters = 8
        config.structured_mapper.concreteness_resource = "concreteness_norms.tsv"
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.interpretability.SentenceEmbeddingAdapter"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.interpretability._build_structured_mapper"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.interpretability._build_control_mapper"))
            stack.enter_context(
                patch("colors_of_meaning.interface.cli.interpretability.SklearnInterpretabilityEvaluator")
            )
            stack.enter_context(patch("colors_of_meaning.interface.cli.interpretability.BrysbaertConcretenessLexicon"))
            use_case_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.interpretability.EvaluateInterpretabilityUseCase")
            )
            use_case = _build_use_case(InterpretabilityArgs(), config)

        assert use_case is use_case_class.return_value


class TestInterpretabilityReporting:
    def test_should_report_validated_verdict_when_all_axes_pass(self) -> None:
        assert _verdict(_report(validated=True)) == "VALIDATED"

    def test_should_report_falsified_verdict_when_an_axis_fails(self) -> None:
        assert _verdict(_report(validated=False)) == "FALSIFIED"

    def test_should_summarise_no_falsified_axes_as_none(self) -> None:
        assert _falsified_summary(_report(validated=True)) == "none"

    def test_should_summarise_falsified_axis_names(self) -> None:
        assert _falsified_summary(_report(validated=False)) == "hue_topic"

    def test_should_mark_passing_axes_in_table(self) -> None:
        rows = _axis_rows(_report(validated=True))

        assert all(row.endswith("| pass |") for row in rows[2:])

    def test_should_mark_falsified_axis_in_table(self) -> None:
        rows = _axis_rows(_report(validated=False))

        assert rows[2].endswith("| falsified |")

    def test_should_record_library_versions_in_provenance(self) -> None:
        assert "numpy" in _provenance_line()

    def test_should_emit_reproduce_command_with_dataset(self) -> None:
        assert "--dataset imdb" in _reproduce_command(InterpretabilityArgs(dataset="imdb"))

    def test_should_write_report_with_verdict(self, tmp_path: Path) -> None:
        output_path = tmp_path / "interpretability.md"

        _write_report(str(output_path), InterpretabilityArgs(), _report(validated=True))

        assert "Overall verdict: **VALIDATED**." in output_path.read_text()

    def test_should_print_axis_rows(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_table(_report(validated=True))

        assert any(line.startswith("| hue <-> topic") for line in _printed_lines(print_mock))


def _run_main(tmp_path: Path, report: InterpretabilityReport) -> Path:
    output_path = tmp_path / "interpretability.md"
    args = InterpretabilityArgs(output_path=str(output_path))
    with ExitStack() as stack:
        config = Mock()
        config.training.seed = 42
        config.dataset.test_split = "test"
        stack.enter_context(
            patch("colors_of_meaning.interface.cli.interpretability.SynestheticConfig")
        ).from_yaml.return_value = config
        use_case = Mock()
        use_case.execute.return_value = report
        stack.enter_context(
            patch("colors_of_meaning.interface.cli.interpretability._build_use_case")
        ).return_value = use_case
        dataset = Mock()
        dataset.get_samples.return_value = [Mock()]
        stack.enter_context(
            patch("colors_of_meaning.interface.cli.interpretability._setup_dataset")
        ).return_value = dataset
        stack.enter_context(patch("builtins.print"))
        main(args)
    return output_path


class TestInterpretabilityMain:
    def test_should_write_report_file(self, tmp_path: Path) -> None:
        output_path = _run_main(tmp_path, _report(validated=True))

        assert output_path.exists()

    def test_should_write_falsified_verdict_into_report(self, tmp_path: Path) -> None:
        output_path = _run_main(tmp_path, _report(validated=False))

        assert "FALSIFIED" in output_path.read_text()
