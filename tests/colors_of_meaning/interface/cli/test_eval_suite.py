from contextlib import ExitStack
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.application.use_case.evaluation_suite_use_case import (
    EvaluatedCell,
    EvaluationCell,
    UnfaithfulProxyError,
)
from colors_of_meaning.domain.model.distance_fidelity import DistanceFidelity
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.interface.cli.eval_suite import (
    EvalSuiteArgs,
    main,
    _build_cells,
    _build_color_classifier,
    _build_evaluate_use_case_factory,
    _encode_documents,
    _fidelity_rows,
    _load_codebook,
    _print_table,
    _provenance_line,
    _reproduce_command,
    _result_row,
    _result_rows,
    _run_fidelity_gate,
    _setup_dataset,
    _write_report,
)


def _fidelity(is_faithful: bool = True) -> DistanceFidelity:
    spearman = 0.99 if is_faithful else 0.10
    return DistanceFidelity(
        spearman=spearman, accuracy_delta=0.3, pair_count=1500, threshold_spearman=0.95, max_accuracy_delta=1.0
    )


def _evaluated_cell(dataset: str = "ag_news", budget=4000, bits_per_token=12.0) -> EvaluatedCell:
    cell = EvaluationCell(
        dataset=dataset,
        method="color",
        distance="sliced",
        budget=budget,
        requires_fidelity=True,
        bits_per_token=bits_per_token,
    )
    result = EvaluationResult(accuracy=0.83, macro_f1=0.82, recall_at_k={}, mrr=0.7)
    return EvaluatedCell(cell=cell, result=result, seconds=12.3)


def _printed_lines(print_mock: Mock) -> List[str]:
    return [str(call.args[0]) for call in print_mock.call_args_list if call.args]


class TestEvalSuiteHelpers:
    def test_should_mark_cells_requiring_fidelity_when_distance_is_sliced(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], distance="sliced"))

        assert all(cell.requires_fidelity for cell in cells)

    def test_should_not_require_fidelity_when_distance_is_exact(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], distance="wasserstein"))

        assert cells[0].requires_fidelity is False

    def test_should_build_one_cell_per_dataset(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb", "newsgroups"]))

        assert [cell.dataset for cell in cells] == ["ag_news", "imdb", "newsgroups"]

    def test_should_apply_uniform_budget_when_per_dataset_budgets_absent(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], budget=1500))

        assert [cell.budget for cell in cells] == [1500, 1500]

    def test_should_apply_per_dataset_budgets_when_provided(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb", "newsgroups"], budgets=[4000, 600, 600]))

        assert [cell.budget for cell in cells] == [4000, 600, 600]

    def test_should_raise_when_per_dataset_budgets_length_mismatches_datasets(self) -> None:
        with pytest.raises(ValueError, match="one budget per dataset"):
            _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], budgets=[4000]))

    def test_should_return_dataset_adapter_instance_when_setting_up_dataset(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.IMDBDatasetAdapter") as adapter_class:
            adapter = _setup_dataset("imdb")

        assert adapter is adapter_class.return_value

    def test_should_raise_file_not_found_when_codebook_is_absent(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.FileColorCodebookRepository") as repo_class:
            repo_class.return_value.load.return_value = None

            with pytest.raises(FileNotFoundError, match="Codebook not found"):
                _load_codebook("missing")

    def test_should_return_codebook_when_present(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.FileColorCodebookRepository") as repo_class:
            codebook = Mock()
            repo_class.return_value.load.return_value = codebook

            assert _load_codebook("codebook_4096") is codebook

    def test_should_encode_one_document_per_sample(self) -> None:
        embedding_adapter = Mock()
        embedding_adapter.encode_document_sentences.return_value = np.zeros((1, 3), dtype=np.float32)
        encode_use_case = Mock()
        encode_use_case.execute.return_value = Mock()
        samples = [Mock(text="a."), Mock(text="b."), Mock(text="c.")]

        documents = _encode_documents(embedding_adapter, encode_use_case, samples)

        assert len(documents) == 3

    def test_should_build_color_histogram_classifier(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.QuantizedColorMapper"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.EncodeDocumentUseCase"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._create_distance_calculator"))
            classifier_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.ColorHistogramClassifier")
            )

            classifier = _build_color_classifier(Mock(), Mock(), Mock(), Mock(), "sliced", 5)

        assert classifier is classifier_class.return_value

    def test_should_build_evaluate_use_case_for_a_cell(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._setup_dataset"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._build_color_classifier"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SklearnMetricsCalculator"))
            use_case_class = stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.EvaluateUseCase"))
            factory = _build_evaluate_use_case_factory(EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

            use_case = factory(_evaluated_cell().cell)

        assert use_case is use_case_class.return_value


class TestEvalSuiteFidelityGate:
    def test_should_return_fidelity_from_gate_use_case(self) -> None:
        with ExitStack() as stack:
            dataset = Mock()
            dataset.get_samples.return_value = [Mock(text="a."), Mock(text="b.")]
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._setup_dataset")).return_value = (
                dataset
            )
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.QuantizedColorMapper"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.EncodeDocumentUseCase"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._encode_documents")).return_value = [
                Mock(),
                Mock(),
            ]
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SlicedWassersteinDistanceCalculator"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.WassersteinDistanceCalculator"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SpearmanRankCorrelationCalculator"))
            gate_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.EvaluateDistanceFidelityUseCase")
            )
            gate_class.return_value.execute.return_value = _fidelity()
            config = Mock()
            config.training.seed = 42

            fidelity = _run_fidelity_gate(EvalSuiteArgs(), config, Mock(), Mock(), Mock())

        assert fidelity.is_faithful is True


class TestEvalSuiteReporting:
    def test_should_mark_fidelity_yes_when_faithful(self) -> None:
        rows = _fidelity_rows(_fidelity(is_faithful=True))

        assert rows[-1].endswith("| yes |")

    def test_should_mark_fidelity_no_when_unfaithful(self) -> None:
        rows = _fidelity_rows(_fidelity(is_faithful=False))

        assert rows[-1].endswith("| no |")

    def test_should_format_budget_as_full_when_cell_budget_is_none(self) -> None:
        row = _result_row(_evaluated_cell(budget=None))

        assert "| full |" in row

    def test_should_format_bits_as_not_available_when_absent(self) -> None:
        row = _result_row(_evaluated_cell(bits_per_token=None))

        assert "| n/a |" in row

    def test_should_record_library_versions_in_provenance(self) -> None:
        assert "numpy" in _provenance_line()

    def test_should_emit_per_dataset_budgets_in_reproduce_command(self) -> None:
        command = _reproduce_command(EvalSuiteArgs(datasets=["ag_news", "imdb"], budgets=[4000, 600]))

        assert "--budgets 4000 600" in command

    def test_should_emit_uniform_budget_in_reproduce_command_when_no_per_dataset_budgets(self) -> None:
        command = _reproduce_command(EvalSuiteArgs(datasets=["ag_news"], budget=1500))

        assert "--budget 1500" in command

    def test_should_write_a_result_row_per_cell(self, tmp_path: Path) -> None:
        output_path = tmp_path / "eval_results.md"

        _write_report(
            str(output_path), _fidelity(), [_evaluated_cell("ag_news"), _evaluated_cell("imdb")], EvalSuiteArgs()
        )

        rendered = output_path.read_text()
        rows_present = [name for name in ("ag_news", "imdb") if f"| {name} | color | sliced |" in rendered]
        assert rows_present == ["ag_news", "imdb"]

    def test_should_print_result_rows(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_table(_fidelity(), [_evaluated_cell("ag_news")])

        assert any(line.startswith("| ag_news | color | sliced |") for line in _printed_lines(print_mock))

    def test_should_not_surface_a_constant_mrr_column_in_results(self) -> None:
        header = _result_rows([_evaluated_cell("ag_news")])[0]

        assert "mrr" not in header


def _run_main(tmp_path: Path, fidelity: DistanceFidelity, evaluated: List[EvaluatedCell]) -> Path:
    output_path = tmp_path / "eval_results.md"
    args = EvalSuiteArgs(datasets=["ag_news"], output_path=str(output_path))
    with ExitStack() as stack:
        config = Mock()
        config.training.seed = 42
        stack.enter_context(
            patch("colors_of_meaning.interface.cli.eval_suite.SynestheticConfig")
        ).from_yaml.return_value = config
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SentenceEmbeddingAdapter"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.create_color_mapper"))
        repo_class = stack.enter_context(
            patch("colors_of_meaning.interface.cli.eval_suite.FileColorCodebookRepository")
        )
        repo_class.return_value.load.return_value = Mock()
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._run_fidelity_gate")).return_value = (
            fidelity
        )
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._build_evaluate_use_case_factory"))
        suite_class = stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.EvaluationSuiteUseCase"))
        suite_class.return_value.execute.return_value = evaluated
        stack.enter_context(patch("builtins.print"))
        main(args)
    return output_path


class TestEvalSuiteMain:
    def test_should_write_report_when_proxy_is_faithful(self, tmp_path: Path) -> None:
        output_path = _run_main(tmp_path, _fidelity(is_faithful=True), [_evaluated_cell("ag_news")])

        assert output_path.exists()

    def test_should_write_evaluated_cell_into_report(self, tmp_path: Path) -> None:
        output_path = _run_main(tmp_path, _fidelity(is_faithful=True), [_evaluated_cell("ag_news")])

        assert "| ag_news | color | sliced |" in output_path.read_text()

    def test_should_propagate_unfaithful_proxy_error_without_writing_report(self, tmp_path: Path) -> None:
        output_path = tmp_path / "eval_results.md"
        args = EvalSuiteArgs(datasets=["ag_news"], output_path=str(output_path))
        with ExitStack() as stack:
            config = Mock()
            config.training.seed = 42
            stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.SynestheticConfig")
            ).from_yaml.return_value = config
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SentenceEmbeddingAdapter"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.create_color_mapper"))
            repo_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.FileColorCodebookRepository")
            )
            repo_class.return_value.load.return_value = Mock()
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._run_fidelity_gate")).return_value = (
                _fidelity(is_faithful=False)
            )
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._build_evaluate_use_case_factory"))
            suite_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.EvaluationSuiteUseCase")
            )
            suite_class.return_value.execute.side_effect = UnfaithfulProxyError(_fidelity(is_faithful=False))
            stack.enter_context(patch("builtins.print"))

            with pytest.raises(UnfaithfulProxyError):
                main(args)

        assert not output_path.exists()
