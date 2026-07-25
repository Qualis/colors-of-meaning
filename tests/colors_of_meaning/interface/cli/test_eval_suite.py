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
    _build_classifier,
    _build_color_classifier,
    _build_color_retriever,
    _build_evaluate_use_case_factory,
    _build_retrieval_factory_when_requested,
    _build_retrieval_use_case_factory,
    _build_retriever,
    _encode_documents,
    _fidelity_rows,
    _print_table,
    _provenance_line,
    _reproduce_command,
    _result_row,
    _result_rows,
    _results_note,
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


def _method_cell(method: str) -> EvaluationCell:
    return EvaluationCell(dataset="ag_news", method=method, distance="sliced", budget=4000, requires_fidelity=False)


def _retrieval_evaluated_cell(method: str = "color", mrr: float = 0.9, recall=None) -> EvaluatedCell:
    recall_at_k = {5: 0.8} if recall is None else recall
    cell = EvaluationCell(
        dataset="ag_news",
        method=method,
        distance="sliced",
        budget=4000,
        requires_fidelity=method == "color",
        bits_per_token=12.0,
        supports_retrieval=True,
    )
    result = EvaluationResult(accuracy=0.83, macro_f1=0.82, recall_at_k={}, mrr=0.0)
    retrieval = EvaluationResult(accuracy=0.0, macro_f1=0.0, recall_at_k=recall_at_k, mrr=mrr)
    return EvaluatedCell(cell=cell, result=result, seconds=12.3, retrieval=retrieval)


def _tfidf_skipped_cell() -> EvaluatedCell:
    cell = EvaluationCell(
        dataset="ag_news",
        method="tfidf",
        distance="sliced",
        budget=4000,
        requires_fidelity=False,
        bits_per_token=None,
        supports_retrieval=False,
    )
    result = EvaluationResult(accuracy=0.82, macro_f1=0.81, recall_at_k={}, mrr=0.0)
    return EvaluatedCell(
        cell=cell,
        result=result,
        seconds=5.0,
        retrieval_skip_reason="tfidf is classification-only; retrieval skipped",
    )


def _printed_lines(print_mock: Mock) -> List[str]:
    return [str(call.args[0]) for call in print_mock.call_args_list if call.args]


class TestEvalSuiteHelpers:
    def test_should_mark_cells_requiring_fidelity_when_distance_is_sliced(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], methods=["color"], distance="sliced"))

        assert all(cell.requires_fidelity for cell in cells)

    def test_should_not_require_fidelity_when_distance_is_exact(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["color"], distance="wasserstein"))

        assert cells[0].requires_fidelity is False

    def test_should_build_one_cell_per_dataset(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb", "newsgroups"], methods=["color"]))

        assert [cell.dataset for cell in cells] == ["ag_news", "imdb", "newsgroups"]

    def test_should_apply_uniform_budget_when_per_dataset_budgets_absent(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], methods=["color"], budget=1500))

        assert [cell.budget for cell in cells] == [1500, 1500]

    def test_should_apply_per_dataset_budgets_when_provided(self) -> None:
        cells = _build_cells(
            EvalSuiteArgs(datasets=["ag_news", "imdb", "newsgroups"], methods=["color"], budgets=[4000, 600, 600])
        )

        assert [cell.budget for cell in cells] == [4000, 600, 600]

    def test_should_build_one_cell_per_method_per_dataset(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], methods=["color", "tfidf", "hnsw"]))

        assert [(cell.dataset, cell.method) for cell in cells] == [
            ("ag_news", "color"),
            ("ag_news", "tfidf"),
            ("ag_news", "hnsw"),
            ("imdb", "color"),
            ("imdb", "tfidf"),
            ("imdb", "hnsw"),
        ]

    def test_should_share_one_budget_across_methods_of_a_dataset(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["color", "tfidf", "hnsw"], budget=4000))

        assert {cell.budget for cell in cells} == {4000}

    def test_should_require_fidelity_only_for_color_cells_under_sliced(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["color", "tfidf", "hnsw"], distance="sliced"))

        assert [cell.method for cell in cells if cell.requires_fidelity] == ["color"]

    def test_should_assign_twelve_bits_per_token_to_color(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["color"]))

        assert cells[0].bits_per_token == 12.0

    def test_should_assign_full_embedding_bits_per_token_to_hnsw(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["hnsw"]))

        assert cells[0].bits_per_token == 12288.0

    def test_should_assign_no_bits_per_token_to_tfidf(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["tfidf"]))

        assert cells[0].bits_per_token is None

    def test_should_mark_color_and_hnsw_as_retrieval_capable(self) -> None:
        cells = _build_cells(EvalSuiteArgs(datasets=["ag_news"], methods=["color", "tfidf", "hnsw"]))

        assert [cell.method for cell in cells if cell.supports_retrieval] == ["color", "hnsw"]

    def test_should_produce_identical_cells_across_repeat_builds(self) -> None:
        args = EvalSuiteArgs(datasets=["ag_news", "imdb"], methods=["color", "tfidf", "hnsw"])

        assert _build_cells(args) == _build_cells(args)

    def test_should_build_tfidf_classifier_for_a_tfidf_cell(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.TFIDFClassifier") as tfidf_class:
            classifier = _build_classifier(_method_cell("tfidf"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

        assert classifier is tfidf_class.return_value

    def test_should_build_hnsw_classifier_for_an_hnsw_cell(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.HNSWClassifier") as hnsw_class:
            classifier = _build_classifier(_method_cell("hnsw"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

        assert classifier is hnsw_class.return_value

    def test_should_build_color_classifier_for_a_color_cell(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite._build_color_classifier") as build_color:
            classifier = _build_classifier(_method_cell("color"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

        assert classifier is build_color.return_value

    def test_should_raise_for_an_unknown_classifier_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            _build_classifier(_method_cell("bogus"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

    def test_should_build_color_retriever_for_a_color_cell(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite._build_color_retriever") as build_color:
            retriever = _build_retriever(_method_cell("color"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

        assert retriever is build_color.return_value

    def test_should_build_embedding_retriever_for_an_hnsw_cell(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.EmbeddingRetriever") as retriever_class:
            retriever = _build_retriever(_method_cell("hnsw"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

        assert retriever is retriever_class.return_value

    def test_should_raise_when_building_a_retriever_for_tfidf(self) -> None:
        with pytest.raises(ValueError, match="Retrieval not supported"):
            _build_retriever(_method_cell("tfidf"), EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

    def test_should_build_color_histogram_retriever(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.QuantizedColorMapper"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.EncodeDocumentUseCase"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._create_distance_calculator"))
            retriever_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.ColorHistogramRetriever")
            )

            retriever = _build_color_retriever(Mock(), Mock(), Mock(), Mock(), "sliced")

        assert retriever is retriever_class.return_value

    def test_should_build_retrieval_use_case_for_a_capable_cell(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._setup_dataset"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._build_retriever"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SklearnMetricsCalculator"))
            use_case_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.RetrievalEvaluateUseCase")
            )
            factory = _build_retrieval_use_case_factory(EvalSuiteArgs(), Mock(), Mock(), Mock(), Mock())

            use_case = factory(_method_cell("color"))

        assert use_case is use_case_class.return_value

    def test_should_build_a_retrieval_factory_when_task_is_both(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite._build_retrieval_use_case_factory") as build:
            factory = _build_retrieval_factory_when_requested(
                EvalSuiteArgs(task="both"), Mock(), Mock(), Mock(), Mock()
            )

        assert factory is build.return_value

    def test_should_not_build_a_retrieval_factory_when_task_is_classification(self) -> None:
        factory = _build_retrieval_factory_when_requested(
            EvalSuiteArgs(task="classification"), Mock(), Mock(), Mock(), Mock()
        )

        assert factory is None

    def test_should_raise_when_per_dataset_budgets_length_mismatches_datasets(self) -> None:
        with pytest.raises(ValueError, match="one budget per dataset"):
            _build_cells(EvalSuiteArgs(datasets=["ag_news", "imdb"], budgets=[4000]))

    def test_should_return_dataset_adapter_instance_when_setting_up_dataset(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval_suite.IMDBDatasetAdapter") as adapter_class:
            adapter = _setup_dataset("imdb")

        assert adapter is adapter_class.return_value

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

    def test_should_add_mrr_column_when_retrieval_is_reported(self) -> None:
        header = _result_rows([_retrieval_evaluated_cell()])[0]

        assert "mrr" in header

    def test_should_render_real_mrr_for_a_color_retrieval_row(self) -> None:
        row = _result_row(_retrieval_evaluated_cell(mrr=0.9123), retrieval_reported=True)

        assert "0.9123" in row

    def test_should_render_recall_at_k_for_a_color_retrieval_row(self) -> None:
        row = _result_row(_retrieval_evaluated_cell(recall={5: 0.8, 10: 0.6}), retrieval_reported=True)

        assert "5:0.8000 10:0.6000" in row

    def test_should_render_not_available_retrieval_for_a_skipped_tfidf_row(self) -> None:
        row = _result_row(_tfidf_skipped_cell(), retrieval_reported=True)

        assert "| n/a | n/a |" in row

    def test_should_render_not_available_recall_when_recall_at_k_is_empty(self) -> None:
        row = _result_row(_retrieval_evaluated_cell(mrr=0.9, recall={}), retrieval_reported=True)

        assert "| 0.9000 | n/a |" in row

    def test_should_state_tfidf_skip_reason_in_note_when_retrieval_reported(self) -> None:
        assert "classification-only" in _results_note(True)

    def test_should_state_classification_only_note_when_retrieval_not_reported(self) -> None:
        assert "measured separately" in _results_note(False)

    def test_should_emit_methods_in_reproduce_command(self) -> None:
        command = _reproduce_command(EvalSuiteArgs(methods=["color", "hnsw"]))

        assert "--methods color hnsw" in command

    def test_should_emit_task_both_in_reproduce_command_when_retrieval_requested(self) -> None:
        command = _reproduce_command(EvalSuiteArgs(task="both", k_values=[1, 5]))

        assert "--task both --k-values 1 5" in command

    def test_should_not_emit_task_flag_in_reproduce_command_when_classification_only(self) -> None:
        command = _reproduce_command(EvalSuiteArgs(task="classification"))

        assert "--task" not in command


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
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.load_codebook"))
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

    def test_should_pass_a_retrieval_factory_to_the_suite_when_task_is_both(self, tmp_path: Path) -> None:
        output_path = tmp_path / "eval_results.md"
        args = EvalSuiteArgs(datasets=["ag_news"], methods=["color"], task="both", output_path=str(output_path))
        with ExitStack() as stack:
            config = Mock()
            config.training.seed = 42
            stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.SynestheticConfig")
            ).from_yaml.return_value = config
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.SentenceEmbeddingAdapter"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.create_color_mapper"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.load_codebook"))
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._run_fidelity_gate")).return_value = (
                _fidelity(is_faithful=True)
            )
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite._build_evaluate_use_case_factory"))
            suite_class = stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval_suite.EvaluationSuiteUseCase")
            )
            suite_class.return_value.execute.return_value = [_evaluated_cell("ag_news")]
            stack.enter_context(patch("builtins.print"))
            main(args)

        assert suite_class.call_args.kwargs["retrieval_use_case_factory"] is not None

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
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval_suite.load_codebook"))
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
