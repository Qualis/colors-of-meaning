from contextlib import ExitStack
from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.grounding_report import GroundingReport
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.interface.cli.grounding import (
    AuditedAnswer,
    CandidateAnswer,
    GroundingArgs,
    GroundingPipeline,
    RagSample,
    _audit,
    _build_mapper,
    _build_pipeline,
    _create_distance_calculator,
    _encode,
    _flagged_count,
    _load_sample,
    _print_table,
    _provenance_line,
    _reproduce_command,
    _report_lines,
    _verdict,
    _verdict_rows,
    _write_report,
    main,
)

MODULE = "colors_of_meaning.interface.cli.grounding"


def _config() -> Mock:
    config = Mock()
    config.training.seed = 42
    config.distance.smoothing_epsilon = 1e-8
    config.distance.sinkhorn_reg = None
    return config


def _sample() -> RagSample:
    return RagSample(
        question="What causes rain?",
        passages=["The sun evaporates water.", "Vapour condenses into clouds."],
        answers=[CandidateAnswer("grounded", "Rain forms from condensation."), CandidateAnswer("off", "Stocks fell.")],
    )


def _audited(grounded_flags: List[bool]) -> List[AuditedAnswer]:
    return [
        AuditedAnswer(
            answer=CandidateAnswer(f"a{index}", "text"),
            report=GroundingReport(divergence=1.0 if grounded else 40.0, threshold=20.0),
        )
        for index, grounded in enumerate(grounded_flags)
    ]


def _printed_lines(print_mock: Mock) -> List[str]:
    return [str(call.args[0]) for call in print_mock.call_args_list if call.args]


class TestGroundingArgs:
    def test_should_default_to_the_rag_grounding_sample(self) -> None:
        assert GroundingArgs().sample == "samples/rag_grounding.yaml"

    def test_should_default_the_flagging_threshold(self) -> None:
        assert GroundingArgs().threshold == 25.0

    def test_should_default_to_the_sliced_metric(self) -> None:
        assert GroundingArgs().metric == "sliced"


class TestLoadSample:
    def _write_sample(self, tmp_path: Path) -> Path:
        sample = tmp_path / "rag.yaml"
        sample.write_text(
            "question: What causes rain?\n"
            "passages:\n  - The sun evaporates water.\n  - Vapour condenses.\n"
            "answers:\n  - id: grounded\n    text: Rain from condensation.\n"
            "  - id: off_context\n    text: Stocks fell.\n"
        )
        return sample

    def test_should_parse_the_question(self, tmp_path: Path) -> None:
        assert _load_sample(str(self._write_sample(tmp_path))).question == "What causes rain?"

    def test_should_parse_all_passages(self, tmp_path: Path) -> None:
        assert len(_load_sample(str(self._write_sample(tmp_path))).passages) == 2

    def test_should_parse_candidate_answers_with_ids(self, tmp_path: Path) -> None:
        answers = _load_sample(str(self._write_sample(tmp_path))).answers

        assert [answer.answer_id for answer in answers] == ["grounded", "off_context"]


class TestBuildMapper:
    def test_should_load_weights_from_the_model_path(self) -> None:
        with patch(f"{MODULE}.create_color_mapper"):
            mapper = _build_mapper(GroundingArgs(model_path="m.pth"), _config())

        mapper.load_weights.assert_called_once_with("m.pth")

    def test_should_select_the_unconstrained_mapper_for_plain(self) -> None:
        with patch(f"{MODULE}.create_color_mapper") as factory:
            _build_mapper(GroundingArgs(mapper="plain"), _config())

        assert factory.call_args.args[0] == "unconstrained"

    def test_should_select_the_structured_mapper_for_structured(self) -> None:
        with patch(f"{MODULE}.create_color_mapper") as factory:
            _build_mapper(GroundingArgs(mapper="structured"), _config())

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


class TestBuildPipeline:
    def test_should_assemble_the_pipeline_from_collaborators(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.SentenceEmbeddingAdapter"))
            stack.enter_context(patch(f"{MODULE}._build_mapper"))
            stack.enter_context(patch(f"{MODULE}.load_codebook"))
            stack.enter_context(patch(f"{MODULE}._create_distance_calculator"))
            pipeline = _build_pipeline(GroundingArgs(), _config())

        assert isinstance(pipeline, GroundingPipeline)


def _mock_pipeline(report: GroundingReport) -> GroundingPipeline:
    embedder = Mock()
    embedder.encode_document_sentences.return_value = np.zeros((2, 4), dtype=np.float32)
    encode = Mock()
    encode.execute.return_value = ColoredDocument(histogram=np.full(4, 0.25, dtype=np.float64))
    use_case = Mock()
    use_case.execute_against_contexts.return_value = report
    return GroundingPipeline(embedder=embedder, encode=encode, use_case=use_case)


class TestEncode:
    def test_should_encode_text_through_the_embedder_and_encode_use_case(self) -> None:
        pipeline = _mock_pipeline(GroundingReport(1.0, 20.0))

        document = _encode(pipeline, "some text", "answer_id")

        assert isinstance(document, ColoredDocument)


class TestAudit:
    def test_should_produce_one_audited_answer_per_candidate(self) -> None:
        pipeline = _mock_pipeline(GroundingReport(1.0, 20.0))

        audited = _audit(pipeline, _sample(), threshold=20.0)

        assert len(audited) == 2

    def test_should_carry_the_port_report_for_each_answer(self) -> None:
        report = GroundingReport(1.0, 20.0)
        pipeline = _mock_pipeline(report)

        audited = _audit(pipeline, _sample(), threshold=20.0)

        assert audited[0].report is report


class TestVerdict:
    def test_should_label_a_within_threshold_answer_as_grounded(self) -> None:
        assert _verdict(GroundingReport(divergence=5.0, threshold=20.0)) == "grounded"

    def test_should_label_an_above_threshold_answer_as_off_context(self) -> None:
        assert _verdict(GroundingReport(divergence=40.0, threshold=20.0)) == "off-context"


class TestReporting:
    def test_should_render_a_verdict_row_per_answer(self) -> None:
        rows = _verdict_rows(_audited([True, False]))

        assert len(rows) == 4

    def test_should_mark_an_off_context_answer_in_its_row(self) -> None:
        rows = _verdict_rows(_audited([True, False]))

        assert rows[3].endswith("| off-context |")

    def test_should_count_only_the_flagged_answers(self) -> None:
        assert _flagged_count(_audited([True, False])) == 1

    def test_should_record_numpy_in_provenance(self) -> None:
        assert "numpy" in _provenance_line()

    def test_should_emit_the_sample_in_the_reproduce_command(self) -> None:
        assert "--sample rag.yaml" in _reproduce_command(GroundingArgs(sample="rag.yaml"))

    def test_should_title_the_report(self) -> None:
        lines = _report_lines(GroundingArgs(), _sample(), _audited([True, False]))

        assert lines[0] == "# Answer grounding audit"

    def test_should_state_the_not_a_factuality_checker_caveat(self) -> None:
        lines = _report_lines(GroundingArgs(), _sample(), _audited([True, False]))

        assert any("not a factuality or citation checker" in line for line in lines)

    def test_should_write_the_report_with_the_audit_heading(self, tmp_path: Path) -> None:
        output_path = tmp_path / "grounding_audit.md"

        _write_report(GroundingArgs(output_path=str(output_path)), _sample(), _audited([True, False]))

        assert "Answer grounding audit" in output_path.read_text()

    def test_should_print_the_verdict_rows(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_table(_audited([True, False]))

        assert any(line.startswith("| answer | divergence") for line in _printed_lines(print_mock))


def _run_main(tmp_path: Path, sample: RagSample, report: GroundingReport) -> Path:
    output_path = tmp_path / "grounding_audit.md"
    args = GroundingArgs(output_path=str(output_path))
    with ExitStack() as stack:
        config = Mock()
        config.training.seed = 42
        stack.enter_context(patch(f"{MODULE}.SynestheticConfig")).from_yaml.return_value = config
        stack.enter_context(patch(f"{MODULE}._load_sample")).return_value = sample
        stack.enter_context(patch(f"{MODULE}._build_pipeline")).return_value = _mock_pipeline(report)
        stack.enter_context(patch("builtins.print"))
        main(args)
    return output_path


class TestGroundingMain:
    def test_should_write_the_report_file(self, tmp_path: Path) -> None:
        output_path = _run_main(tmp_path, _sample(), GroundingReport(1.0, 20.0))

        assert output_path.exists()

    def test_should_raise_when_the_sample_has_no_answers(self, tmp_path: Path) -> None:
        empty = RagSample(question="q", passages=["p"], answers=[])
        with ExitStack() as stack:
            stack.enter_context(patch(f"{MODULE}.SynestheticConfig"))
            stack.enter_context(patch(f"{MODULE}._load_sample")).return_value = empty
            stack.enter_context(patch("builtins.print"))
            with pytest.raises(ValueError, match="No candidate answers"):
                main(GroundingArgs(output_path=str(tmp_path / "x.md")))


REPO_ROOT = Path(__file__).resolve().parents[4]


def _committed_report_row(prefix: str) -> str:
    report = (REPO_ROOT / "reports" / "grounding_audit.md").read_text()
    return next(line for line in report.splitlines() if line.startswith(prefix))


class TestCommittedArtifact:
    def test_should_parse_the_committed_rag_sample(self) -> None:
        sample = _load_sample(str(REPO_ROOT / "samples" / "rag_grounding.yaml"))

        assert [answer.answer_id for answer in sample.answers] == ["grounded", "off_context"]

    def test_should_keep_the_grounded_answer_below_threshold_in_the_committed_report(self) -> None:
        assert _committed_report_row("| grounded |").endswith("| grounded |")

    def test_should_flag_the_off_context_answer_in_the_committed_report(self) -> None:
        assert _committed_report_row("| off_context |").endswith("| off-context |")

    def test_should_unignore_the_committed_report_in_gitignore(self) -> None:
        assert "!reports/grounding_audit.md" in (REPO_ROOT / ".gitignore").read_text()
