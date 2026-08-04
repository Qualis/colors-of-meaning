from pathlib import Path
from unittest.mock import Mock

import pytest

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.authorship_report import (
    AuthorshipCorpus,
    AuthorshipHeldOut,
    AuthorshipReport,
    AuthorshipScalingPoint,
)
from colors_of_meaning.infrastructure.evaluation.tfidf_classifier import TFIDFClassifier
from colors_of_meaning.interface.cli.authorship import (
    AuthorshipArgs,
    _authors_works_rows,
    _bits_per_token,
    _build_classifier,
    _build_color_mapper,
    _build_document_corpus,
    _build_evaluate_factory,
    _build_scaling_sweep,
    _corpus_summary,
    _format_bits,
    _held_out_rows,
    _provenance_line,
    _report_lines,
    _reproduce_command,
    _scaling_rows,
    _split_rows,
    main,
)

MODULE = "colors_of_meaning.interface.cli.authorship"


def _corpus() -> AuthorshipCorpus:
    return AuthorshipCorpus(
        authors_works=[("austen", 6), ("darwin", 4)],
        split_counts=[("train", 100), ("validation", 20), ("test", 30)],
    )


def _report() -> AuthorshipReport:
    return AuthorshipReport(
        corpus=_corpus(),
        held_out=[
            AuthorshipHeldOut("color", 12.0, 0.108, 0.106),
            AuthorshipHeldOut("tfidf", None, 0.372, 0.347),
        ],
        chance=0.0455,
        scaling=[AuthorshipScalingPoint(60, 5340, 0.092, 0.016)],
    )


class TestAuthorshipArgs:
    def test_should_default_to_the_documents_config(self) -> None:
        assert AuthorshipArgs().config == "configs/documents.yaml"

    def test_should_default_the_scaling_manifest_path(self) -> None:
        assert AuthorshipArgs().scaling_manifest == "reports/data/authorship_scaling.json"


class TestFormatBits:
    def test_should_render_a_dash_when_bits_are_absent(self) -> None:
        assert _format_bits(None) == "—"

    def test_should_render_whole_bits_without_decimals(self) -> None:
        assert _format_bits(12.0) == "12"


class TestBitsPerToken:
    def test_should_assign_twelve_bits_to_the_color_method(self) -> None:
        assert _bits_per_token(["color", "tfidf"])["color"] == 12.0

    def test_should_assign_no_bits_to_the_tfidf_method(self) -> None:
        assert _bits_per_token(["color", "tfidf"])["tfidf"] is None


class TestCorpusSummary:
    def test_should_read_the_authors_and_works_from_the_adapter(self) -> None:
        adapter = Mock()
        adapter.works_per_author.return_value = [("austen", 6)]
        adapter.get_samples.return_value = [Mock(), Mock()]

        corpus = _corpus_summary(adapter, seed=42)

        assert corpus.authors_works == [("austen", 6)]

    def test_should_count_the_paragraphs_in_each_split(self) -> None:
        adapter = Mock()
        adapter.works_per_author.return_value = [("austen", 6)]
        adapter.get_samples.return_value = [Mock(), Mock(), Mock()]

        corpus = _corpus_summary(adapter, seed=42)

        assert corpus.split_counts == [("train", 3), ("validation", 3), ("test", 3)]


class TestBuildDocumentCorpus:
    def test_should_pass_the_cap_to_the_document_corpus_adapter(self, mocker) -> None:
        adapter = mocker.patch(f"{MODULE}.DocumentCorpusDatasetAdapter")

        _build_document_corpus(AuthorshipArgs(), cap=60)

        assert adapter.call_args.kwargs["paragraphs_per_work"] == 60


class TestBuildColorMapper:
    def test_should_load_the_trained_weights_into_the_mapper(self, mocker) -> None:
        create = mocker.patch(f"{MODULE}.create_color_mapper")

        _build_color_mapper(AuthorshipArgs(model_path="model.pth"), Mock())

        create.return_value.load_weights.assert_called_once_with("model.pth")


class TestBuildClassifier:
    def test_should_build_the_tfidf_classifier(self) -> None:
        classifier = _build_classifier("tfidf", AuthorshipArgs(), Mock(), Mock(), Mock(), Mock())

        assert isinstance(classifier, TFIDFClassifier)

    def test_should_build_the_color_classifier(self, mocker) -> None:
        color = mocker.patch(f"{MODULE}.ColorHistogramClassifier")

        result = _build_classifier("color", AuthorshipArgs(), Mock(), Mock(), Mock(), Mock())

        assert result is color.return_value

    def test_should_raise_for_an_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            _build_classifier("bogus", AuthorshipArgs(), Mock(), Mock(), Mock(), Mock())


class TestBuildEvaluateFactory:
    def test_should_build_an_evaluate_use_case_for_a_method(self) -> None:
        factory = _build_evaluate_factory(AuthorshipArgs(), Mock(), Mock(), Mock(), Mock(), Mock())

        assert isinstance(factory("tfidf"), EvaluateUseCase)


class TestBuildScalingSweep:
    def test_should_return_the_sweep_run_callable(self, mocker) -> None:
        sweep = mocker.patch(f"{MODULE}.AuthorshipScalingSweep")

        result = _build_scaling_sweep(AuthorshipArgs(), Mock(), Mock(), Mock())

        assert result is sweep.return_value.run

    def test_should_default_the_sweep_to_a_single_worker(self, mocker) -> None:
        sweep = mocker.patch(f"{MODULE}.AuthorshipScalingSweep")

        _build_scaling_sweep(AuthorshipArgs(), Mock(), Mock(), Mock())

        assert sweep.call_args.kwargs["scaling_workers"] == 1

    def test_should_pass_the_requested_scaling_workers_to_the_sweep(self, mocker) -> None:
        sweep = mocker.patch(f"{MODULE}.AuthorshipScalingSweep")

        _build_scaling_sweep(AuthorshipArgs(scaling_workers=4), Mock(), Mock(), Mock())

        assert sweep.call_args.kwargs["scaling_workers"] == 4


class TestReportRows:
    def test_should_render_a_row_per_author(self) -> None:
        assert "| austen | 6 |" in _authors_works_rows(_corpus())

    def test_should_render_a_row_per_split(self) -> None:
        assert "| validation | 20 |" in _split_rows(_corpus())

    def test_should_render_the_color_held_out_row(self) -> None:
        assert "| color | 12 | 0.1080 | 0.1060 |" in _held_out_rows(_report())

    def test_should_append_the_chance_row(self) -> None:
        assert "| chance | — | 0.0455 | — |" in _held_out_rows(_report())

    def test_should_render_the_scaling_row_with_times_chance(self) -> None:
        assert "| 60 | 5340 | 0.092 ± 0.016 | 2.02× |" in _scaling_rows(_report())


class TestProvenanceLine:
    def test_should_record_the_numpy_version(self) -> None:
        assert "numpy" in _provenance_line()

    def test_should_record_the_scikit_learn_version(self) -> None:
        assert "scikit-learn" in _provenance_line()


class TestReproduceCommand:
    def test_should_reference_the_authorship_env(self) -> None:
        assert "tox -e authorship --" in _reproduce_command(AuthorshipArgs())


class TestVisualizeCommand:
    def test_should_reference_the_visualize_documents_env(self) -> None:
        from colors_of_meaning.interface.cli.authorship import _visualize_command

        assert "tox -e visualize_documents --" in _visualize_command(AuthorshipArgs())


class TestReportLines:
    def test_should_title_the_report(self) -> None:
        lines = _report_lines(AuthorshipArgs(), _report())

        assert lines[0].startswith("# Authorship by colour")

    def test_should_embed_the_a4_gallery_figure(self) -> None:
        lines = _report_lines(AuthorshipArgs(), _report())

        assert "![Per-book A4 colour signatures](figures/documents_a4_gallery.png)" in lines

    def test_should_point_to_the_lossless_codec(self) -> None:
        lines = _report_lines(AuthorshipArgs(), _report())

        assert any("Lossless A4 colour-barcode" in line for line in lines)


class TestMain:
    def _adapter(self) -> Mock:
        adapter = Mock()
        adapter.works_per_author.return_value = [("austen", 6)]
        adapter.get_samples.return_value = [Mock(), Mock()]
        return adapter

    def _patch_main(self, mocker, adapter: Mock, report: AuthorshipReport) -> None:
        config = mocker.patch(f"{MODULE}.SynestheticConfig").from_yaml.return_value
        config.training.seed = 42
        mocker.patch(f"{MODULE}._build_document_corpus", return_value=adapter)
        mocker.patch(f"{MODULE}.SentenceEmbeddingAdapter")
        mocker.patch(f"{MODULE}._build_color_mapper")
        mocker.patch(f"{MODULE}.load_codebook")
        mocker.patch(f"{MODULE}._build_evaluate_factory")
        mocker.patch(f"{MODULE}._build_scaling_sweep")
        mocker.patch(f"{MODULE}.JsonAuthorshipScalingRepository")
        use_case = mocker.patch(f"{MODULE}.AuthorshipReportUseCase")
        use_case.return_value.execute.return_value = report
        mocker.patch("builtins.print")

    def test_should_write_the_report_to_the_output_path(self, mocker, tmp_path: Path) -> None:
        self._patch_main(mocker, self._adapter(), _report())
        output_path = str(tmp_path / "documents_authorship.md")

        main(AuthorshipArgs(output_path=output_path))

        assert "Authorship by colour" in Path(output_path).read_text()

    def test_should_raise_when_no_authors_qualify(self, mocker, tmp_path: Path) -> None:
        adapter = Mock()
        adapter.works_per_author.return_value = []
        adapter.get_samples.return_value = []
        self._patch_main(mocker, adapter, _report())

        with pytest.raises(ValueError, match="No qualifying authors"):
            main(AuthorshipArgs(output_path=str(tmp_path / "x.md")))
