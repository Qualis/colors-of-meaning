from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.rate_distortion_point import (
    RateDistortionFrontier,
    RateDistortionPoint,
)
from colors_of_meaning.infrastructure.ml.color_vq_compression_baseline import (
    ColorVqCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.gzip_compression_baseline import (
    GzipCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.pq_compression_baseline import (
    PQCompressionBaseline,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.interface.cli.rate_distortion import (
    RateDistortionArgs,
    _build_baseline_factory,
    _build_dataset_repository,
    _build_evaluate_factory,
    _create_distance_calculator,
    _pq_subquantizers,
    _source_flags,
    main,
)

MODULE = "colors_of_meaning.interface.cli.rate_distortion"


def _frontier() -> RateDistortionFrontier:
    return RateDistortionFrontier(
        [
            RateDistortionPoint("color_vq", 3.0, 5.0, 0.70),
            RateDistortionPoint("pq", 3.0, 0.02, None),
            RateDistortionPoint("gzip", 48.0, 0.0, None),
        ]
    )


def _distance_config() -> Mock:
    config = Mock()
    config.distance.sinkhorn_reg = 1.0
    config.distance.smoothing_epsilon = 1e-8
    return config


class TestPqSubquantizers:
    def test_should_use_one_subquantizer_for_smallest_budget(self) -> None:
        assert _pq_subquantizers(2) == 1

    def test_should_match_color_bits_at_largest_budget(self) -> None:
        assert _pq_subquantizers(16) == 4

    def test_should_clamp_to_at_least_one_subquantizer(self) -> None:
        assert _pq_subquantizers(1) == 1


class TestCreateDistanceCalculator:
    def test_should_build_wasserstein_calculator(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("wasserstein", codebook, _distance_config())

        assert isinstance(calculator, WassersteinDistanceCalculator)

    def test_should_build_jensen_shannon_calculator(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        calculator = _create_distance_calculator("jensen_shannon", codebook, _distance_config())

        assert isinstance(calculator, JensenShannonDistanceCalculator)

    def test_should_raise_for_unknown_distance(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)

        with pytest.raises(ValueError, match="Unknown distance"):
            _create_distance_calculator("cosine", codebook, _distance_config())


class TestBuildBaselineFactory:
    def test_should_build_color_vq_baseline(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        assert isinstance(factory("color_vq", 2), ColorVqCompressionBaseline)

    def test_should_build_pq_baseline_for_pq_method(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        assert isinstance(factory("pq", 16), PQCompressionBaseline)

    def test_should_match_pq_subquantizers_to_color_bits(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        assert factory("pq", 16).num_subspaces == 4

    def test_should_set_pq_centroids_for_three_bit_subquantizers(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        assert factory("pq", 16).num_centroids == 8

    def test_should_build_gzip_baseline_only_at_primary_budget(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        assert isinstance(factory("gzip", 2), GzipCompressionBaseline)

    def test_should_skip_gzip_at_non_primary_budget(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        assert factory("gzip", 4) is None

    def test_should_raise_for_unknown_method(self) -> None:
        factory = _build_baseline_factory(Mock(), _distance_config(), primary_budget=2)

        with pytest.raises(ValueError, match="Unknown method"):
            factory("unknown", 2)


class TestBuildEvaluateFactory:
    def test_should_build_evaluate_use_case_for_color_vq(self) -> None:
        factory = _build_evaluate_factory(RateDistortionArgs(), _distance_config(), Mock(), Mock(), Mock())

        assert isinstance(factory("color_vq", 2), EvaluateUseCase)

    def test_should_skip_downstream_evaluation_for_non_color_methods(self) -> None:
        factory = _build_evaluate_factory(RateDistortionArgs(), _distance_config(), Mock(), Mock(), Mock())

        assert factory("pq", 2) is None


class TestBuildDatasetRepository:
    def test_should_build_document_corpus_adapter_for_documents_source(self, mocker) -> None:
        adapter = mocker.patch(f"{MODULE}.DocumentCorpusDatasetAdapter")

        result = _build_dataset_repository(RateDistortionArgs(source="documents", documents_dir="docs"))

        assert result is adapter.return_value

    def test_should_pass_documents_dir_to_the_corpus_adapter(self, mocker) -> None:
        adapter = mocker.patch(f"{MODULE}.DocumentCorpusDatasetAdapter")

        _build_dataset_repository(RateDistortionArgs(source="documents", documents_dir="docs"))

        assert adapter.call_args.kwargs["documents_dir"] == "docs"

    def test_should_pass_split_strategy_to_the_corpus_adapter(self, mocker) -> None:
        adapter = mocker.patch(f"{MODULE}.DocumentCorpusDatasetAdapter")

        _build_dataset_repository(RateDistortionArgs(source="documents", split_strategy="paragraph"))

        assert adapter.call_args.kwargs["split_strategy"] == "paragraph"

    def test_should_build_hugging_face_adapter_for_dataset_source(self, mocker) -> None:
        agnews = mocker.patch(f"{MODULE}.AGNewsDatasetAdapter")

        result = _build_dataset_repository(RateDistortionArgs(source="dataset", dataset="ag_news"))

        assert result is agnews.return_value


class TestSourceFlags:
    def test_should_emit_the_documents_source_flag(self) -> None:
        assert "--source documents" in _source_flags(RateDistortionArgs(source="documents"))

    def test_should_emit_the_split_strategy_for_documents(self) -> None:
        assert "--split-strategy work" in _source_flags(RateDistortionArgs(source="documents"))

    def test_should_emit_only_the_dataset_flag_for_dataset_source(self) -> None:
        assert _source_flags(RateDistortionArgs(source="dataset", dataset="imdb")) == "--dataset imdb"


class TestRateDistortionCli:
    def _setup(self, mocker, tmp_path, frontier, **overrides) -> SimpleNamespace:
        mocker.patch(f"{MODULE}.SynestheticConfig").from_yaml.return_value = Mock()
        dataset = mocker.patch(f"{MODULE}.AGNewsDatasetAdapter")
        dataset.return_value.get_samples.return_value = [Mock(text="a"), Mock(text="b")]
        documents = mocker.patch(f"{MODULE}.DocumentCorpusDatasetAdapter")
        documents.return_value.get_samples.return_value = [Mock(text="a"), Mock(text="b")]
        embedding = mocker.patch(f"{MODULE}.SentenceEmbeddingAdapter")
        embedding.return_value.encode_batch.return_value = np.zeros((2, 8), dtype=np.float32)
        mocker.patch(f"{MODULE}.create_color_mapper")
        use_case = mocker.patch(f"{MODULE}.RateDistortionSweepUseCase")
        use_case.return_value.execute.return_value = frontier
        renderer = mocker.patch(f"{MODULE}.MatplotlibFigureRenderer")
        mocker.patch("builtins.print")
        args = RateDistortionArgs(
            output_path=str(tmp_path / "rate_distortion.md"),
            figure_path=str(tmp_path / "rate_distortion.png"),
            **overrides,
        )
        return SimpleNamespace(use_case=use_case, renderer=renderer, documents=documents, args=args)

    def test_should_pass_budgets_to_the_sweep(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier())

        main(context.args)

        assert context.use_case.return_value.execute.call_args.kwargs["budgets"] == [2, 4, 8, 16]

    def test_should_pass_methods_to_the_sweep(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier(), methods=["color_vq", "gzip"])

        main(context.args)

        assert context.use_case.return_value.execute.call_args.kwargs["methods"] == ["color_vq", "gzip"]

    def test_should_forward_the_accuracy_toggle_to_the_sweep(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier(), with_accuracy=True)

        main(context.args)

        assert context.use_case.return_value.execute.call_args.kwargs["with_accuracy"] is True

    def test_should_write_the_report_to_the_output_path(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier())

        main(context.args)

        assert "Rate-distortion frontier" in (tmp_path / "rate_distortion.md").read_text()

    def test_should_record_the_with_accuracy_flag_in_the_reproduce_command(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier(), with_accuracy=True)

        main(context.args)

        assert "--with-accuracy" in (tmp_path / "rate_distortion.md").read_text()

    def test_should_omit_the_with_accuracy_flag_when_not_requested(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier(), with_accuracy=False)

        main(context.args)

        assert "--with-accuracy" not in (tmp_path / "rate_distortion.md").read_text()

    def test_should_render_the_figure_to_the_figure_path(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier())

        main(context.args)

        context.renderer.return_value.render_rate_distortion.assert_called_once_with(
            _frontier_arg(context), str(tmp_path / "rate_distortion.png")
        )

    def test_should_build_the_document_corpus_adapter_when_source_is_documents(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier(), source="documents", documents_dir="mydocs")

        main(context.args)

        assert context.documents.call_args.kwargs["documents_dir"] == "mydocs"

    def test_should_record_the_documents_source_in_the_reproduce_command(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier(), source="documents")

        main(context.args)

        assert "--source documents" in (tmp_path / "rate_distortion.md").read_text()

    def test_should_record_the_output_path_in_the_reproduce_command(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier())

        main(context.args)

        assert f"--output-path {context.args.output_path}" in (tmp_path / "rate_distortion.md").read_text()

    def test_should_record_the_figure_path_in_the_reproduce_command(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, _frontier())

        main(context.args)

        assert f"--figure-path {context.args.figure_path}" in (tmp_path / "rate_distortion.md").read_text()


def _frontier_arg(context: SimpleNamespace) -> RateDistortionFrontier:
    return context.use_case.return_value.execute.return_value
