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
from colors_of_meaning.infrastructure.ml.sliced_wasserstein_distance_calculator import (
    SlicedWassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.interface.cli.rate_distortion import (
    NO_INVERSION_SUMMARY,
    RateDistortionArgs,
    SweepRun,
    _build_baseline_factory,
    _build_dataset_repository,
    _build_evaluate_factory,
    _create_distance_calculator,
    _diagnosis,
    _diagnosis_lines,
    _comparable_diagnosis,
    _inversion_summary,
    _pq_subquantizers,
    _reject_empty_sweep_axes,
    _resolved_seeds,
    _run_methods,
    _source_flags,
    _sweep_grid,
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

    def test_should_build_sliced_wasserstein_calculator(self) -> None:
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=2)
        config = _distance_config()
        config.training.seed = 42

        calculator = _create_distance_calculator("sliced", codebook, config)

        assert isinstance(calculator, SlicedWassersteinDistanceCalculator)

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
        factory = _build_evaluate_factory(
            RateDistortionArgs(), _distance_config(), Mock(), Mock(), Mock(), "wasserstein"
        )

        assert isinstance(factory("color_vq", 2), EvaluateUseCase)

    def test_should_skip_downstream_evaluation_for_non_color_methods(self) -> None:
        factory = _build_evaluate_factory(
            RateDistortionArgs(), _distance_config(), Mock(), Mock(), Mock(), "wasserstein"
        )

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


def _accuracy_frontier(accuracies: dict) -> RateDistortionFrontier:
    return RateDistortionFrontier(
        [RateDistortionPoint("color_vq", bits, 5.0, accuracy) for bits, accuracy in accuracies.items()]
    )


def _seeded_config(seed: int = 42) -> Mock:
    config = Mock()
    config.training.seed = seed
    return config


class TestResolvedSeeds:
    def test_should_fall_back_to_the_configured_seed_when_none_are_requested(self) -> None:
        assert _resolved_seeds(RateDistortionArgs(), _seeded_config(7)) == [7]

    def test_should_honour_the_requested_seeds(self) -> None:
        assert _resolved_seeds(RateDistortionArgs(seeds=[1, 2]), _seeded_config()) == [1, 2]

    def test_should_reject_a_sweep_axis_before_any_encoding(self, mocker, tmp_path) -> None:
        context = TestRateDistortionCli()._setup(mocker, tmp_path, _frontier(), methods=[])

        with pytest.raises(ValueError, match="methods must name at least one value"):
            main(context.args)


class TestSweepGrid:
    def test_should_cross_every_distance_with_every_seed(self) -> None:
        args = RateDistortionArgs(distance=["jensen_shannon", "wasserstein"], seeds=[1, 2], with_accuracy=True)

        assert _sweep_grid(args, _seeded_config()) == [
            ("jensen_shannon", 1),
            ("jensen_shannon", 2),
            ("wasserstein", 1),
            ("wasserstein", 2),
        ]

    def test_should_sweep_a_single_cell_when_no_accuracy_is_requested(self) -> None:
        args = RateDistortionArgs(distance=["jensen_shannon", "wasserstein"], seeds=[1, 2], with_accuracy=False)

        assert _sweep_grid(args, _seeded_config()) == [("jensen_shannon", 1)]

    def test_should_sweep_every_method_only_on_the_primary_run(self) -> None:
        args = RateDistortionArgs(methods=["color_vq", "gzip"])

        assert (_run_methods(args, 0), _run_methods(args, 1)) == (["color_vq", "gzip"], ["color_vq"])


class TestDiagnosis:
    def test_should_average_the_accuracy_across_seeds_for_each_distance(self) -> None:
        runs = [
            SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.70})),
            SweepRun("jensen_shannon", 2, _accuracy_frontier({9.0: 0.80})),
        ]

        points = _diagnosis(RateDistortionArgs(distance=["jensen_shannon"]), runs)

        assert points[0].mean_accuracy == pytest.approx(0.75, abs=1e-9)

    def test_should_report_the_seed_spread_across_runs(self) -> None:
        runs = [
            SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.70})),
            SweepRun("jensen_shannon", 2, _accuracy_frontier({9.0: 0.80})),
        ]

        points = _diagnosis(RateDistortionArgs(distance=["jensen_shannon"]), runs)

        assert points[0].stdev_accuracy == pytest.approx(0.0707106781, abs=1e-6)

    def test_should_report_a_zero_spread_for_a_single_seed(self) -> None:
        runs = [SweepRun("wasserstein", 1, _accuracy_frontier({9.0: 0.70}))]

        points = _diagnosis(RateDistortionArgs(distance=["wasserstein"]), runs)

        assert points[0].stdev_accuracy == 0.0

    def test_should_ignore_points_without_a_measured_accuracy(self) -> None:
        runs = [SweepRun("wasserstein", 1, RateDistortionFrontier([RateDistortionPoint("color_vq", 9.0, 5.0, None)]))]

        assert _diagnosis(RateDistortionArgs(distance=["wasserstein"]), runs) == []

    def test_should_ignore_runs_measured_under_another_distance(self) -> None:
        runs = [SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.70}))]

        assert _diagnosis(RateDistortionArgs(distance=["wasserstein"]), runs) == []


class TestInversionVerdict:
    def test_should_call_the_inversion_a_metric_artifact_when_only_one_distance_inverts(self) -> None:
        summary = _inversion_summary({"jensen_shannon": True, "wasserstein": False})

        assert "metric artifact" in summary and "jensen_shannon" in summary

    def test_should_refuse_to_attribute_the_shape_when_only_one_distance_was_measured(self) -> None:
        assert "cannot separate" in _inversion_summary({"jensen_shannon": True})

    def test_should_attribute_the_drop_to_the_bit_budget_when_every_distance_inverts(self) -> None:
        assert "property of the bit budget" in _inversion_summary({"jensen_shannon": True, "wasserstein": True})

    def test_should_report_no_inversion_when_accuracy_peaks_at_the_widest_budget(self) -> None:
        assert "No distance inverts" in _inversion_summary({"jensen_shannon": False, "wasserstein": False})

    def test_should_write_the_metric_artifact_verdict_when_one_distance_peaks_early(self) -> None:
        runs = [
            SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.76, 12.0: 0.63})),
            SweepRun("wasserstein", 1, _accuracy_frontier({9.0: 0.78, 12.0: 0.80})),
        ]
        args = RateDistortionArgs(distance=["jensen_shannon", "wasserstein"])

        lines = _diagnosis_lines(args, _diagnosis(args, runs))

        assert any("metric artifact" in line for line in lines)

    def test_should_write_the_no_inversion_verdict_when_every_distance_peaks_widest(self) -> None:
        runs = [
            SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.70, 12.0: 0.76})),
            SweepRun("wasserstein", 1, _accuracy_frontier({9.0: 0.78, 12.0: 0.80})),
        ]
        args = RateDistortionArgs(distance=["jensen_shannon", "wasserstein"])

        lines = _diagnosis_lines(args, _diagnosis(args, runs))

        assert NO_INVERSION_SUMMARY in lines

    def test_should_name_the_peak_budget_for_each_measured_distance(self) -> None:
        runs = [SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.76, 12.0: 0.63}))]
        args = RateDistortionArgs(distance=["jensen_shannon"])

        lines = _diagnosis_lines(args, _diagnosis(args, runs))

        assert any("peaks at 9.00 bits (0.7600)" in line for line in lines)

    def test_should_emit_no_diagnosis_section_when_no_accuracy_was_measured(self) -> None:
        assert _diagnosis_lines(RateDistortionArgs(), []) == []

    def test_should_skip_a_distance_that_produced_no_measurement(self) -> None:
        runs = [SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.76}))]
        args = RateDistortionArgs(distance=["jensen_shannon", "wasserstein"])

        lines = _diagnosis_lines(args, _diagnosis(args, runs))

        assert not any("Under `wasserstein`" in line for line in lines)


class TestComparableDiagnosis:
    def test_should_withhold_the_diagnosis_when_a_single_cell_was_swept(self) -> None:
        runs = [SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.76}))]

        assert _comparable_diagnosis(RateDistortionArgs(distance=["jensen_shannon"]), runs) == []

    def test_should_report_the_diagnosis_when_more_than_one_cell_was_swept(self) -> None:
        runs = [
            SweepRun("jensen_shannon", 1, _accuracy_frontier({9.0: 0.76})),
            SweepRun("jensen_shannon", 2, _accuracy_frontier({9.0: 0.70})),
        ]

        assert _comparable_diagnosis(RateDistortionArgs(distance=["jensen_shannon"]), runs) != []


class TestSweepAxisValidation:
    def test_should_reject_an_empty_distance_list(self) -> None:
        with pytest.raises(ValueError, match="distance must name at least one value"):
            _reject_empty_sweep_axes(RateDistortionArgs(distance=[]))

    def test_should_reject_an_empty_seed_list(self) -> None:
        with pytest.raises(ValueError, match="seeds must name at least one value"):
            _reject_empty_sweep_axes(RateDistortionArgs(seeds=[]))

    def test_should_reject_an_empty_budget_list(self) -> None:
        with pytest.raises(ValueError, match="budgets must name at least one value"):
            _reject_empty_sweep_axes(RateDistortionArgs(budgets=[]))

    def test_should_reject_a_repeated_distance(self) -> None:
        with pytest.raises(ValueError, match="distance must not repeat a value"):
            _reject_empty_sweep_axes(RateDistortionArgs(distance=["wasserstein", "wasserstein"]))

    def test_should_reject_a_repeated_seed(self) -> None:
        with pytest.raises(ValueError, match="seeds must not repeat a value"):
            _reject_empty_sweep_axes(RateDistortionArgs(seeds=[42, 42]))

    def test_should_reject_an_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown method: bogus"):
            _reject_empty_sweep_axes(RateDistortionArgs(methods=["bogus"]))

    def test_should_reject_a_budget_below_two_bins_per_dimension(self) -> None:
        with pytest.raises(ValueError, match="budgets must be at least 2"):
            _reject_empty_sweep_axes(RateDistortionArgs(budgets=[0]))

    def test_should_accept_the_default_sweep_axes(self) -> None:
        assert _reject_empty_sweep_axes(RateDistortionArgs()) is None


class TestSweepFanOut:
    def test_should_run_one_sweep_per_distance_and_seed(self, mocker, tmp_path) -> None:
        context = TestRateDistortionCli()._setup(
            mocker, tmp_path, _frontier(), distance=["jensen_shannon", "wasserstein"], seeds=[1, 2], with_accuracy=True
        )

        main(context.args)

        assert context.use_case.return_value.execute.call_count == 4

    def test_should_pass_each_seed_through_to_the_sweep(self, mocker, tmp_path) -> None:
        context = TestRateDistortionCli()._setup(
            mocker, tmp_path, _frontier(), distance=["jensen_shannon", "wasserstein"], seeds=[1, 2], with_accuracy=True
        )

        main(context.args)

        seeds = [call.kwargs["seed"] for call in context.use_case.return_value.execute.call_args_list]
        assert seeds == [1, 2, 1, 2]

    def test_should_build_the_downstream_classifier_with_the_swept_distance(self, mocker, tmp_path) -> None:
        factory = mocker.patch(f"{MODULE}._build_evaluate_factory")
        context = TestRateDistortionCli()._setup(
            mocker, tmp_path, _frontier(), distance=["jensen_shannon", "wasserstein"], seeds=[1], with_accuracy=True
        )

        main(context.args)

        assert [call.args[-1] for call in factory.call_args_list] == ["jensen_shannon", "wasserstein"]

    def test_should_narrow_the_later_sweeps_to_the_color_codec(self, mocker, tmp_path) -> None:
        context = TestRateDistortionCli()._setup(
            mocker, tmp_path, _frontier(), distance=["jensen_shannon", "wasserstein"], seeds=[1], with_accuracy=True
        )

        main(context.args)

        methods = [call.kwargs["methods"] for call in context.use_case.return_value.execute.call_args_list]
        assert methods == [["color_vq", "gzip", "pq"], ["color_vq"]]
