from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from colors_of_meaning.application.use_case.compare_structure_objectives_use_case import ArmRequest
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.objective_comparison import (
    ObjectiveArmResult,
    ObjectiveComparison,
)
from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import (
    AGNewsDatasetAdapter,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structure_objectives import (
    cosine_centred,
    margin_ranking,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.retrieval_evaluate_use_case import (
    RetrievalEvaluateUseCase,
)
from colors_of_meaning.infrastructure.evaluation.pca_projection_control import (
    PcaProjectionControl,
)
from colors_of_meaning.interface.cli.compare_objectives import (
    BASELINE_ARM,
    COMMITTED_CONTROL,
    NOISE_CONTROL,
    PCA_CONTROL,
    UNCONSTRAINED_HEAD,
    UNCONSTRAINED_HEAD_PRECLAMP,
    CompareObjectivesArgs,
    TrainedMapperCache,
    _adoption_sentence,
    _adoption_verdict,
    _arm_objective,
    _bits_per_token,
    _arm_rows,
    _build_evaluate_factory,
    _build_lab_colors_factory,
    _build_mapper,
    _build_retrieval_factory,
    _downstream_arms,
    _format_metric,
    _format_recall,
    _held_out_embeddings,
    _format_margin,
    _margin_rows,
    _reject_unrunnable_request,
    _reproduce_command,
    _setup_dataset,
    _training_arm,
    main,
)

MODULE = "colors_of_meaning.interface.cli.compare_objectives"
CHALLENGER_ARM = "delta_e_correlation"
EMBEDDING_DIM = 6


def _config() -> Mock:
    config = Mock()
    config.projector.embedding_dim = EMBEDDING_DIM
    config.projector.hidden_dim_1 = 5
    config.projector.hidden_dim_2 = 4
    config.projector.dropout_rate = 0.0
    config.training.device = "cpu"
    config.training.seed = 42
    config.training.epochs = 1
    config.training.learning_rate = 0.01
    config.training.batch_size = 4
    config.codebook.bins_per_dimension = 2
    config.dataset.train_split = "train"
    config.dataset.test_split = "test"
    config.dataset.max_samples = 8
    return config


def _embeddings(count: int = 6) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(count, EMBEDDING_DIM)).astype(np.float32)


def _request(arm: str, seed: int = 42) -> ArmRequest:
    return ArmRequest(arm=arm, seed=seed, train_embeddings=_embeddings(), eval_embeddings=_embeddings(4))


def _arm(arm: str, mean_rho: float, accuracy: float = 0.8, stdev_rho: float = 0.02) -> ObjectiveArmResult:
    return ObjectiveArmResult(arm=arm, mean_rho=mean_rho, stdev_rho=stdev_rho, seeds=8, accuracy=accuracy)


def _held_artifact_comparison() -> ObjectiveComparison:
    return _comparison(
        _arm(BASELINE_ARM, -0.39, accuracy=0.63),
        _arm(CHALLENGER_ARM, -0.60, accuracy=0.73),
        controls=[_arm(COMMITTED_CONTROL, -0.38, accuracy=0.82)],
    )


def _comparison(*results: ObjectiveArmResult, controls=()) -> ObjectiveComparison:
    return ObjectiveComparison(results=list(results), baseline_arm=BASELINE_ARM, controls=list(controls))


def _committed_artifact(tmp_path) -> Path:
    artifact = tmp_path / "projector.pth"
    artifact.write_bytes(b"")
    return artifact


def _stash_checkpoint(train_use_case: Mock) -> float:
    train_use_case.call_args.kwargs["color_mapper"].epoch_checkpoints().append({"weights": 1})
    return 0.0


class TestArmRegistry:
    def test_should_resolve_the_named_objective_for_an_objective_arm(self) -> None:
        assert _arm_objective("margin_ranking") is margin_ranking

    def test_should_fall_back_to_the_baseline_objective_for_a_control(self) -> None:
        assert _arm_objective(NOISE_CONTROL) is cosine_centred

    def test_should_reject_an_arm_name_it_does_not_know(self) -> None:
        with pytest.raises(ValueError, match="Unknown arm: delta_e_corelation"):
            _arm_objective("delta_e_corelation")

    def test_should_train_the_preclamp_control_under_the_unconstrained_head_arm(self) -> None:
        assert _training_arm(UNCONSTRAINED_HEAD_PRECLAMP) == UNCONSTRAINED_HEAD

    def test_should_leave_an_ordinary_arm_as_its_own_training_arm(self) -> None:
        assert _training_arm(BASELINE_ARM) == BASELINE_ARM

    def test_should_remove_the_lab_head_for_the_unconstrained_control(self) -> None:
        mapper = _build_mapper(UNCONSTRAINED_HEAD, 42, _config())

        assert mapper.network.constrain_to_lab is False

    def test_should_keep_the_lab_head_for_an_objective_arm(self) -> None:
        mapper = _build_mapper(BASELINE_ARM, 42, _config())

        assert mapper.network.constrain_to_lab is True

    def test_should_resolve_the_dataset_adapter_by_name(self) -> None:
        assert isinstance(_setup_dataset("ag_news"), AGNewsDatasetAdapter)


class TestTrainedMapperCache:
    def _cache(self, mocker) -> SimpleNamespace:
        train_use_case = mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))
        return SimpleNamespace(cache=cache, train_use_case=train_use_case)

    def test_should_train_an_objective_arm_once_per_seed(self, mocker) -> None:
        context = self._cache(mocker)

        context.cache.mapper(_request(BASELINE_ARM))
        context.cache.mapper(_request(BASELINE_ARM))

        assert context.train_use_case.call_count == 1

    def test_should_share_one_training_between_the_head_control_and_its_preclamp_twin(self, mocker) -> None:
        context = self._cache(mocker)

        first = context.cache.mapper(_request(UNCONSTRAINED_HEAD))
        second = context.cache.mapper(_request(UNCONSTRAINED_HEAD_PRECLAMP))

        assert first is second

    def test_should_leave_the_noise_control_untrained(self, mocker) -> None:
        context = self._cache(mocker)

        context.cache.mapper(_request(NOISE_CONTROL))

        assert context.train_use_case.call_count == 0

    def test_should_train_each_seed_separately(self, mocker) -> None:
        context = self._cache(mocker)

        context.cache.mapper(_request(BASELINE_ARM, seed=1))
        context.cache.mapper(_request(BASELINE_ARM, seed=2))

        assert context.train_use_case.call_count == 2

    def test_should_select_the_checkpoint_on_the_held_out_selection_slice(self, mocker) -> None:
        context = self._cache(mocker)
        selection = context.cache._selection_embeddings

        context.cache.mapper(_request(BASELINE_ARM))

        assert context.train_use_case.return_value.execute.call_args.kwargs["evaluation_embeddings"] is selection


class TestLabColorsFactory:
    def test_should_project_the_pca_control_without_training(self, mocker) -> None:
        train_use_case = mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))

        _build_lab_colors_factory(cache)(_request(PCA_CONTROL))

        assert train_use_case.call_count == 0

    def test_should_project_the_pca_control_through_the_fitted_linear_map(self, mocker) -> None:
        mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))
        request = _request(PCA_CONTROL)

        colors = _build_lab_colors_factory(cache)(request)

        expected = (
            PcaProjectionControl(seed=request.seed).fit(request.train_embeddings).transform(request.eval_embeddings)
        )
        assert colors == expected

    def test_should_read_the_preclamp_control_from_the_raw_coordinates(self, mocker) -> None:
        mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        coordinates = np.full((4, 3), 900.0)
        mocker.patch.object(PyTorchColorMapper, "embed_batch_to_coordinates", return_value=coordinates)
        rescale = mocker.patch(f"{MODULE}.rescale_preserving_ranks", return_value=[])
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))

        _build_lab_colors_factory(cache)(_request(UNCONSTRAINED_HEAD_PRECLAMP))

        assert rescale.call_args.args[0] is coordinates

    def test_should_read_an_objective_arm_through_the_clamped_lab_head(self, mocker) -> None:
        mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))
        request = _request(BASELINE_ARM)

        colors = _build_lab_colors_factory(cache)(request)

        expected = cache.mapper(request).embed_batch_to_lab(request.eval_embeddings)
        assert colors == expected


class TestDownstreamNomination:
    def test_should_honour_explicit_downstream_arms(self) -> None:
        args = CompareObjectivesArgs(downstream_arms=["margin_ranking"], downstream_controls=[])

        assert _downstream_arms(args, _comparison(_arm(BASELINE_ARM, -0.39))) == ["margin_ranking"]

    def test_should_nominate_the_baseline_and_the_strongest_challenger(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.55), _arm("margin_ranking", -0.45))
        args = CompareObjectivesArgs(downstream_controls=[])

        assert _downstream_arms(args, comparison) == [BASELINE_ARM, CHALLENGER_ARM]

    def test_should_nominate_nothing_when_the_top_k_is_disabled(self) -> None:
        args = CompareObjectivesArgs(downstream_top_k=0)

        assert _downstream_arms(args, _comparison(_arm(BASELINE_ARM, -0.39))) == []

    def test_should_always_measure_the_committed_control_downstream(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.55))

        assert _downstream_arms(CompareObjectivesArgs(), comparison)[-1] == COMMITTED_CONTROL

    def test_should_never_nominate_an_arm_that_is_not_backed_by_a_mapper(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(PCA_CONTROL, -0.70))
        args = CompareObjectivesArgs(downstream_controls=[])

        assert PCA_CONTROL not in _downstream_arms(args, comparison)

    def test_should_skip_a_downstream_control_that_was_not_requested_as_a_control(self) -> None:
        args = CompareObjectivesArgs(controls=[NOISE_CONTROL])

        assert COMMITTED_CONTROL not in _downstream_arms(args, _comparison(_arm(BASELINE_ARM, -0.39)))


class TestReportFormatting:
    def test_should_render_a_missing_metric_as_not_available(self) -> None:
        assert _format_metric(None) == "n/a"

    def test_should_render_a_measured_metric_to_four_decimals(self) -> None:
        assert _format_metric(0.8125) == "0.8125"

    def test_should_render_missing_recall_as_not_available(self) -> None:
        assert _format_recall(None) == "n/a"

    def test_should_render_recall_at_each_depth(self) -> None:
        assert _format_recall({5: 0.5, 1: 0.25}) == "1:0.2500 5:0.5000"

    def test_should_write_one_table_row_per_arm(self) -> None:
        rows = _arm_rows([_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.45)])

        assert len(rows) == 4

    def test_should_mark_a_challenger_that_fails_the_rule(self) -> None:
        rows = _margin_rows(_comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.40)))

        assert rows[-1].endswith("| no - margin under threshold |")

    def test_should_render_an_unmeasurable_margin_as_not_available(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.60, stdev_rho=0.0))

        assert _format_margin(comparison, comparison.challengers()[0]) == "n/a"

    def test_should_render_a_measurable_margin_in_pooled_sigma(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.40), _arm(CHALLENGER_ARM, -0.50))

        assert _format_margin(comparison, comparison.challengers()[0]) == "5.00"

    def test_should_record_the_adoption_rule_parameters_in_the_reproduce_command(self) -> None:
        command = _reproduce_command(CompareObjectivesArgs())

        assert "--adoption-threshold-sigma 2.0 --max-accuracy-drop 0.01" in command

    def test_should_mark_a_challenger_that_clears_the_rule(self) -> None:
        rows = _margin_rows(_comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.60)))

        assert rows[-1].endswith("| yes |")

    def test_should_state_that_the_committed_projector_is_unchanged_when_the_baseline_holds(self) -> None:
        sentence = _adoption_sentence(_comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.40)))

        assert "unchanged" in sentence

    def test_should_state_that_the_projector_is_retrained_when_a_challenger_wins(self) -> None:
        sentence = _adoption_sentence(_comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.60)))

        assert "retrained" in sentence

    def test_should_hold_the_shipped_artifact_when_it_outscores_the_adopted_arm(self) -> None:
        assert "left in place" in _adoption_sentence(_held_artifact_comparison())

    def test_should_quote_both_accuracies_when_the_shipped_artifact_is_held(self) -> None:
        assert "0.8200 against the adopted arm's 0.7300" in _adoption_sentence(_held_artifact_comparison())

    def test_should_replace_the_shipped_artifact_when_the_adopted_arm_matches_its_accuracy(self) -> None:
        comparison = _comparison(
            _arm(BASELINE_ARM, -0.39),
            _arm(CHALLENGER_ARM, -0.60, accuracy=0.82),
            controls=[_arm(COMMITTED_CONTROL, -0.38, accuracy=0.82)],
        )

        assert "retrained" in _adoption_sentence(comparison)

    def test_should_replace_the_shipped_artifact_when_no_committed_control_was_scored(self) -> None:
        comparison = _comparison(
            _arm(BASELINE_ARM, -0.39),
            _arm(CHALLENGER_ARM, -0.60),
            controls=[_arm(NOISE_CONTROL, -0.05), _arm(PCA_CONTROL, -0.30)],
        )

        assert "retrained" in _adoption_sentence(comparison)

    def test_should_replace_the_shipped_artifact_when_it_was_not_measured_downstream(self) -> None:
        committed = ObjectiveArmResult(arm=COMMITTED_CONTROL, mean_rho=-0.38, stdev_rho=0.0, seeds=8)
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.60), controls=[committed])

        assert "retrained" in _adoption_sentence(comparison)

    def test_should_explain_a_challenger_rejected_for_a_margin_under_the_threshold(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.40))

        assert _adoption_verdict(comparison, comparison.challengers()[0]) == "no - margin under threshold"

    def test_should_explain_a_challenger_rejected_for_an_unmeasured_accuracy(self) -> None:
        challenger = ObjectiveArmResult(arm=CHALLENGER_ARM, mean_rho=-0.60, stdev_rho=0.02, seeds=8)
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), challenger)

        assert _adoption_verdict(comparison, challenger) == "no - accuracy not measured"

    def test_should_explain_a_challenger_rejected_by_the_accuracy_guard(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.60, accuracy=0.5))

        assert _adoption_verdict(comparison, comparison.challengers()[0]) == "no - accuracy guard"

    def test_should_explain_a_challenger_rejected_for_an_unmeasurable_spread(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39, stdev_rho=0.0), _arm(CHALLENGER_ARM, -0.60, stdev_rho=0.0))

        assert _adoption_verdict(comparison, comparison.challengers()[0]) == "no - seed spread is unmeasurable"

    def test_should_mark_a_qualifying_challenger_as_clearing_the_rule(self) -> None:
        comparison = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.60))

        assert _adoption_verdict(comparison, comparison.challengers()[0]) == "yes"

    def test_should_record_every_seed_in_the_reproduce_command(self) -> None:
        command = _reproduce_command(CompareObjectivesArgs(seeds=[1, 2]))

        assert "--seeds 1 2" in command


class TestHeldOutSplit:
    def test_should_split_the_held_out_encoding_into_selection_and_structure_slices(self) -> None:
        dataset_repository = Mock()
        dataset_repository.get_samples.return_value = [Mock(text="a")] * 6
        embedding_adapter = Mock()
        embedding_adapter.encode_batch.return_value = _embeddings(6)
        args = CompareObjectivesArgs(selection_samples=2, structure_samples=4)

        selection, structure = _held_out_embeddings(dataset_repository, embedding_adapter, args, _config())

        assert (len(selection), len(structure)) == (2, 4)

    def test_should_request_both_slices_in_one_draw_from_the_test_split(self) -> None:
        dataset_repository = Mock()
        dataset_repository.get_samples.return_value = [Mock(text="a")] * 6
        embedding_adapter = Mock()
        embedding_adapter.encode_batch.return_value = _embeddings(6)
        args = CompareObjectivesArgs(selection_samples=2, structure_samples=4)

        _held_out_embeddings(dataset_repository, embedding_adapter, args, _config())

        assert dataset_repository.get_samples.call_args.kwargs["max_samples"] == 6


class TestCompareObjectivesCli:
    def _setup(self, mocker, tmp_path, **overrides) -> SimpleNamespace:
        mocker.patch(f"{MODULE}.SynestheticConfig").from_yaml.return_value = _config()
        dataset = mocker.patch(f"{MODULE}.AGNewsDatasetAdapter")
        dataset.return_value.get_samples.return_value = [Mock(text="a")] * 6
        embedding = mocker.patch(f"{MODULE}.SentenceEmbeddingAdapter")
        embedding.return_value.encode_batch.return_value = _embeddings(6)
        mocker.patch(f"{MODULE}.load_codebook", return_value=ColorCodebook.create_uniform_grid(bins_per_dimension=2))
        use_case = mocker.patch(f"{MODULE}.CompareStructureObjectivesUseCase")
        use_case.return_value.execute.return_value = _comparison(_arm(BASELINE_ARM, -0.39), _arm(CHALLENGER_ARM, -0.45))
        renderer = mocker.patch(f"{MODULE}.MatplotlibFigureRenderer")
        mocker.patch("builtins.print")
        args = CompareObjectivesArgs(
            output_path=str(tmp_path / "structure_objective.md"),
            figure_path=str(tmp_path / "structure_objective.png"),
            committed_model_path=str(_committed_artifact(tmp_path)),
            selection_samples=2,
            structure_samples=4,
            **overrides,
        )
        return SimpleNamespace(use_case=use_case, renderer=renderer, args=args)

    def test_should_write_the_report_to_the_output_path(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        assert "Structure objective alignment" in (tmp_path / "structure_objective.md").read_text()

    def test_should_write_one_report_row_per_arm(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        report = (tmp_path / "structure_objective.md").read_text()
        assert report.count(f"| {BASELINE_ARM} |") == 1

    def test_should_record_the_adopted_arm_in_the_report(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        assert "Adopted arm" in (tmp_path / "structure_objective.md").read_text()

    def test_should_render_the_comparison_figure(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        assert context.renderer.return_value.render_objective_comparison.call_count == 1

    def test_should_forward_the_requested_arms_to_the_use_case(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, arms=[BASELINE_ARM])

        main(context.args)

        assert context.use_case.return_value.execute.call_args.kwargs["arms"] == [BASELINE_ARM]

    def test_should_nominate_downstream_arms_on_a_second_pass(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        assert context.use_case.return_value.execute.call_args.kwargs["downstream_arms"] == [
            BASELINE_ARM,
            CHALLENGER_ARM,
            COMMITTED_CONTROL,
        ]

    def test_should_pin_the_downstream_seeds_on_the_second_pass(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, downstream_seeds=[7, 8])

        main(context.args)

        assert context.use_case.return_value.execute.call_args.kwargs["downstream_seeds"] == [7, 8]

    def test_should_withhold_the_decision_log_on_the_nominating_pass(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        assert context.use_case.return_value.execute.call_args_list[0].kwargs["log_decision"] is False

    def test_should_derive_the_downstream_bit_budget_from_the_loaded_codebook(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        assert context.use_case.call_args.kwargs["downstream_bits_per_token"] == 3.0

    def test_should_run_a_single_pass_when_no_downstream_arm_is_nominated(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, downstream_top_k=0)

        main(context.args)

        assert context.use_case.return_value.execute.call_count == 1

    def test_should_still_log_the_decision_when_no_downstream_arm_is_nominated(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path, downstream_top_k=0)

        main(context.args)

        assert context.use_case.return_value.log_decision.call_count == 1

    def test_should_thread_one_correlation_id_through_both_passes(self, mocker, tmp_path) -> None:
        context = self._setup(mocker, tmp_path)

        main(context.args)

        ids = {call.kwargs["correlation_id"] for call in context.use_case.return_value.execute.call_args_list}
        assert len(ids) == 1


class TestDownstreamFactories:
    def _cache(self, mocker) -> TrainedMapperCache:
        mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        return TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))

    def test_should_build_a_classification_use_case_over_the_trained_mapper(self, mocker) -> None:
        factory = _build_evaluate_factory(
            self._cache(mocker),
            CompareObjectivesArgs(distance="jensen_shannon"),
            _config(),
            Mock(),
            ColorCodebook.create_uniform_grid(bins_per_dimension=2),
        )

        assert isinstance(factory(_request(BASELINE_ARM)), EvaluateUseCase)

    def test_should_build_a_retrieval_use_case_over_the_trained_mapper(self, mocker) -> None:
        factory = _build_retrieval_factory(
            self._cache(mocker),
            CompareObjectivesArgs(distance="jensen_shannon"),
            _config(),
            Mock(),
            ColorCodebook.create_uniform_grid(bins_per_dimension=2),
        )

        assert isinstance(factory(_request(BASELINE_ARM)), RetrievalEvaluateUseCase)


class TestCommittedControl:
    def test_should_read_the_committed_artifact_from_disk_instead_of_training(self, mocker) -> None:
        train_use_case = mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        load_weights = mocker.patch.object(PyTorchColorMapper, "load_weights")
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))

        cache.mapper(_request(COMMITTED_CONTROL))

        assert load_weights.call_count == 1 and train_use_case.call_count == 0

    def test_should_load_the_committed_artifact_from_the_configured_path(self, mocker) -> None:
        load_weights = mocker.patch.object(PyTorchColorMapper, "load_weights")
        args = CompareObjectivesArgs(committed_model_path="artifacts/models/other.pth")
        cache = TrainedMapperCache(args, _config(), _embeddings(4))

        cache.mapper(_request(COMMITTED_CONTROL))

        assert load_weights.call_args.args[0] == "artifacts/models/other.pth"

    def test_should_release_the_epoch_checkpoints_after_training_an_arm(self, mocker) -> None:
        train_use_case = mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        train_use_case.return_value.execute.side_effect = lambda **kwargs: _stash_checkpoint(train_use_case)
        cache = TrainedMapperCache(CompareObjectivesArgs(), _config(), _embeddings(4))

        mapper = cache.mapper(_request(BASELINE_ARM))

        assert mapper.epoch_checkpoints() == []


class TestRequestValidation:
    def test_should_reject_a_misspelled_arm_before_any_training(self) -> None:
        args = CompareObjectivesArgs(arms=[BASELINE_ARM, "delta_e_corelation"])

        with pytest.raises(ValueError, match="Unknown arm: delta_e_corelation"):
            _reject_unrunnable_request(args)

    def test_should_reject_a_misspelled_control_before_any_training(self) -> None:
        args = CompareObjectivesArgs(controls=["nosie"])

        with pytest.raises(ValueError, match="Unknown arm: nosie"):
            _reject_unrunnable_request(args)

    def test_should_reject_an_arm_list_without_the_baseline(self) -> None:
        args = CompareObjectivesArgs(arms=[CHALLENGER_ARM])

        with pytest.raises(ValueError, match="must include the baseline arm"):
            _reject_unrunnable_request(args)

    def test_should_reject_a_downstream_arm_that_is_not_backed_by_a_mapper(self) -> None:
        args = CompareObjectivesArgs(downstream_arms=[PCA_CONTROL])

        with pytest.raises(ValueError, match="Unknown downstream arm: pca3"):
            _reject_unrunnable_request(args)

    def test_should_reject_an_empty_seed_list(self) -> None:
        args = CompareObjectivesArgs(seeds=[])

        with pytest.raises(ValueError, match="seeds must name at least one value"):
            _reject_unrunnable_request(args)

    def test_should_reject_a_repeated_seed(self) -> None:
        args = CompareObjectivesArgs(seeds=[42, 42])

        with pytest.raises(ValueError, match="seeds must not repeat a value"):
            _reject_unrunnable_request(args)

    def test_should_reject_an_arm_that_is_also_listed_as_a_control(self) -> None:
        args = CompareObjectivesArgs(controls=[BASELINE_ARM, NOISE_CONTROL])

        with pytest.raises(ValueError, match="an arm cannot also be a control"):
            _reject_unrunnable_request(args)

    def test_should_reject_an_empty_k_value_list(self) -> None:
        args = CompareObjectivesArgs(k_values=[])

        with pytest.raises(ValueError, match="k-values must name at least one value"):
            _reject_unrunnable_request(args)

    def test_should_reject_explicit_downstream_arms_without_the_baseline(self) -> None:
        args = CompareObjectivesArgs(downstream_arms=["margin_ranking"])

        with pytest.raises(ValueError, match="downstream arms must include the baseline arm"):
            _reject_unrunnable_request(args)

    def test_should_reject_a_missing_committed_projector(self, tmp_path) -> None:
        args = CompareObjectivesArgs(committed_model_path=str(tmp_path / "absent.pth"))

        with pytest.raises(FileNotFoundError, match="Committed projector not found"):
            _reject_unrunnable_request(args)

    def test_should_skip_the_artifact_check_when_the_committed_control_is_not_requested(self, tmp_path) -> None:
        args = CompareObjectivesArgs(
            controls=[NOISE_CONTROL], downstream_controls=[], committed_model_path=str(tmp_path / "absent.pth")
        )

        assert _reject_unrunnable_request(args) is None

    def test_should_accept_the_default_request(self, tmp_path) -> None:
        args = CompareObjectivesArgs(committed_model_path=str(_committed_artifact(tmp_path)))

        assert _reject_unrunnable_request(args) is None


class TestBitsPerToken:
    def test_should_derive_the_bit_budget_from_the_codebook_size(self) -> None:
        assert _bits_per_token(ColorCodebook.create_uniform_grid(bins_per_dimension=16)) == 12.0

    def test_should_derive_a_smaller_bit_budget_from_a_smaller_codebook(self) -> None:
        assert _bits_per_token(ColorCodebook.create_uniform_grid(bins_per_dimension=4)) == 6.0
