from concurrent.futures import Executor, ThreadPoolExecutor
from unittest.mock import Mock

from colors_of_meaning.infrastructure.ml.authorship_scaling_sweep import (
    AuthorshipScalingSweep,
    create_scaling_process_pool,
    pin_worker_torch_threads,
)
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

MODULE = "colors_of_meaning.infrastructure.ml.authorship_scaling_sweep"


def _config() -> SynestheticConfig:
    return SynestheticConfig.from_yaml("configs/documents.yaml")


def _build_sweep(**overrides) -> AuthorshipScalingSweep:
    defaults = dict(
        config=_config(),
        reference_adapter=Mock(),
        embedding_adapter=Mock(),
        documents_dir="./documents",
        min_paragraph_chars=200,
        split_strategy="work",
        validation_fraction=0.2,
        test_fraction=0.2,
        distance_smoothing=1e-8,
        k_neighbors=5,
        caps=[60],
        seeds=1,
        scratch_dir="/tmp/scaling_scratch",  # nosec B108
    )
    defaults.update(overrides)
    return AuthorshipScalingSweep(**defaults)


def _thread_pool_factory(workers: int) -> Executor:
    return ThreadPoolExecutor(max_workers=workers)


def _deterministic_outcome(cap: int, seed: int) -> tuple:
    return (0.1 * (seed + 1) + cap / 1000.0, cap * 10)


class TestRun:
    def test_should_produce_one_point_per_cap(self, mocker) -> None:
        sweep = _build_sweep(caps=[60, 150])
        mocker.patch.object(sweep, "_train_and_eval", side_effect=_deterministic_outcome)

        assert [point.paragraphs_per_work for point in sweep.run()] == [60, 150]

    def test_should_produce_identical_points_when_run_across_two_workers(self, mocker) -> None:
        serial = _build_sweep(caps=[60, 150], seeds=3)
        mocker.patch.object(serial, "_train_and_eval", side_effect=_deterministic_outcome)
        parallel = _build_sweep(caps=[60, 150], seeds=3, scaling_workers=2, executor_factory=_thread_pool_factory)
        mocker.patch.object(parallel, "_train_and_eval", side_effect=_deterministic_outcome)

        assert parallel.run() == serial.run()


class TestPointForCap:
    def test_should_average_accuracy_across_seeds(self) -> None:
        sweep = _build_sweep(seeds=2)

        point = sweep._point_for_cap(60, {(60, 0): (0.1, 100), (60, 1): (0.3, 100)})

        assert point.mean_accuracy == 0.2

    def test_should_record_the_train_paragraph_count(self) -> None:
        sweep = _build_sweep(seeds=2)

        point = sweep._point_for_cap(60, {(60, 0): (0.1, 5340), (60, 1): (0.3, 5340)})

        assert point.train_paragraphs == 5340


class TestExecuteGrid:
    def test_should_train_every_cap_and_seed_combination_when_single_threaded(self, mocker) -> None:
        sweep = _build_sweep(caps=[60, 150], seeds=2)
        mocker.patch.object(sweep, "_train_and_eval", side_effect=_deterministic_outcome)

        assert sorted(sweep._execute_grid()) == [(60, 0), (60, 1), (150, 0), (150, 1)]

    def test_should_use_the_executor_factory_when_more_than_one_worker_is_requested(self, mocker) -> None:
        factory = mocker.Mock(side_effect=_thread_pool_factory)
        sweep = _build_sweep(caps=[60], seeds=2, scaling_workers=4, executor_factory=factory)
        mocker.patch.object(sweep, "_train_and_eval", side_effect=_deterministic_outcome)

        sweep._execute_grid()

        factory.assert_called_once_with(4)

    def test_should_key_parallel_outcomes_by_cap_and_seed(self, mocker) -> None:
        sweep = _build_sweep(caps=[60, 150], seeds=2, scaling_workers=2, executor_factory=_thread_pool_factory)
        mocker.patch.object(sweep, "_train_and_eval", side_effect=_deterministic_outcome)

        assert sweep._execute_grid()[(150, 1)] == _deterministic_outcome(150, 1)


class TestWorkerConfiguration:
    def test_should_pin_each_worker_to_a_single_torch_thread(self, mocker) -> None:
        torch_module = mocker.patch(f"{MODULE}.torch")

        pin_worker_torch_threads()

        torch_module.set_num_threads.assert_called_once_with(1)

    def test_should_create_a_process_pool_sized_to_the_requested_workers(self, mocker) -> None:
        pool_class = mocker.patch(f"{MODULE}.ProcessPoolExecutor")

        create_scaling_process_pool(3)

        assert pool_class.call_args.kwargs["max_workers"] == 3

    def test_should_pin_torch_threads_from_the_process_pool_initializer(self, mocker) -> None:
        pool_class = mocker.patch(f"{MODULE}.ProcessPoolExecutor")

        create_scaling_process_pool(3)

        assert pool_class.call_args.kwargs["initializer"] is pin_worker_torch_threads


class TestTrainAndEval:
    def test_should_return_the_held_out_accuracy_and_train_count(self, mocker) -> None:
        sweep = _build_sweep()
        mocker.patch.object(sweep, "_train_projector", return_value=(Mock(), Mock(), 5340))
        mocker.patch.object(sweep, "_held_out_accuracy", return_value=0.15)

        assert sweep._train_and_eval(60, 0) == (0.15, 5340)

    def test_should_log_the_completed_training_with_its_cap_and_seed(self, mocker, caplog) -> None:
        sweep = _build_sweep()
        mocker.patch.object(sweep, "_train_projector", return_value=(Mock(), Mock(), 5340))
        mocker.patch.object(sweep, "_held_out_accuracy", return_value=0.15)

        with caplog.at_level("INFO"):
            sweep._train_and_eval(60, 3)

        assert [record.seed for record in caplog.records] == [3]


class TestTrainProjector:
    def _patch_collaborators(self, mocker) -> None:
        adapter = mocker.patch(f"{MODULE}.DocumentCorpusDatasetAdapter").return_value
        adapter.get_samples.return_value = [Mock(text="paragraph one", label=0), Mock(text="paragraph two", label=1)]
        for name in (
            "SupervisedPyTorchColorMapper",
            "ValidationAccuracyCheckpointSelector",
            "TrainColorMappingUseCase",
            "ColorCodebook",
            "seed_everything",
            "EncodeDocumentUseCase",
            "QuantizedColorMapper",
            "SpearmanStructurePreservationEvaluator",
            "FileColorCodebookRepository",
            "JensenShannonDistanceCalculator",
        ):
            mocker.patch(f"{MODULE}.{name}")

    def test_should_return_the_train_paragraph_count(self, mocker) -> None:
        self._patch_collaborators(mocker)
        sweep = _build_sweep()

        _mapper, _codebook, train_count = sweep._train_projector(60, 0)

        assert train_count == 2

    def test_should_set_the_training_labels_on_the_mapper(self, mocker) -> None:
        self._patch_collaborators(mocker)
        mapper_class = mocker.patch(f"{MODULE}.SupervisedPyTorchColorMapper")
        sweep = _build_sweep()

        sweep._train_projector(60, 0)

        mapper_class.return_value.set_training_labels.assert_called_once()

    def test_should_run_the_training_use_case(self, mocker) -> None:
        self._patch_collaborators(mocker)
        train_use_case = mocker.patch(f"{MODULE}.TrainColorMappingUseCase")
        sweep = _build_sweep()

        sweep._train_projector(60, 0)

        train_use_case.return_value.execute.assert_called_once()


class TestHeldOutAccuracy:
    def test_should_return_the_evaluated_accuracy(self, mocker) -> None:
        for name in (
            "ColorHistogramClassifier",
            "EncodeDocumentUseCase",
            "QuantizedColorMapper",
            "JensenShannonDistanceCalculator",
            "SklearnMetricsCalculator",
        ):
            mocker.patch(f"{MODULE}.{name}")
        evaluate = mocker.patch(f"{MODULE}.EvaluateUseCase")
        evaluate.return_value.execute.return_value = Mock(accuracy=0.42)
        sweep = _build_sweep()

        assert sweep._held_out_accuracy(Mock(), Mock()) == 0.42
