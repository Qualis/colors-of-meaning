from unittest.mock import Mock

from colors_of_meaning.infrastructure.ml.authorship_scaling_sweep import (
    AuthorshipScalingSweep,
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


class TestRun:
    def test_should_produce_one_point_per_cap(self, mocker) -> None:
        sweep = _build_sweep(caps=[60, 150])
        mocker.patch.object(sweep, "_point_for_cap", side_effect=lambda cap: cap)

        assert sweep.run() == [60, 150]


class TestPointForCap:
    def test_should_average_accuracy_across_seeds(self, mocker) -> None:
        sweep = _build_sweep(seeds=2)
        mocker.patch.object(sweep, "_train_and_eval", side_effect=[(0.1, 100), (0.3, 100)])

        point = sweep._point_for_cap(60)

        assert point.mean_accuracy == 0.2

    def test_should_record_the_train_paragraph_count(self, mocker) -> None:
        sweep = _build_sweep(seeds=2)
        mocker.patch.object(sweep, "_train_and_eval", side_effect=[(0.1, 5340), (0.3, 5340)])

        point = sweep._point_for_cap(60)

        assert point.train_paragraphs == 5340


class TestTrainAndEval:
    def test_should_return_the_held_out_accuracy_and_train_count(self, mocker) -> None:
        sweep = _build_sweep()
        mocker.patch.object(sweep, "_train_projector", return_value=(Mock(), Mock(), 5340))
        mocker.patch.object(sweep, "_held_out_accuracy", return_value=0.15)

        assert sweep._train_and_eval(60, 0) == (0.15, 5340)


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
