import logging
import time
import uuid
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
import torch

from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
)
from colors_of_meaning.domain.model.authorship_report import AuthorshipScalingPoint
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
)
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator import (
    SpearmanStructurePreservationEvaluator,
)
from colors_of_meaning.infrastructure.evaluation.validation_accuracy_checkpoint_selector import (
    ValidationAccuracyCheckpointSelector,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.shared.determinism import seed_everything
from colors_of_meaning.shared.synesthetic_config import (
    SupervisedMapperConfig,
    SynestheticConfig,
)

logger = logging.getLogger(__name__)

TrainingJob = Tuple[int, int]
TrainingOutcome = Tuple[float, int]
ExecutorFactory = Callable[[int], Executor]


def pin_worker_torch_threads() -> None:
    torch.set_num_threads(1)


def create_scaling_process_pool(workers: int) -> Executor:
    return ProcessPoolExecutor(max_workers=workers, initializer=pin_worker_torch_threads)


class AuthorshipScalingSweep:
    def __init__(
        self,
        config: SynestheticConfig,
        reference_adapter: DocumentCorpusDatasetAdapter,
        embedding_adapter: SentenceEmbeddingAdapter,
        documents_dir: str,
        min_paragraph_chars: int,
        split_strategy: str,
        validation_fraction: float,
        test_fraction: float,
        distance_smoothing: float,
        k_neighbors: int,
        caps: List[int],
        seeds: int,
        scratch_dir: str,
        scaling_workers: int = 1,
        executor_factory: Optional[ExecutorFactory] = None,
    ) -> None:
        self.config = config
        self.reference_adapter = reference_adapter
        self.embedding_adapter = embedding_adapter
        self.documents_dir = documents_dir
        self.min_paragraph_chars = min_paragraph_chars
        self.split_strategy = split_strategy
        self.validation_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.distance_smoothing = distance_smoothing
        self.k_neighbors = k_neighbors
        self.caps = caps
        self.seeds = seeds
        self.scratch_dir = scratch_dir
        self.scaling_workers = scaling_workers
        self.executor_factory = executor_factory or create_scaling_process_pool

    def run(self) -> List[AuthorshipScalingPoint]:
        outcomes = self._execute_grid()
        return [self._point_for_cap(cap, outcomes) for cap in self.caps]

    def _grid(self) -> List[TrainingJob]:
        return [(cap, seed) for cap in self.caps for seed in range(self.seeds)]

    def _execute_grid(self) -> Dict[TrainingJob, TrainingOutcome]:
        if self.scaling_workers <= 1:
            return {job: self._train_and_eval(*job) for job in self._grid()}
        return self._execute_grid_in_parallel()

    def _execute_grid_in_parallel(self) -> Dict[TrainingJob, TrainingOutcome]:
        jobs = self._grid()
        with self.executor_factory(self.scaling_workers) as executor:
            submitted = [executor.submit(self._train_and_eval, cap, seed) for cap, seed in jobs]
            return {job: future.result() for job, future in zip(jobs, submitted, strict=True)}

    def _point_for_cap(self, cap: int, outcomes: Dict[TrainingJob, TrainingOutcome]) -> AuthorshipScalingPoint:
        ordered = [outcomes[(cap, seed)] for seed in range(self.seeds)]
        accuracies = [accuracy for accuracy, _count in ordered]
        return AuthorshipScalingPoint(
            paragraphs_per_work=cap,
            train_paragraphs=ordered[0][1],
            mean_accuracy=float(np.mean(accuracies)),
            std_accuracy=float(np.std(accuracies)),
        )

    def _train_and_eval(self, cap: int, seed: int) -> TrainingOutcome:
        started_at = time.perf_counter()
        mapper, codebook, train_count = self._train_projector(cap, seed)
        accuracy = self._held_out_accuracy(mapper, codebook)
        self._log_completed_training(cap, seed, train_count, accuracy, time.perf_counter() - started_at)
        return accuracy, train_count

    @staticmethod
    def _log_completed_training(cap: int, seed: int, train_count: int, accuracy: float, elapsed: float) -> None:
        logger.info(
            "Completed scaling sweep training",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "paragraphs_per_work": cap,
                "seed": seed,
                "train_paragraphs": train_count,
                "accuracy": accuracy,
                "elapsed_seconds": round(elapsed, 1),
            },
        )

    def _train_projector(self, cap: int, seed: int) -> Tuple[ColorMapper, ColorCodebook, int]:
        adapter = self._corpus_at_cap(cap)
        train_texts, train_labels = self._texts_labels(adapter, "train", seed)
        seed_everything(seed)
        train_embeddings = self.embedding_adapter.encode_batch(train_texts, batch_size=self.config.training.batch_size)
        mapper = self._new_mapper(seed)
        mapper.set_training_labels(train_labels)
        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=self.config.codebook.bins_per_dimension)
        selector = self._selector(adapter, mapper, codebook, train_embeddings, train_labels, seed)
        self._run_training(mapper, codebook, train_embeddings, selector, cap, seed)
        return mapper, codebook, len(train_texts)

    def _corpus_at_cap(self, cap: int) -> DocumentCorpusDatasetAdapter:
        return DocumentCorpusDatasetAdapter(
            documents_dir=self.documents_dir,
            min_paragraph_chars=self.min_paragraph_chars,
            paragraphs_per_work=cap,
            split_strategy=self.split_strategy,
            validation_fraction=self.validation_fraction,
            test_fraction=self.test_fraction,
        )

    def _texts_labels(
        self, adapter: DocumentCorpusDatasetAdapter, split: str, seed: int
    ) -> Tuple[List[str], npt.NDArray]:
        samples = adapter.get_samples(split=split, max_samples=self.config.dataset.max_samples, seed=seed)
        return [sample.text for sample in samples], np.array([sample.label for sample in samples])

    def _new_mapper(self, seed: int) -> SupervisedPyTorchColorMapper:
        supervised = cast(SupervisedMapperConfig, self.config.supervised_mapper)
        return SupervisedPyTorchColorMapper(
            input_dim=self.config.projector.embedding_dim,
            hidden_dim_1=self.config.projector.hidden_dim_1,
            hidden_dim_2=self.config.projector.hidden_dim_2,
            dropout_rate=self.config.projector.dropout_rate,
            device=self.config.training.device,
            num_classes=supervised.num_classes,
            classification_weight=supervised.classification_weight,
            seed=seed,
        )

    def _selector(
        self,
        adapter: DocumentCorpusDatasetAdapter,
        mapper: ColorMapper,
        codebook: ColorCodebook,
        train_embeddings: npt.NDArray,
        train_labels: npt.NDArray,
        seed: int,
    ) -> ValidationAccuracyCheckpointSelector:
        validation_texts, validation_labels = self._texts_labels(adapter, "validation", seed)
        validation_embeddings = self.embedding_adapter.encode_batch(
            validation_texts, batch_size=self.config.training.batch_size
        )
        return ValidationAccuracyCheckpointSelector(
            encode_use_case=EncodeDocumentUseCase(QuantizedColorMapper(mapper, codebook)),
            distance_calculator=JensenShannonDistanceCalculator(smoothing_epsilon=self.distance_smoothing),
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            validation_embeddings=validation_embeddings,
            validation_labels=validation_labels,
            k=self.k_neighbors,
        )

    def _run_training(
        self,
        mapper: ColorMapper,
        codebook: ColorCodebook,
        train_embeddings: npt.NDArray,
        selector: ValidationAccuracyCheckpointSelector,
        cap: int,
        seed: int,
    ) -> None:
        use_case = TrainColorMappingUseCase(
            color_mapper=mapper,
            structure_preservation_evaluator=SpearmanStructurePreservationEvaluator(seed=seed),
            codebook_repository=FileColorCodebookRepository(base_path=self.scratch_dir),
            codebook_factory=None,
            checkpoint_selector=selector,
        )
        use_case.execute(
            embeddings=train_embeddings,
            evaluation_embeddings=train_embeddings,
            epochs=self.config.training.epochs,
            learning_rate=self.config.training.learning_rate,
            bins_per_dimension=self.config.codebook.bins_per_dimension,
            model_name=str(Path(self.scratch_dir) / f"projector_{cap}_{seed}.pth"),
            codebook_name=f"codebook_{cap}_{seed}",
            codebook_mode="uniform",
            num_bins=self.config.codebook.num_bins,
            seed=seed,
        )

    def _held_out_accuracy(self, mapper: ColorMapper, codebook: ColorCodebook) -> float:
        classifier = ColorHistogramClassifier(
            self.embedding_adapter,
            EncodeDocumentUseCase(QuantizedColorMapper(mapper, codebook)),
            JensenShannonDistanceCalculator(smoothing_epsilon=self.distance_smoothing),
            k=self.k_neighbors,
        )
        use_case = EvaluateUseCase(classifier, SklearnMetricsCalculator(), self.reference_adapter)
        result = use_case.execute(bits_per_token=None, max_samples=None, seed=self.config.training.seed)
        return result.accuracy
