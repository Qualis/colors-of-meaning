from typing import Optional

from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.service.metrics_calculator import MetricsCalculator
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


class EvaluateUseCase:
    def __init__(
        self,
        classifier: Classifier,
        metrics_calculator: MetricsCalculator,
        dataset_repository: DatasetRepository,
    ) -> None:
        self.classifier = classifier
        self.metrics_calculator = metrics_calculator
        self.dataset_repository = dataset_repository

    def execute(self, bits_per_token: Optional[float] = None, max_samples: Optional[int] = None) -> EvaluationResult:
        train_samples = self.dataset_repository.get_samples(split="train", max_samples=max_samples)
        test_samples = self.dataset_repository.get_samples(split="test", max_samples=max_samples)

        self.classifier.fit(train_samples)

        y_true = [sample.label for sample in test_samples]
        y_pred = self.classifier.predict(test_samples)

        result = self.metrics_calculator.calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            bits_per_token=bits_per_token,
        )

        return result
