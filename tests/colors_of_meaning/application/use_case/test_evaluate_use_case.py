import pytest
from unittest.mock import Mock

from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult


class TestEvaluateUseCase:
    @pytest.fixture
    def mock_classifier(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_metrics_calculator(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_dataset_repository(self) -> Mock:
        return Mock()

    @pytest.fixture
    def use_case(
        self,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
        mock_dataset_repository: Mock,
    ) -> EvaluateUseCase:
        return EvaluateUseCase(
            classifier=mock_classifier,
            metrics_calculator=mock_metrics_calculator,
            dataset_repository=mock_dataset_repository,
        )

    def test_should_load_train_dataset_from_repository(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        mock_metrics_calculator.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0
        )

        use_case.execute()

        mock_dataset_repository.get_samples.assert_any_call(split="train", max_samples=None)

    def test_should_load_test_dataset_from_repository(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        mock_metrics_calculator.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0
        )

        use_case.execute()

        mock_dataset_repository.get_samples.assert_any_call(split="test", max_samples=None)

    def test_should_fit_classifier_with_train_samples(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        mock_metrics_calculator.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0
        )

        use_case.execute()

        mock_classifier.fit.assert_called_once_with(train_samples)

    def test_should_predict_on_test_samples(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        mock_metrics_calculator.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0
        )

        use_case.execute()

        mock_classifier.predict.assert_called_once_with(test_samples)

    def test_should_calculate_metrics_with_true_and_predicted_labels(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=1, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [1]
        mock_metrics_calculator.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0
        )

        use_case.execute()

        mock_metrics_calculator.calculate_classification_metrics.assert_called_once_with(
            y_true=[1], y_pred=[1], bits_per_token=None
        )

    def test_should_return_evaluation_result(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        expected_result = EvaluationResult(accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0)
        mock_metrics_calculator.calculate_classification_metrics.return_value = expected_result

        result = use_case.execute()

        assert result == expected_result

    def test_should_pass_bits_per_token_to_metrics_calculator(
        self,
        use_case: EvaluateUseCase,
        mock_dataset_repository: Mock,
        mock_classifier: Mock,
        mock_metrics_calculator: Mock,
    ) -> None:
        train_samples = [EvaluationSample(text="train", label=0, split="train")]
        test_samples = [EvaluationSample(text="test", label=0, split="test")]
        mock_dataset_repository.get_samples.side_effect = [train_samples, test_samples]
        mock_classifier.predict.return_value = [0]
        mock_metrics_calculator.calculate_classification_metrics.return_value = EvaluationResult(
            accuracy=0.9, macro_f1=0.85, recall_at_k={}, mrr=0.0, bits_per_token=12.0
        )

        use_case.execute(bits_per_token=12.0)

        mock_metrics_calculator.calculate_classification_metrics.assert_called_once_with(
            y_true=[0], y_pred=[0], bits_per_token=12.0
        )
