from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from colors_of_meaning.interface.cli.eval import EvalArgs, main


class TestEvalCLI:
    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.eval.TFIDFClassifier")
    @patch("colors_of_meaning.interface.cli.eval.EvaluateUseCase")
    @patch("colors_of_meaning.interface.cli.eval.SklearnMetricsCalculator")
    @patch("builtins.print")
    def test_should_execute_evaluation_with_tfidf(
        self,
        mock_print: Mock,
        mock_metrics_class: Mock,
        mock_use_case_class: Mock,
        mock_classifier_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config

        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics

        mock_use_case = Mock()
        mock_result = Mock()
        mock_result.accuracy = 0.85
        mock_result.macro_f1 = 0.80
        mock_result.mrr = 0.75
        mock_result.recall_at_k = {}
        mock_result.bits_per_token = None
        mock_use_case.execute.return_value = mock_result
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        args = EvalArgs(config=str(config_path), dataset="ag_news", method="tfidf")

        main(args)

        mock_use_case.execute.assert_called_once()

    def test_should_create_eval_args_with_defaults(self) -> None:
        args = EvalArgs()

        assert args.dataset == "ag_news"
        assert args.method == "color"
        assert args.k_neighbors == 5

    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.IMDBDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.eval.TFIDFClassifier")
    @patch("colors_of_meaning.interface.cli.eval.EvaluateUseCase")
    @patch("colors_of_meaning.interface.cli.eval.SklearnMetricsCalculator")
    @patch("builtins.print")
    def test_should_print_results_with_recall_at_k(
        self,
        mock_print: Mock,
        mock_metrics_class: Mock,
        mock_use_case_class: Mock,
        mock_classifier_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config

        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics

        mock_use_case = Mock()
        mock_result = Mock()
        mock_result.accuracy = 0.85
        mock_result.macro_f1 = 0.80
        mock_result.mrr = 0.75
        mock_result.recall_at_k = {1: 0.7, 5: 0.9, 10: 0.95}
        mock_result.bits_per_token = None
        mock_use_case.execute.return_value = mock_result
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        args = EvalArgs(config=str(config_path), dataset="imdb", method="tfidf")

        main(args)

        # Verify recall@k was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Recall@1" in str(call) for call in print_calls)
        assert any("Recall@5" in str(call) for call in print_calls)
        assert any("Recall@10" in str(call) for call in print_calls)

    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.eval.ColorHistogramClassifier")
    @patch("colors_of_meaning.interface.cli.eval.EvaluateUseCase")
    @patch("colors_of_meaning.interface.cli.eval.SklearnMetricsCalculator")
    @patch("colors_of_meaning.interface.cli.eval.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.eval.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.eval.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.eval.WassersteinDistanceCalculator")
    @patch("colors_of_meaning.interface.cli.eval.EncodeDocumentUseCase")
    @patch("colors_of_meaning.domain.service.color_mapper.QuantizedColorMapper")
    @patch("builtins.print")
    def test_should_execute_evaluation_with_color_method(
        self,
        mock_print: Mock,
        mock_quantized_mapper_class: Mock,
        mock_encode_use_case_class: Mock,
        mock_distance_calc_class: Mock,
        mock_codebook_repo_class: Mock,
        mock_color_mapper_class: Mock,
        mock_embedding_adapter_class: Mock,
        mock_metrics_class: Mock,
        mock_use_case_class: Mock,
        mock_classifier_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 768
        mock_config.projector.hidden_dim_1 = 512
        mock_config.projector.hidden_dim_2 = 256
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config_class.from_yaml.return_value = mock_config

        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        mock_embedding_adapter = Mock()
        mock_embedding_adapter_class.return_value = mock_embedding_adapter

        mock_color_mapper = Mock()
        mock_color_mapper_class.return_value = mock_color_mapper

        mock_codebook = Mock()
        mock_codebook_repo = Mock()
        mock_codebook_repo.load.return_value = mock_codebook
        mock_codebook_repo_class.return_value = mock_codebook_repo

        mock_quantized_mapper = Mock()
        mock_quantized_mapper_class.return_value = mock_quantized_mapper

        mock_encode_use_case = Mock()
        mock_encode_use_case_class.return_value = mock_encode_use_case

        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics

        mock_use_case = Mock()
        mock_result = Mock()
        mock_result.accuracy = 0.90
        mock_result.macro_f1 = 0.88
        mock_result.mrr = 0.85
        mock_result.recall_at_k = {}
        mock_result.bits_per_token = 12.0
        mock_use_case.execute.return_value = mock_result
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        args = EvalArgs(config=str(config_path), dataset="ag_news", method="color")

        main(args)

        mock_use_case.execute.assert_called_once()
        # Verify bits_per_token was passed
        assert mock_use_case.execute.call_args[1]["bits_per_token"] == 12.0
        # Verify bits_per_token was printed
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Bits per token" in str(call) for call in print_calls)

    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.eval.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.eval.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.eval.FileColorCodebookRepository")
    @patch("colors_of_meaning.domain.service.color_mapper.QuantizedColorMapper")
    def test_should_raise_error_when_codebook_not_found(
        self,
        mock_quantized_mapper_class: Mock,
        mock_codebook_repo_class: Mock,
        mock_color_mapper_class: Mock,
        mock_embedding_adapter_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 768
        mock_config.projector.hidden_dim_1 = 512
        mock_config.projector.hidden_dim_2 = 256
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config_class.from_yaml.return_value = mock_config

        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        mock_embedding_adapter = Mock()
        mock_embedding_adapter_class.return_value = mock_embedding_adapter

        mock_color_mapper = Mock()
        mock_color_mapper_class.return_value = mock_color_mapper

        # Mock codebook repository to return None
        mock_codebook_repo = Mock()
        mock_codebook_repo.load.return_value = None
        mock_codebook_repo_class.return_value = mock_codebook_repo

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        args = EvalArgs(config=str(config_path), dataset="ag_news", method="color")

        with pytest.raises(FileNotFoundError, match="Codebook not found"):
            main(args)

    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    def test_should_raise_error_for_unknown_method(
        self,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config

        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        # Create args with invalid method by bypassing the Literal type
        args = EvalArgs(config=str(config_path), dataset="ag_news")
        args.method = "invalid_method"  # type: ignore

        with pytest.raises(ValueError, match="Unknown method: invalid_method"):
            main(args)

    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.eval.HNSWClassifier")
    @patch("colors_of_meaning.interface.cli.eval.EvaluateUseCase")
    @patch("colors_of_meaning.interface.cli.eval.SklearnMetricsCalculator")
    @patch("colors_of_meaning.interface.cli.eval.SentenceEmbeddingAdapter")
    @patch("builtins.print")
    def test_should_execute_evaluation_with_hnsw_method(
        self,
        mock_print: Mock,
        mock_embedding_adapter_class: Mock,
        mock_metrics_class: Mock,
        mock_use_case_class: Mock,
        mock_classifier_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config

        mock_dataset = Mock()
        mock_dataset_class.return_value = mock_dataset

        mock_embedding_adapter = Mock()
        mock_embedding_adapter_class.return_value = mock_embedding_adapter

        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        mock_metrics = Mock()
        mock_metrics_class.return_value = mock_metrics

        mock_use_case = Mock()
        mock_result = Mock()
        mock_result.accuracy = 0.87
        mock_result.macro_f1 = 0.84
        mock_result.mrr = 0.80
        mock_result.recall_at_k = {}
        mock_result.bits_per_token = None
        mock_use_case.execute.return_value = mock_result
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        args = EvalArgs(config=str(config_path), dataset="ag_news", method="hnsw", k_neighbors=3)

        main(args)

        mock_use_case.execute.assert_called_once()
        # Verify HNSW was used
        mock_classifier_class.assert_called_once()
        assert mock_classifier_class.call_args[1]["k"] == 3
