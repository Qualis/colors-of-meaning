from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from colors_of_meaning.interface.cli.eval import (
    DEFAULT_SINKHORN_REG,
    EvalArgs,
    main,
    _build_dataset_repository,
    _build_document_corpus,
    _create_color_classifier,
    _create_distance_calculator,
    _create_retriever,
    _print_retrieval_results,
    _resolve_max_samples,
    _resolve_paragraphs_per_work,
    _resolved_sinkhorn_reg,
)
from colors_of_meaning.domain.model.evaluation_result import EvaluationResult
from colors_of_meaning.domain.model.retrieval_evaluation import RetrievalEvaluation


def _retrieval_evaluation(bits_per_token=None) -> RetrievalEvaluation:
    result = EvaluationResult(
        accuracy=0.0, macro_f1=0.0, recall_at_k={1: 0.7, 5: 0.9}, mrr=0.65, bits_per_token=bits_per_token
    )
    return RetrievalEvaluation(result=result, ndcg_at_k={1: 0.6, 5: 0.8})


class TestEvalDocumentsSource:
    @patch("builtins.print")
    @patch("colors_of_meaning.interface.cli.eval.DocumentCorpusDatasetAdapter")
    def test_should_build_document_corpus_when_source_is_documents(self, mock_adapter: Mock, _print: Mock) -> None:
        result = _build_dataset_repository(EvalArgs(source="documents"), Mock())

        assert result is mock_adapter.return_value

    @patch("builtins.print")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    def test_should_build_hugging_face_adapter_when_source_is_dataset(self, mock_agnews: Mock, _print: Mock) -> None:
        result = _build_dataset_repository(EvalArgs(source="dataset", dataset="ag_news"), Mock())

        assert result is mock_agnews.return_value

    @patch("builtins.print")
    @patch("colors_of_meaning.interface.cli.eval.DocumentCorpusDatasetAdapter")
    def test_should_pass_documents_dir_to_the_corpus_adapter(self, mock_adapter: Mock, _print: Mock) -> None:
        _build_document_corpus(EvalArgs(source="documents", documents_dir="docs"), Mock())

        assert mock_adapter.call_args.kwargs["documents_dir"] == "docs"


def _run_color_classifier_with_mocked_factory(mapper_type: str) -> tuple:
    config = Mock()
    args = EvalArgs(mapper_type=mapper_type)
    with ExitStack() as stack:
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.SentenceEmbeddingAdapter"))
        mock_create_mapper = stack.enter_context(patch("colors_of_meaning.interface.cli.eval.create_color_mapper"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.load_codebook"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.WassersteinDistanceCalculator"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.EncodeDocumentUseCase"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.ColorHistogramClassifier"))
        stack.enter_context(patch("colors_of_meaning.domain.service.color_mapper.QuantizedColorMapper"))
        stack.enter_context(patch("builtins.print"))
        mock_mapper = Mock()
        mock_create_mapper.return_value = mock_mapper
        _create_color_classifier(args, config)
    return mock_create_mapper, mock_mapper


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

    @patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.eval.TFIDFClassifier")
    @patch("colors_of_meaning.interface.cli.eval.EvaluateUseCase")
    @patch("colors_of_meaning.interface.cli.eval.SklearnMetricsCalculator")
    @patch("builtins.print")
    def test_should_forward_configured_seed_to_use_case(
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
        mock_config.training.seed = 99
        mock_config_class.from_yaml.return_value = mock_config

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

        assert mock_use_case.execute.call_args[1]["seed"] == 99

    def test_should_create_eval_args_with_defaults(self) -> None:
        args = EvalArgs()

        assert args.dataset == "ag_news"
        assert args.method == "color"
        assert args.k_neighbors == 5

    def test_should_default_mapper_type_to_unconstrained(self) -> None:
        args = EvalArgs()

        assert args.mapper_type == "unconstrained"

    def test_should_route_through_factory_when_mapper_type_is_structured(self) -> None:
        mock_create_mapper, _ = _run_color_classifier_with_mocked_factory("structured")

        assert mock_create_mapper.call_args[0][0] == "structured"

    def test_should_load_weights_when_mapper_type_is_structured(self) -> None:
        _, mock_mapper = _run_color_classifier_with_mocked_factory("structured")

        mock_mapper.load_weights.assert_called_once()

    def test_should_route_through_factory_when_mapper_type_is_supervised(self) -> None:
        mock_create_mapper, _ = _run_color_classifier_with_mocked_factory("supervised")

        assert mock_create_mapper.call_args[0][0] == "supervised"

    def test_should_load_weights_when_mapper_type_is_supervised(self) -> None:
        _, mock_mapper = _run_color_classifier_with_mocked_factory("supervised")

        mock_mapper.load_weights.assert_called_once()

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
    @patch("colors_of_meaning.interface.cli.eval.create_color_mapper")
    @patch("colors_of_meaning.interface.cli.eval.load_codebook")
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
        mock_load_codebook: Mock,
        mock_create_color_mapper: Mock,
        mock_embedding_adapter_class: Mock,
        mock_metrics_class: Mock,
        mock_use_case_class: Mock,
        mock_classifier_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 384
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
        mock_create_color_mapper.return_value = mock_color_mapper

        mock_codebook = Mock()
        mock_load_codebook.return_value = mock_codebook

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
        mock_distance_calc_class.assert_called_once_with(
            codebook=mock_codebook, sinkhorn_reg=mock_config.distance.sinkhorn_reg
        )
        assert mock_use_case.execute.call_args[1]["bits_per_token"] == 12.0
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any("Bits per token" in str(call) for call in print_calls)

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
        mock_classifier_class.assert_called_once()
        assert mock_classifier_class.call_args[1]["k"] == 3


class TestEvalDistanceSelection:
    def test_should_default_distance_to_wasserstein(self) -> None:
        assert EvalArgs().distance == "wasserstein"

    def test_should_create_exact_wasserstein_calculator_when_distance_is_wasserstein(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval.WassersteinDistanceCalculator") as calculator_class:
            calculator = _create_distance_calculator("wasserstein", Mock(), Mock())

        assert calculator is calculator_class.return_value

    def test_should_create_sliced_calculator_when_distance_is_sliced(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval.SlicedWassersteinDistanceCalculator") as calculator_class:
            calculator = _create_distance_calculator("sliced", Mock(), Mock())

        assert calculator is calculator_class.return_value

    def test_should_create_jensen_shannon_calculator_when_distance_is_jensen_shannon(self) -> None:
        with patch("colors_of_meaning.interface.cli.eval.JensenShannonDistanceCalculator") as calculator_class:
            calculator = _create_distance_calculator("jensen_shannon", Mock(), Mock())

        assert calculator is calculator_class.return_value

    def test_should_use_entropic_regularisation_when_distance_is_sinkhorn(self) -> None:
        config = Mock()
        config.distance.sinkhorn_reg = None
        with patch("colors_of_meaning.interface.cli.eval.WassersteinDistanceCalculator") as calculator_class:
            _create_distance_calculator("sinkhorn", Mock(), config)

        assert calculator_class.call_args[1]["sinkhorn_reg"] == DEFAULT_SINKHORN_REG

    def test_should_raise_value_error_when_distance_is_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unknown distance"):
            _create_distance_calculator("unknown", Mock(), Mock())

    def test_should_default_sinkhorn_reg_when_config_value_is_absent(self) -> None:
        config = Mock()
        config.distance.sinkhorn_reg = None

        assert _resolved_sinkhorn_reg(config) == DEFAULT_SINKHORN_REG

    def test_should_keep_config_sinkhorn_reg_when_present(self) -> None:
        config = Mock()
        config.distance.sinkhorn_reg = 0.25

        assert _resolved_sinkhorn_reg(config) == 0.25

    def test_should_override_max_samples_when_argument_is_provided(self) -> None:
        config = Mock()
        config.dataset.max_samples = 400

        assert _resolve_max_samples(EvalArgs(max_samples=2000), config) == 2000

    def test_should_fall_back_to_config_max_samples_when_not_overridden(self) -> None:
        config = Mock()
        config.dataset.max_samples = 400

        assert _resolve_max_samples(EvalArgs(), config) == 400

    def test_should_override_paragraphs_per_work_when_argument_is_provided(self) -> None:
        config = Mock()
        config.dataset.paragraphs_per_work = 60

        assert _resolve_paragraphs_per_work(EvalArgs(paragraphs_per_work=300), config) == 300

    def test_should_fall_back_to_config_paragraphs_per_work_when_not_overridden(self) -> None:
        config = Mock()
        config.dataset.paragraphs_per_work = 300

        assert _resolve_paragraphs_per_work(EvalArgs(), config) == 300


def _create_color_retriever_with_mocked_factory() -> tuple:
    with ExitStack() as stack:
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.SentenceEmbeddingAdapter"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.create_color_mapper"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.load_codebook"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.WassersteinDistanceCalculator"))
        stack.enter_context(patch("colors_of_meaning.interface.cli.eval.EncodeDocumentUseCase"))
        retriever_class = stack.enter_context(patch("colors_of_meaning.interface.cli.eval.ColorHistogramRetriever"))
        stack.enter_context(patch("colors_of_meaning.domain.service.color_mapper.QuantizedColorMapper"))
        stack.enter_context(patch("builtins.print"))
        retriever, bits = _create_retriever(EvalArgs(method="color"), Mock())
    return retriever, bits, retriever_class


class TestEvalRetrieverSelection:
    def test_should_default_task_to_classification(self) -> None:
        assert EvalArgs().task == "classification"

    def test_should_default_k_values(self) -> None:
        assert EvalArgs().k_values == [1, 5, 10]

    def test_should_build_color_retriever_when_method_is_color(self) -> None:
        retriever, _, retriever_class = _create_color_retriever_with_mocked_factory()

        assert retriever is retriever_class.return_value

    def test_should_report_color_retriever_bits_per_token(self) -> None:
        _, bits, _ = _create_color_retriever_with_mocked_factory()

        assert bits == 12.0

    def test_should_build_embedding_retriever_when_method_is_hnsw(self) -> None:
        with ExitStack() as stack:
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval.SentenceEmbeddingAdapter"))
            retriever_class = stack.enter_context(patch("colors_of_meaning.interface.cli.eval.EmbeddingRetriever"))
            stack.enter_context(patch("builtins.print"))
            retriever, _ = _create_retriever(EvalArgs(method="hnsw"), Mock())

        assert retriever is retriever_class.return_value

    def test_should_raise_when_retrieval_requested_for_tfidf(self) -> None:
        with pytest.raises(ValueError, match="Retrieval not supported"):
            _create_retriever(EvalArgs(method="tfidf"), Mock())

    def test_should_raise_for_unknown_method_when_building_retriever(self) -> None:
        args = EvalArgs()
        args.method = "weird"  # type: ignore

        with pytest.raises(ValueError, match="Unknown method"):
            _create_retriever(args, Mock())


class TestRetrievalPrinting:
    def test_should_caption_retrieval_output_as_label_based(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_retrieval_results(EvalArgs(method="color"), _retrieval_evaluation())

        assert any("Label-based retrieval" in str(call) for call in print_mock.call_args_list)

    def test_should_print_mrr_in_retrieval_output(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_retrieval_results(EvalArgs(method="color"), _retrieval_evaluation())

        assert any("MRR" in str(call) for call in print_mock.call_args_list)

    def test_should_print_recall_at_k_in_retrieval_output(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_retrieval_results(EvalArgs(method="color"), _retrieval_evaluation())

        assert any("Recall@1" in str(call) for call in print_mock.call_args_list)

    def test_should_print_ndcg_in_retrieval_output(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_retrieval_results(EvalArgs(method="color"), _retrieval_evaluation())

        assert any("NDCG@1" in str(call) for call in print_mock.call_args_list)

    def test_should_print_bits_per_token_in_retrieval_output(self) -> None:
        with patch("builtins.print") as print_mock:
            _print_retrieval_results(EvalArgs(method="hnsw"), _retrieval_evaluation(bits_per_token=12.0))

        assert any("Bits per token" in str(call) for call in print_mock.call_args_list)


class TestEvalRetrievalMain:
    def test_should_route_to_retrieval_when_task_is_retrieval(self, tmp_path: Path) -> None:
        with ExitStack() as stack:
            config = Mock()
            config.training.seed = 1
            stack.enter_context(
                patch("colors_of_meaning.interface.cli.eval.SynestheticConfig")
            ).from_yaml.return_value = config
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval.AGNewsDatasetAdapter"))
            create = stack.enter_context(patch("colors_of_meaning.interface.cli.eval._create_retriever"))
            create.return_value = (Mock(), 12.0)
            use_case_class = stack.enter_context(patch("colors_of_meaning.interface.cli.eval.RetrievalEvaluateUseCase"))
            use_case = Mock()
            use_case.execute.return_value = _retrieval_evaluation(bits_per_token=12.0)
            use_case_class.return_value = use_case
            stack.enter_context(patch("colors_of_meaning.interface.cli.eval.SklearnMetricsCalculator"))
            stack.enter_context(patch("builtins.print"))
            config_path = tmp_path / "config.yaml"
            config_path.write_text("dummy")

            main(EvalArgs(config=str(config_path), method="color", task="retrieval"))

        use_case.execute.assert_called_once()
