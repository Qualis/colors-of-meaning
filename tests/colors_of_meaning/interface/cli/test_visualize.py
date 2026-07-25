from unittest.mock import Mock, patch

import pytest

from colors_of_meaning.interface.cli.visualize import VisualizeArgs, main


class TestVisualizeArgs:
    def test_should_create_args_with_defaults(self) -> None:
        args = VisualizeArgs()

        assert args.visualization_type == "codebook"

    def test_should_have_default_codebook_name(self) -> None:
        args = VisualizeArgs()

        assert args.codebook_name == "codebook_4096"

    def test_should_have_default_output_dir(self) -> None:
        args = VisualizeArgs()

        assert args.output_dir == "reports/figures"

    def test_should_have_default_dataset(self) -> None:
        args = VisualizeArgs()

        assert args.dataset == "ag_news"

    def test_should_have_default_max_samples(self) -> None:
        args = VisualizeArgs()

        assert args.max_samples == 500


class TestVisualizeCLICodebook:
    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeCodebookUseCase")
    @patch("builtins.print")
    def test_should_execute_codebook_visualization(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
        mock_renderer_class: Mock,
    ) -> None:
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="codebook", codebook_name="codebook_4096")
        main(args)

        mock_use_case.execute.assert_called_once()

    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeCodebookUseCase")
    @patch("builtins.print")
    def test_should_pass_codebook_name_to_use_case(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
        mock_renderer_class: Mock,
    ) -> None:
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="codebook", codebook_name="test_codebook")
        main(args)

        mock_use_case.execute.assert_called_once_with("test_codebook", "reports/figures/codebook_palette.png")


class TestVisualizeCLIHistograms:
    @patch("colors_of_meaning.interface.cli.visualize._encode_samples")
    @patch("colors_of_meaning.interface.cli.visualize.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.visualize.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeDocumentsUseCase")
    @patch("builtins.print")
    def test_should_execute_histograms_visualization(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_renderer_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        mock_encode_samples: Mock,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config
        mock_dataset = Mock()
        mock_sample = Mock()
        mock_sample.label = 0
        mock_dataset.get_samples.return_value = [mock_sample]
        mock_dataset.get_label_names.return_value = ["Class A"]
        mock_dataset_class.return_value = mock_dataset
        mock_encode_samples.return_value = [Mock()]
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="histograms")
        main(args)

        mock_use_case.execute_histograms.assert_called_once()

    @patch("colors_of_meaning.interface.cli.visualize._encode_samples")
    @patch("colors_of_meaning.interface.cli.visualize.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.visualize.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeDocumentsUseCase")
    @patch("builtins.print")
    def test_should_load_dataset_samples_for_histograms(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_renderer_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        mock_encode_samples: Mock,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config
        mock_dataset = Mock()
        mock_sample = Mock()
        mock_sample.label = 0
        mock_dataset.get_samples.return_value = [mock_sample]
        mock_dataset.get_label_names.return_value = ["Class A"]
        mock_dataset_class.return_value = mock_dataset
        mock_encode_samples.return_value = [Mock()]
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="histograms", max_samples=100)
        main(args)

        mock_dataset.get_samples.assert_called_once_with(split="test", max_samples=100)


class TestVisualizeCLIProjection:
    @patch("colors_of_meaning.interface.cli.visualize._encode_samples")
    @patch("colors_of_meaning.interface.cli.visualize.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.visualize.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeDocumentsUseCase")
    @patch("builtins.print")
    def test_should_execute_projection_visualization(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_renderer_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        mock_encode_samples: Mock,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config
        mock_dataset = Mock()
        mock_sample = Mock()
        mock_sample.label = 0
        mock_dataset.get_samples.return_value = [mock_sample]
        mock_dataset.get_label_names.return_value = ["Class A"]
        mock_dataset_class.return_value = mock_dataset
        mock_encode_samples.return_value = [Mock()]
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="projection")
        main(args)

        mock_use_case.execute_projection.assert_called_once()


class TestVisualizeCLIConfusionMatrix:
    @patch("colors_of_meaning.interface.cli.visualize._create_classifier")
    @patch("colors_of_meaning.interface.cli.visualize.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.visualize.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeDocumentsUseCase")
    @patch("builtins.print")
    def test_should_execute_confusion_matrix_visualization(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_renderer_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        mock_create_classifier: Mock,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config
        mock_dataset = Mock()
        mock_sample = Mock()
        mock_sample.label = 0
        mock_dataset.get_samples.return_value = [mock_sample]
        mock_dataset.get_label_names.return_value = ["Class A"]
        mock_dataset_class.return_value = mock_dataset
        mock_classifier = Mock()
        mock_classifier.predict.return_value = [0]
        mock_create_classifier.return_value = mock_classifier
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="confusion_matrix")
        main(args)

        mock_use_case.execute_confusion_matrix.assert_called_once()

    @patch("colors_of_meaning.interface.cli.visualize._create_classifier")
    @patch("colors_of_meaning.interface.cli.visualize.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.visualize.AGNewsDatasetAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.MatplotlibFigureRenderer")
    @patch("colors_of_meaning.interface.cli.visualize.VisualizeDocumentsUseCase")
    @patch("builtins.print")
    def test_should_fit_classifier_before_prediction(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_renderer_class: Mock,
        mock_dataset_class: Mock,
        mock_config_class: Mock,
        mock_create_classifier: Mock,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config
        mock_dataset = Mock()
        mock_sample = Mock()
        mock_sample.label = 0
        mock_dataset.get_samples.return_value = [mock_sample]
        mock_dataset.get_label_names.return_value = ["Class A"]
        mock_dataset_class.return_value = mock_dataset
        mock_classifier = Mock()
        mock_classifier.predict.return_value = [0]
        mock_create_classifier.return_value = mock_classifier
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        args = VisualizeArgs(visualization_type="confusion_matrix")
        main(args)

        mock_classifier.fit.assert_called_once()


class TestVisualizeCLIUnknownType:
    def test_should_raise_error_for_unknown_visualization_type(self) -> None:
        args = VisualizeArgs()
        args.visualization_type = "invalid"  # type: ignore

        with pytest.raises(ValueError, match="Unknown visualization type: invalid"):
            main(args)


class TestEncodeSamples:
    @patch("colors_of_meaning.interface.cli.visualize.EncodeDocumentUseCase")
    @patch("colors_of_meaning.interface.cli.visualize.QuantizedColorMapper")
    @patch("colors_of_meaning.interface.cli.visualize.load_codebook")
    @patch("colors_of_meaning.interface.cli.visualize.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.visualize.SentenceEmbeddingAdapter")
    @patch("builtins.print")
    def test_should_encode_samples_to_colored_documents(
        self,
        mock_print: Mock,
        mock_embedding_class: Mock,
        mock_mapper_class: Mock,
        mock_load_codebook: Mock,
        mock_quantized_class: Mock,
        mock_encode_class: Mock,
    ) -> None:
        from colors_of_meaning.interface.cli.visualize import _encode_samples

        mock_config = Mock()
        mock_config.projector.embedding_dim = 384
        mock_config.projector.hidden_dim_1 = 128
        mock_config.projector.hidden_dim_2 = 64
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"

        mock_embedding = Mock()
        mock_embedding.encode_document_sentences.return_value = Mock()
        mock_embedding_class.return_value = mock_embedding

        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        mock_codebook = Mock()
        mock_load_codebook.return_value = mock_codebook

        mock_encode = Mock()
        mock_doc = Mock()
        mock_encode.execute.return_value = mock_doc
        mock_encode_class.return_value = mock_encode

        mock_sample = Mock()
        mock_sample.text = "test text"

        result = _encode_samples([mock_sample], mock_config, "model.pth", "codebook")

        assert len(result) == 1

    @patch("colors_of_meaning.interface.cli.visualize.EncodeDocumentUseCase")
    @patch("colors_of_meaning.interface.cli.visualize.QuantizedColorMapper")
    @patch("colors_of_meaning.interface.cli.visualize.load_codebook")
    @patch("colors_of_meaning.interface.cli.visualize.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.visualize.SentenceEmbeddingAdapter")
    @patch("builtins.print")
    def test_should_print_progress_every_100_samples(
        self,
        mock_print: Mock,
        mock_embedding_class: Mock,
        mock_mapper_class: Mock,
        mock_load_codebook: Mock,
        mock_quantized_class: Mock,
        mock_encode_class: Mock,
    ) -> None:
        from colors_of_meaning.interface.cli.visualize import _encode_samples

        mock_config = Mock()
        mock_config.projector.embedding_dim = 384
        mock_config.projector.hidden_dim_1 = 128
        mock_config.projector.hidden_dim_2 = 64
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"

        mock_embedding = Mock()
        mock_embedding.encode_document_sentences.return_value = Mock()
        mock_embedding_class.return_value = mock_embedding

        mock_load_codebook.return_value = Mock()

        mock_encode = Mock()
        mock_encode.execute.return_value = Mock()
        mock_encode_class.return_value = mock_encode

        samples = [Mock(text=f"text {i}") for i in range(100)]
        _encode_samples(samples, mock_config, "model.pth", "codebook")

        mock_print.assert_called_once_with("  Encoded 100/100 documents")


class TestCreateClassifier:
    @patch("colors_of_meaning.interface.cli.visualize.TFIDFClassifier")
    def test_should_create_tfidf_classifier(self, mock_tfidf_class: Mock) -> None:
        from colors_of_meaning.interface.cli.visualize import _create_classifier

        mock_classifier = Mock()
        mock_tfidf_class.return_value = mock_classifier

        args = VisualizeArgs(method="tfidf")
        result = _create_classifier(args, Mock())

        assert result == mock_classifier

    @patch("colors_of_meaning.interface.cli.visualize.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.HNSWClassifier")
    def test_should_create_hnsw_classifier(
        self,
        mock_hnsw_class: Mock,
        mock_embedding_class: Mock,
    ) -> None:
        from colors_of_meaning.interface.cli.visualize import _create_classifier

        mock_classifier = Mock()
        mock_hnsw_class.return_value = mock_classifier

        args = VisualizeArgs(method="hnsw", k_neighbors=3)
        result = _create_classifier(args, Mock())

        assert result == mock_classifier

    @patch("colors_of_meaning.interface.cli.visualize.EncodeDocumentUseCase")
    @patch("colors_of_meaning.interface.cli.visualize.QuantizedColorMapper")
    @patch("colors_of_meaning.interface.cli.visualize.load_codebook")
    @patch("colors_of_meaning.interface.cli.visualize.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.visualize.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.visualize.ColorHistogramClassifier")
    @patch("colors_of_meaning.interface.cli.visualize.WassersteinDistanceCalculator")
    def test_should_create_color_classifier(
        self,
        mock_wasserstein_class: Mock,
        mock_color_classifier_class: Mock,
        mock_embedding_class: Mock,
        mock_mapper_class: Mock,
        mock_load_codebook: Mock,
        mock_quantized_class: Mock,
        mock_encode_class: Mock,
    ) -> None:
        from colors_of_meaning.interface.cli.visualize import _create_classifier

        mock_config = Mock()
        mock_config.projector.embedding_dim = 384
        mock_config.projector.hidden_dim_1 = 128
        mock_config.projector.hidden_dim_2 = 64
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"

        mock_codebook = Mock()
        mock_load_codebook.return_value = mock_codebook

        mock_classifier = Mock()
        mock_color_classifier_class.return_value = mock_classifier

        args = VisualizeArgs(method="color")
        result = _create_classifier(args, mock_config)

        assert result == mock_classifier
        mock_wasserstein_class.assert_called_once_with(
            codebook=mock_codebook, sinkhorn_reg=mock_config.distance.sinkhorn_reg
        )

    def test_should_raise_for_unknown_method(self) -> None:
        from colors_of_meaning.interface.cli.visualize import _create_classifier

        args = VisualizeArgs()
        args.method = "invalid"  # type: ignore

        with pytest.raises(ValueError, match="Unknown method: invalid"):
            _create_classifier(args, Mock())
