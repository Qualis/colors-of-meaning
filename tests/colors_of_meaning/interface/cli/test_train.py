from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np
import pytest

from colors_of_meaning.interface.cli.train import (
    main,
    TrainArgs,
    _apply_determinism,
    _select_evaluation_embeddings,
    _create_color_mapper,
    _create_codebook_factory,
    _create_dataset_adapter,
    _load_supervised_data,
    _load_texts_from_file,
    _load_training_data,
    _load_documents_data,
    _build_document_corpus,
    _resolve_paragraphs_per_work,
    _maybe_build_validation_selector,
    _build_validation_selector,
    _print_selection_score,
    _configure_supervised_mapper,
    _configure_structured_mapper,
    _uses_label_sentiment,
    _execute_training,
)
from colors_of_meaning.infrastructure.ml.learned_color_codebook_factory import (
    LearnedColorCodebookFactory,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)


class TestDocumentsSource:
    @patch("colors_of_meaning.interface.cli.train.DocumentCorpusDatasetAdapter")
    def test_should_build_document_corpus_with_the_requested_split_strategy(self, mock_adapter: Mock) -> None:
        _build_document_corpus(TrainArgs(source="documents", split_strategy="paragraph"), Mock())

        assert mock_adapter.call_args.kwargs["split_strategy"] == "paragraph"

    @patch("builtins.print")
    @patch("colors_of_meaning.interface.cli.train.DocumentCorpusDatasetAdapter")
    def test_should_return_author_labels_for_supervised_documents(self, mock_adapter: Mock, _print: Mock) -> None:
        mock_adapter.return_value.get_samples.return_value = [Mock(text="a", label=0), Mock(text="b", label=1)]

        _texts, labels, _sentiment = _load_documents_data(
            TrainArgs(source="documents", mapper_type="supervised"), Mock()
        )

        assert list(labels) == [0, 1]

    @patch("builtins.print")
    @patch("colors_of_meaning.interface.cli.train.DocumentCorpusDatasetAdapter")
    def test_should_return_no_labels_for_unconstrained_documents(self, mock_adapter: Mock, _print: Mock) -> None:
        mock_adapter.return_value.get_samples.return_value = [Mock(text="a", label=0)]

        _texts, labels, _sentiment = _load_documents_data(
            TrainArgs(source="documents", mapper_type="unconstrained"), Mock()
        )

        assert labels is None

    @patch("colors_of_meaning.interface.cli.train._load_documents_data")
    @patch("colors_of_meaning.interface.cli.train._validate_sentiment_source")
    def test_should_route_to_documents_loader_when_source_is_documents(
        self, _validate: Mock, mock_load_documents: Mock
    ) -> None:
        mock_load_documents.return_value = (["t"], None, None)

        result = _load_training_data(TrainArgs(source="documents"), Mock())

        assert result == (["t"], None, None)

    def test_should_override_paragraphs_per_work_when_argument_is_provided(self) -> None:
        config = Mock()
        config.dataset.paragraphs_per_work = 60

        assert _resolve_paragraphs_per_work(TrainArgs(paragraphs_per_work=300), config) == 300

    def test_should_fall_back_to_config_paragraphs_per_work_when_not_overridden(self) -> None:
        config = Mock()
        config.dataset.paragraphs_per_work = 300

        assert _resolve_paragraphs_per_work(TrainArgs(), config) == 300


class TestValidationSelection:
    def test_should_return_no_selector_when_selecting_on_structure(self) -> None:
        result = _maybe_build_validation_selector(
            TrainArgs(select_on="structure"), Mock(), Mock(), np.zeros((2, 2)), np.array([0, 1])
        )

        assert result is None

    def test_should_raise_when_validation_selection_lacks_documents_source(self) -> None:
        with pytest.raises(ValueError, match="requires --source documents"):
            _maybe_build_validation_selector(
                TrainArgs(select_on="validation", source="dataset"), Mock(), Mock(), np.zeros((2, 2)), np.array([0, 1])
            )

    def test_should_raise_when_validation_selection_lacks_labels(self) -> None:
        with pytest.raises(ValueError, match="requires --mapper-type supervised"):
            _maybe_build_validation_selector(
                TrainArgs(select_on="validation", source="documents"), Mock(), Mock(), np.zeros((2, 2)), None
            )

    @patch("colors_of_meaning.interface.cli.train._build_validation_selector")
    def test_should_build_selector_for_valid_validation_selection(self, mock_build: Mock) -> None:
        result = _maybe_build_validation_selector(
            TrainArgs(select_on="validation", source="documents"), Mock(), Mock(), np.zeros((2, 2)), np.array([0, 1])
        )

        assert result is mock_build.return_value

    @patch("colors_of_meaning.interface.cli.train.ValidationAccuracyCheckpointSelector")
    @patch("colors_of_meaning.interface.cli.train.JensenShannonDistanceCalculator")
    @patch("colors_of_meaning.interface.cli.train.EncodeDocumentUseCase")
    @patch("colors_of_meaning.interface.cli.train.QuantizedColorMapper")
    @patch("colors_of_meaning.interface.cli.train.ColorCodebook")
    @patch("colors_of_meaning.interface.cli.train.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.train._build_document_corpus")
    @patch("builtins.print")
    def test_should_construct_the_selector_from_the_validation_split(
        self,
        _print: Mock,
        mock_corpus: Mock,
        _embed: Mock,
        _codebook: Mock,
        _quantized: Mock,
        _encode: Mock,
        _distance: Mock,
        mock_selector: Mock,
    ) -> None:
        mock_corpus.return_value.get_samples.return_value = [Mock(text="a", label=0), Mock(text="b", label=1)]
        config = Mock()
        config.training.batch_size = 8
        config.codebook.bins_per_dimension = 2
        config.distance.smoothing_epsilon = 1e-8

        result = _build_validation_selector(
            TrainArgs(source="documents"), config, Mock(), np.zeros((4, 2)), np.array([0, 0, 1, 1])
        )

        assert result is mock_selector.return_value

    @patch("builtins.print")
    def test_should_print_validation_accuracy_when_selecting_on_validation(self, mock_print: Mock) -> None:
        _print_selection_score(TrainArgs(select_on="validation"), 0.42)

        assert "Validation authorship accuracy" in mock_print.call_args.args[0]

    @patch("builtins.print")
    def test_should_print_structure_correlation_when_selecting_on_structure(self, mock_print: Mock) -> None:
        _print_selection_score(TrainArgs(select_on="structure"), -0.3)

        assert "Structure-preservation correlation" in mock_print.call_args.args[0]


class TestCreateColorMapper:
    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    def test_should_create_unconstrained_mapper(self, mock_config_class: Mock) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 10
        mock_config.projector.hidden_dim_1 = 8
        mock_config.projector.hidden_dim_2 = 4
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config.training.seed = 42

        args = TrainArgs(mapper_type="unconstrained")

        mapper = _create_color_mapper(args, mock_config)

        assert isinstance(mapper, PyTorchColorMapper)

    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    def test_should_create_structured_mapper(self, mock_config_class: Mock) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 10
        mock_config.projector.hidden_dim_1 = 8
        mock_config.projector.hidden_dim_2 = 4
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config.training.seed = 42
        mock_config.structured_mapper.alpha = 1.0
        mock_config.structured_mapper.beta = 1.0
        mock_config.structured_mapper.gamma = 1.0
        mock_config.structured_mapper.num_clusters = 16
        mock_config.structured_mapper.max_chroma = 128.0
        mock_config.structured_mapper.sentiment_source = "none"
        mock_config.structured_mapper.concreteness_resource = "concreteness_norms.tsv"

        args = TrainArgs(mapper_type="structured")

        mapper = _create_color_mapper(args, mock_config)

        assert isinstance(mapper, StructuredPyTorchColorMapper)

    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    def test_should_inject_concreteness_lexicon_into_structured_mapper_when_creating_mapper(
        self, mock_config_class: Mock
    ) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 10
        mock_config.projector.hidden_dim_1 = 8
        mock_config.projector.hidden_dim_2 = 4
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config.training.seed = 42
        mock_config.structured_mapper.alpha = 1.0
        mock_config.structured_mapper.beta = 1.0
        mock_config.structured_mapper.gamma = 1.0
        mock_config.structured_mapper.num_clusters = 16
        mock_config.structured_mapper.max_chroma = 128.0
        mock_config.structured_mapper.sentiment_source = "none"
        mock_config.structured_mapper.concreteness_resource = "concreteness_norms.tsv"

        mapper = _create_color_mapper(TrainArgs(mapper_type="structured"), mock_config)

        assert mapper._concreteness_lexicon is not None

    def test_should_raise_when_structured_config_is_none(self) -> None:
        mock_config = Mock()
        mock_config.structured_mapper = None

        args = TrainArgs(mapper_type="structured")

        with pytest.raises(ValueError):
            _create_color_mapper(args, mock_config)

    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    def test_should_create_supervised_mapper(self, mock_config_class: Mock) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 10
        mock_config.projector.hidden_dim_1 = 8
        mock_config.projector.hidden_dim_2 = 4
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config.training.seed = 42
        mock_config.supervised_mapper.classification_weight = 0.1
        mock_config.supervised_mapper.num_classes = 4

        args = TrainArgs(mapper_type="supervised")

        mapper = _create_color_mapper(args, mock_config)

        assert isinstance(mapper, SupervisedPyTorchColorMapper)

    def test_should_raise_when_supervised_config_is_none(self) -> None:
        mock_config = Mock()
        mock_config.supervised_mapper = None

        args = TrainArgs(mapper_type="supervised")

        with pytest.raises(ValueError):
            _create_color_mapper(args, mock_config)


class TestCreateDatasetAdapter:
    def test_should_create_ag_news_adapter(self) -> None:
        adapter = _create_dataset_adapter("ag_news")

        assert adapter is not None

    def test_should_create_imdb_adapter(self) -> None:
        adapter = _create_dataset_adapter("imdb")

        assert adapter is not None

    def test_should_create_newsgroups_adapter(self) -> None:
        adapter = _create_dataset_adapter("20newsgroups")

        assert adapter is not None

    def test_should_raise_for_unknown_dataset(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            _create_dataset_adapter("nonexistent")


class TestLoadSupervisedData:
    @patch("colors_of_meaning.interface.cli.train._create_dataset_adapter")
    def test_should_load_texts_and_labels_from_dataset(self, mock_create_adapter: Mock) -> None:
        mock_adapter = Mock()
        mock_sample_1 = Mock(text="hello world", label=0)
        mock_sample_2 = Mock(text="foo bar", label=1)
        mock_adapter.get_samples.return_value = [mock_sample_1, mock_sample_2]
        mock_create_adapter.return_value = mock_adapter

        mock_config = Mock()
        mock_config.dataset.name = "ag_news"
        mock_config.dataset.train_split = "train"
        mock_config.dataset.max_samples = None

        texts, labels = _load_supervised_data(mock_config)

        assert texts == ["hello world", "foo bar"]
        assert list(labels) == [0, 1]

    @patch("colors_of_meaning.interface.cli.train._create_dataset_adapter")
    def test_should_forward_configured_seed_to_get_samples(self, mock_create_adapter: Mock) -> None:
        mock_adapter = Mock()
        mock_adapter.get_samples.return_value = [Mock(text="hello", label=0)]
        mock_create_adapter.return_value = mock_adapter

        mock_config = Mock()
        mock_config.dataset.name = "ag_news"
        mock_config.dataset.train_split = "train"
        mock_config.dataset.max_samples = 5
        mock_config.training.seed = 7

        _load_supervised_data(mock_config)

        assert mock_adapter.get_samples.call_args[1]["seed"] == 7


class TestLoadTextsFromFile:
    def test_should_load_texts_from_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "texts.txt"
        file_path.write_text("hello\nworld\n\nfoo\n")

        texts = _load_texts_from_file(str(file_path))

        assert texts == ["hello", "world", "foo"]


class TestConfigureSupervisedMapper:
    def test_should_set_labels_on_supervised_mapper(self) -> None:
        mapper = SupervisedPyTorchColorMapper(input_dim=10, device="cpu", num_classes=3)
        labels = np.array([0, 1, 2])

        _configure_supervised_mapper(mapper, labels)

        assert mapper._training_labels is not None

    def test_should_skip_non_supervised_mapper(self) -> None:
        mapper = Mock()

        _configure_supervised_mapper(mapper, np.array([0, 1]))

        mapper.set_training_labels.assert_not_called()


class TestExecuteTraining:
    @patch("colors_of_meaning.interface.cli.train.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.train.TrainColorMappingUseCase")
    @patch("builtins.print")
    def test_should_execute_training_use_case(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
    ) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = -0.5
        mock_use_case_class.return_value = mock_use_case

        mock_config = Mock()
        mock_config.training.epochs = 10
        mock_config.training.learning_rate = 0.001
        mock_config.training.seed = 42
        mock_config.codebook.bins_per_dimension = 4

        args = TrainArgs()
        mock_mapper = Mock()
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])

        _execute_training(args, mock_config, mock_mapper, embeddings)

        mock_use_case.execute.assert_called_once()

    @patch("colors_of_meaning.interface.cli.train.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.train.TrainColorMappingUseCase")
    @patch("builtins.print")
    def test_should_inject_learned_codebook_factory_into_use_case_when_training(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
    ) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = -0.5
        mock_use_case_class.return_value = mock_use_case

        mock_config = Mock()
        mock_config.training.epochs = 10
        mock_config.training.learning_rate = 0.001
        mock_config.training.seed = 42
        mock_config.codebook.bins_per_dimension = 4
        mock_config.codebook.num_bins = 64

        _execute_training(TrainArgs(), mock_config, Mock(), np.array([[1.0, 2.0], [3.0, 4.0]]))

        assert isinstance(mock_use_case_class.call_args.kwargs["codebook_factory"], LearnedColorCodebookFactory)

    @patch("colors_of_meaning.interface.cli.train.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.train.TrainColorMappingUseCase")
    @patch("builtins.print")
    def test_should_thread_codebook_mode_into_use_case_execute_when_training(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
    ) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = -0.5
        mock_use_case_class.return_value = mock_use_case

        mock_config = Mock()
        mock_config.training.epochs = 10
        mock_config.training.learning_rate = 0.001
        mock_config.training.seed = 42
        mock_config.codebook.bins_per_dimension = 4
        mock_config.codebook.num_bins = 64

        _execute_training(TrainArgs(codebook_mode="learned"), mock_config, Mock(), np.array([[1.0, 2.0]]))

        assert mock_use_case.execute.call_args.kwargs["codebook_mode"] == "learned"


class TestCreateCodebookFactory:
    def test_should_default_codebook_mode_to_uniform_when_not_specified(self) -> None:
        assert TrainArgs().codebook_mode == "uniform"

    def test_should_create_learned_codebook_factory_for_mapper(self) -> None:
        factory = _create_codebook_factory(Mock())

        assert isinstance(factory, LearnedColorCodebookFactory)


class TestTrainCLI:
    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.train.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.train._create_color_mapper")
    @patch("colors_of_meaning.interface.cli.train.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.train.TrainColorMappingUseCase")
    @patch("builtins.open", new_callable=mock_open, read_data="text1\ntext2\ntext3\n")
    @patch("builtins.print")
    def test_should_execute_training_workflow(
        self,
        mock_print: Mock,
        mock_file: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
        mock_create_mapper: Mock,
        mock_adapter_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.training.batch_size = 32
        mock_config.training.epochs = 10
        mock_config.training.learning_rate = 0.001
        mock_config.training.device = "cpu"
        mock_config.training.seed = 42
        mock_config.codebook.bins_per_dimension = 4
        mock_config.projector.embedding_dim = 384
        mock_config.projector.hidden_dim_1 = 128
        mock_config.projector.hidden_dim_2 = 64
        mock_config.projector.dropout_rate = 0.1
        mock_config_class.from_yaml.return_value = mock_config

        mock_adapter = Mock()
        mock_adapter.encode_batch.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_adapter_class.return_value = mock_adapter

        mock_mapper = Mock()
        mock_create_mapper.return_value = mock_mapper

        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        mock_use_case = Mock()
        mock_use_case.execute.return_value = -0.5
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        dataset_path = tmp_path / "train.txt"
        output_model = tmp_path / "model.pth"
        output_codebook = tmp_path / "codebook"

        config_path.write_text("dummy")
        dataset_path.write_text("text1\ntext2\ntext3\n")

        args = TrainArgs(
            config=str(config_path),
            dataset_path=str(dataset_path),
            output_model=str(output_model),
            output_codebook=str(output_codebook),
            mapper_type="unconstrained",
        )

        main(args)

        mock_config_class.from_yaml.assert_called_once_with(str(config_path))
        mock_adapter.encode_batch.assert_called_once()
        mock_use_case.execute.assert_called_once()

    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.train.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.train._create_color_mapper")
    @patch("colors_of_meaning.interface.cli.train._load_supervised_data")
    @patch("colors_of_meaning.interface.cli.train._execute_training")
    @patch("builtins.print")
    def test_should_execute_supervised_training_workflow(
        self,
        mock_print: Mock,
        mock_execute: Mock,
        mock_load_supervised: Mock,
        mock_create_mapper: Mock,
        mock_adapter_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.training.batch_size = 32
        mock_config.training.seed = 42
        mock_config.dataset.name = "ag_news"
        mock_config_class.from_yaml.return_value = mock_config

        mock_load_supervised.return_value = (["text1", "text2"], np.array([0, 1]))

        mock_adapter = Mock()
        mock_adapter.encode_batch.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_adapter_class.return_value = mock_adapter

        mock_mapper = SupervisedPyTorchColorMapper(input_dim=2, device="cpu", num_classes=2)
        mock_create_mapper.return_value = mock_mapper

        config_path = tmp_path / "config.yaml"
        config_path.write_text("dummy")

        args = TrainArgs(
            config=str(config_path),
            mapper_type="supervised",
        )

        main(args)

        mock_load_supervised.assert_called_once_with(mock_config)
        assert mock_mapper._training_labels is not None


class TestApplyDeterminism:
    def test_should_enable_deterministic_algorithms_when_flag_is_set(self) -> None:
        mock_config = Mock()
        mock_config.training.seed = 42

        with patch("torch.use_deterministic_algorithms") as mock_use_deterministic:
            _apply_determinism(TrainArgs(deterministic=True), mock_config)

        mock_use_deterministic.assert_called_once_with(True)

    def test_should_not_enable_deterministic_algorithms_by_default(self) -> None:
        mock_config = Mock()
        mock_config.training.seed = 42

        with patch("torch.use_deterministic_algorithms") as mock_use_deterministic:
            _apply_determinism(TrainArgs(), mock_config)

        mock_use_deterministic.assert_not_called()


class TestSelectEvaluationEmbeddings:
    def test_should_select_evaluation_embeddings_deterministically(self) -> None:
        embeddings = np.arange(40, dtype=np.float32).reshape(20, 2)

        first = _select_evaluation_embeddings(embeddings, seed=5)
        second = _select_evaluation_embeddings(embeddings, seed=5)

        assert np.array_equal(first, second)

    def test_should_cap_evaluation_embeddings_to_maximum(self) -> None:
        embeddings = np.arange(40, dtype=np.float32).reshape(20, 2)

        selected = _select_evaluation_embeddings(embeddings, seed=5, max_evaluation_samples=8)

        assert len(selected) == 8


class TestConfigureStructuredMapper:
    def test_should_set_training_texts_on_structured_mapper(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=2, device="cpu")

        _configure_structured_mapper(mapper, ["a", "b"], None)

        assert mapper._training_texts == ["a", "b"]

    def test_should_set_sentiment_scores_on_structured_mapper_when_dataset_has_labels(self) -> None:
        mapper = StructuredPyTorchColorMapper(input_dim=2, device="cpu")

        _configure_structured_mapper(mapper, ["a", "b"], np.array([0, 1]))

        assert mapper._sentiment_scores is not None

    def test_should_skip_structured_configuration_for_non_structured_mapper(self) -> None:
        mapper = Mock()

        _configure_structured_mapper(mapper, ["a"], None)

        mapper.set_training_texts.assert_not_called()


class TestUsesLabelSentiment:
    def test_should_use_label_sentiment_when_structured_source_is_labels(self) -> None:
        config = Mock()
        config.structured_mapper.sentiment_source = "labels"

        assert _uses_label_sentiment(TrainArgs(mapper_type="structured"), config) is True

    def test_should_not_use_label_sentiment_when_source_is_none(self) -> None:
        config = Mock()
        config.structured_mapper.sentiment_source = "none"

        assert _uses_label_sentiment(TrainArgs(mapper_type="structured"), config) is False


class TestLoadTrainingData:
    @patch("colors_of_meaning.interface.cli.train._load_supervised_data")
    @patch("builtins.print")
    def test_should_load_sentiment_scores_when_structured_source_is_labels(
        self, mock_print: Mock, mock_load: Mock
    ) -> None:
        mock_load.return_value = (["t1", "t2"], np.array([0, 1]))
        config = Mock()
        config.structured_mapper.sentiment_source = "labels"

        _, labels, sentiment = _load_training_data(TrainArgs(mapper_type="structured"), config)

        assert labels is None and sentiment is not None

    @patch("builtins.print")
    def test_should_load_file_texts_when_structured_source_is_none(self, mock_print: Mock, tmp_path: Path) -> None:
        dataset_path = tmp_path / "texts.txt"
        dataset_path.write_text("a\nb\n")
        config = Mock()
        config.structured_mapper.sentiment_source = "none"
        args = TrainArgs(mapper_type="structured", dataset_path=str(dataset_path))

        texts, labels, sentiment = _load_training_data(args, config)

        assert texts == ["a", "b"] and labels is None and sentiment is None

    def test_should_raise_when_sentiment_source_is_unknown(self) -> None:
        config = Mock()
        config.structured_mapper.sentiment_source = "typo"

        with pytest.raises(ValueError):
            _load_training_data(TrainArgs(mapper_type="structured"), config)
