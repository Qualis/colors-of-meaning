from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np
import pytest

from colors_of_meaning.interface.cli.train import (
    main,
    TrainArgs,
    _create_color_mapper,
    _create_dataset_adapter,
    _load_supervised_data,
    _load_texts_from_file,
    _configure_supervised_mapper,
    _execute_training,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)


class TestCreateColorMapper:
    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    def test_should_create_unconstrained_mapper(self, mock_config_class: Mock) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 10
        mock_config.projector.hidden_dim_1 = 8
        mock_config.projector.hidden_dim_2 = 4
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"

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
        mock_config.structured_mapper.alpha = 1.0
        mock_config.structured_mapper.beta = 1.0
        mock_config.structured_mapper.gamma = 1.0
        mock_config.structured_mapper.num_clusters = 16
        mock_config.structured_mapper.max_chroma = 128.0

        args = TrainArgs(mapper_type="structured")

        mapper = _create_color_mapper(args, mock_config)

        assert isinstance(mapper, StructuredPyTorchColorMapper)

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
        mock_use_case_class.return_value = mock_use_case

        mock_config = Mock()
        mock_config.training.epochs = 10
        mock_config.training.learning_rate = 0.001
        mock_config.codebook.bins_per_dimension = 4

        args = TrainArgs()
        mock_mapper = Mock()
        embeddings = np.array([[1.0, 2.0]])

        _execute_training(args, mock_config, mock_mapper, embeddings)

        mock_use_case.execute.assert_called_once()


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
