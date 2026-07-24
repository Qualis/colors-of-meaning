from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np

from colors_of_meaning.interface.cli.encode import (
    main,
    EncodeArgs,
    _load_documents,
    _setup_color_mapper,
    _load_codebook,
    _create_use_case,
    _encode_documents,
    _save_documents,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestEncodeCLI:
    def test_should_load_documents(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "test.txt"
        dataset_path.write_text("doc1\ndoc2\ndoc3\n")

        documents = _load_documents(str(dataset_path))

        assert len(documents) == 3
        assert documents[0] == "doc1"

    @patch("colors_of_meaning.interface.cli.encode.PyTorchColorMapper")
    def test_should_setup_color_mapper(self, mock_mapper_class: Mock) -> None:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 384
        mock_config.projector.hidden_dim_1 = 128
        mock_config.projector.hidden_dim_2 = 64
        mock_config.training.device = "cpu"

        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        result = _setup_color_mapper(mock_config, "model.pth")

        mock_mapper.load_weights.assert_called_once_with("model.pth")
        assert result == mock_mapper

    @patch("colors_of_meaning.interface.cli.encode.FileColorCodebookRepository")
    def test_should_load_codebook(self, mock_repo_class: Mock) -> None:
        mock_repo = Mock()
        mock_codebook = Mock()
        mock_repo.load.return_value = mock_codebook
        mock_repo_class.return_value = mock_repo

        result = _load_codebook("test_codebook")

        assert result == mock_codebook

    @patch("colors_of_meaning.interface.cli.encode.FileColorCodebookRepository")
    def test_should_raise_error_when_codebook_not_found(self, mock_repo_class: Mock) -> None:
        mock_repo = Mock()
        mock_repo.load.return_value = None
        mock_repo_class.return_value = mock_repo

        try:
            _load_codebook("missing_codebook")
            raise AssertionError("Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            assert "not found" in str(e)

    @patch("colors_of_meaning.interface.cli.encode._setup_color_mapper")
    @patch("colors_of_meaning.interface.cli.encode._load_codebook")
    @patch("colors_of_meaning.interface.cli.encode.QuantizedColorMapper")
    @patch("colors_of_meaning.interface.cli.encode.EncodeDocumentUseCase")
    def test_should_create_use_case(
        self,
        mock_use_case_class: Mock,
        mock_quantized_mapper_class: Mock,
        mock_load_codebook: Mock,
        mock_setup_mapper: Mock,
    ) -> None:
        mock_config = Mock()
        mock_mapper = Mock()
        mock_codebook = Mock()
        mock_setup_mapper.return_value = mock_mapper
        mock_load_codebook.return_value = mock_codebook

        _create_use_case(mock_config, "model.pth", "codebook")

        mock_setup_mapper.assert_called_once()
        mock_load_codebook.assert_called_once()

    def test_should_encode_documents(self) -> None:
        mock_adapter = Mock()
        mock_adapter.encode_document_sentences.return_value = np.array([[1.0, 2.0]])

        mock_use_case = Mock()
        mock_doc = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        mock_use_case.execute.return_value = mock_doc

        documents = ["doc1", "doc2"]

        result = _encode_documents(documents, mock_adapter, mock_use_case, "test")

        assert len(result) == 2
        assert mock_adapter.encode_document_sentences.call_count == 2
        assert mock_use_case.execute.call_count == 2

    def test_should_print_progress_every_100_documents(self) -> None:
        mock_adapter = Mock()
        mock_adapter.encode_document_sentences.return_value = np.array([[1.0, 2.0]])

        mock_use_case = Mock()
        mock_doc = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        mock_use_case.execute.return_value = mock_doc

        documents = ["doc" + str(i) for i in range(100)]

        result = _encode_documents(documents, mock_adapter, mock_use_case, "test")

        assert len(result) == 100

    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.encode.pickle")
    def test_should_save_documents(self, mock_pickle: Mock, mock_file: Mock, tmp_path: Path) -> None:
        docs = [ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))]
        output_path = tmp_path / "output.pkl"

        _save_documents(docs, str(output_path))

        mock_pickle.dump.assert_called_once()

    @patch("colors_of_meaning.interface.cli.encode.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.encode._load_documents")
    @patch("colors_of_meaning.interface.cli.encode._create_use_case")
    @patch("colors_of_meaning.interface.cli.encode.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.encode._encode_documents")
    @patch("colors_of_meaning.interface.cli.encode._save_documents")
    @patch("builtins.print")
    def test_should_execute_encode_workflow(
        self,
        mock_print: Mock,
        mock_save: Mock,
        mock_encode: Mock,
        mock_adapter_class: Mock,
        mock_create_use_case: Mock,
        mock_load_docs: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config_class.from_yaml.return_value = mock_config

        mock_load_docs.return_value = ["doc1", "doc2"]
        mock_encode.return_value = []

        args = EncodeArgs(
            config=str(tmp_path / "config.yaml"),
            dataset_path=str(tmp_path / "data.txt"),
            output_path=str(tmp_path / "output.pkl"),
        )

        main(args)

        mock_load_docs.assert_called_once()
        mock_create_use_case.assert_called_once()
        mock_encode.assert_called_once()
        mock_save.assert_called_once()
