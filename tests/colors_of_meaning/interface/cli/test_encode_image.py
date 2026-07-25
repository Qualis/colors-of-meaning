from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.interface.cli.encode_image import (
    EncodeImageArgs,
    _build_encode_use_case,
    _load_documents,
    _resolve_text,
    main,
)

MODULE = "colors_of_meaning.interface.cli.encode_image"


class TestEncodeImageArgs:
    def test_should_default_layout_to_score(self) -> None:
        assert EncodeImageArgs().layout == "score"

    def test_should_default_dpi_to_300(self) -> None:
        assert EncodeImageArgs().dpi == 300


class TestLoadDocuments:
    def test_should_read_non_blank_lines(self, tmp_path: Path) -> None:
        dataset = tmp_path / "docs.txt"
        dataset.write_text("first\n\n  second  \n")

        assert _load_documents(str(dataset)) == ["first", "second"]


class TestResolveText:
    def test_should_use_explicit_text_when_provided(self) -> None:
        args = EncodeImageArgs(text="hello world")

        assert _resolve_text(args) == "hello world"

    @patch(f"{MODULE}._load_documents")
    def test_should_load_indexed_document_when_text_absent(self, mock_load_documents: Mock) -> None:
        mock_load_documents.return_value = ["alpha", "beta", "gamma"]
        args = EncodeImageArgs(text="", index=1)

        assert _resolve_text(args) == "beta"


class TestBuildEncodeUseCase:
    @patch(f"{MODULE}.EncodeDocumentUseCase")
    @patch(f"{MODULE}.QuantizedColorMapper")
    @patch(f"{MODULE}.load_codebook")
    @patch(f"{MODULE}.PyTorchColorMapper")
    def test_should_return_use_case_when_codebook_exists(
        self,
        _mock_mapper_class: Mock,
        mock_load_codebook: Mock,
        _mock_quantized_class: Mock,
        mock_encode_class: Mock,
    ) -> None:
        mock_load_codebook.return_value = Mock()

        use_case = _build_encode_use_case(Mock(), "model.pth", "codebook_4096")

        assert use_case is mock_encode_class.return_value


class TestEncodeImageMain:
    @patch(f"{MODULE}.SentenceEmbeddingAdapter")
    @patch(f"{MODULE}.EncodeDocumentToImageUseCase")
    @patch(f"{MODULE}.PillowDocumentImageRenderer")
    @patch(f"{MODULE}._build_encode_use_case")
    @patch(f"{MODULE}._resolve_text")
    @patch(f"{MODULE}.SynestheticConfig")
    @patch("builtins.print")
    def test_should_encode_document_to_image(
        self,
        _mock_print: Mock,
        _mock_config_class: Mock,
        mock_resolve_text: Mock,
        _mock_build_use_case: Mock,
        _mock_renderer_class: Mock,
        mock_use_case_class: Mock,
        mock_adapter_class: Mock,
    ) -> None:
        mock_resolve_text.return_value = "a document"
        mock_adapter_class.return_value.encode_document_sentences.return_value = np.ones((2, 384))
        mock_use_case_class.return_value.execute.return_value = ColoredDocument.from_color_sequence([0, 1], num_bins=8)

        main(EncodeImageArgs())

        mock_use_case_class.return_value.execute.assert_called_once()
