import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.interface.cli.decode_image import (
    DecodeImageArgs,
    _load_corpus,
    _print_summary,
    main,
)

MODULE = "colors_of_meaning.interface.cli.decode_image"


class TestDecodeImageArgs:
    def test_should_default_k_to_5(self) -> None:
        assert DecodeImageArgs().k == 5

    def test_should_default_codebook_name(self) -> None:
        assert DecodeImageArgs().codebook_name == "codebook_4096"


class TestLoadCorpus:
    def test_should_return_empty_when_no_path(self) -> None:
        assert _load_corpus("") == []

    def test_should_load_corpus_from_pickle(self, tmp_path: Path) -> None:
        corpus = [ColoredDocument.from_color_sequence([0, 1], num_bins=8, document_id="d0")]
        corpus_path = tmp_path / "corpus.pkl"
        corpus_path.write_bytes(pickle.dumps(corpus))  # nosemgrep

        assert len(_load_corpus(str(corpus_path))) == 1


class TestPrintSummary:
    def test_should_print_recovered_cell_count(self, capsys: pytest.CaptureFixture[str]) -> None:
        document = ColoredDocument.from_color_sequence([0, 1, 2], num_bins=8, document_id="decoded")

        _print_summary(document, [])

        assert "Recovered 3 cells" in capsys.readouterr().out

    def test_should_print_nearest_neighbours(self, capsys: pytest.CaptureFixture[str]) -> None:
        document = ColoredDocument.from_color_sequence([0, 1, 2], num_bins=8, document_id="decoded")

        _print_summary(document, [("source_doc", 0.0512)])

        assert "source_doc" in capsys.readouterr().out


class TestDecodeImageMain:
    @patch(f"{MODULE}.DecodeImageToDocumentUseCase")
    @patch(f"{MODULE}.PillowDocumentImageRenderer")
    @patch(f"{MODULE}.CompareDocumentsUseCase")
    @patch(f"{MODULE}.WassersteinDistanceCalculator")
    @patch(f"{MODULE}._load_corpus")
    @patch(f"{MODULE}.load_codebook")
    @patch("builtins.print")
    def test_should_decode_image_and_print_summary(
        self,
        _mock_print: Mock,
        _mock_load_codebook: Mock,
        mock_load_corpus: Mock,
        _mock_distance_class: Mock,
        _mock_compare_class: Mock,
        _mock_renderer_class: Mock,
        mock_use_case_class: Mock,
    ) -> None:
        mock_load_corpus.return_value = []
        document = ColoredDocument.from_color_sequence([0, 1], num_bins=8, document_id="decoded")
        mock_use_case_class.return_value.execute.return_value = (document, [("source_doc", 0.05)])

        main(DecodeImageArgs())

        mock_use_case_class.return_value.execute.assert_called_once()
