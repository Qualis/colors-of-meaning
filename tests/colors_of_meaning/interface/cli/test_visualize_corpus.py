from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.interface.cli.visualize_corpus import (
    VisualizeCorpusArgs,
    main,
    _parse_corpus_specs,
    _strip_gutenberg_boilerplate,
    _extract_paragraphs,
    _load_corpus_paragraphs,
    _build_encoder,
    _encode_corpora,
)

MODULE = "colors_of_meaning.interface.cli.visualize_corpus"


class TestVisualizeCorpusArgs:
    def test_should_default_codebook_name(self) -> None:
        args = VisualizeCorpusArgs()

        assert args.codebook_name == "codebook_4096"

    def test_should_default_top_colors(self) -> None:
        args = VisualizeCorpusArgs()

        assert args.top_colors == 24


class TestParseCorpusSpecs:
    def test_should_parse_label_path_pairs(self) -> None:
        specs = _parse_corpus_specs("Darwin=a.txt,Smith=b.txt")

        assert specs == [("Darwin", "a.txt"), ("Smith", "b.txt")]


GUTENBERG_TEXT = "header junk\n*** START OF EBOOK ***\nthe body\n*** END OF EBOOK ***\nlicense"


class TestStripGutenbergBoilerplate:
    def test_should_strip_content_before_start_marker(self) -> None:
        result = _strip_gutenberg_boilerplate(GUTENBERG_TEXT)

        assert "header junk" not in result

    def test_should_retain_body_between_markers(self) -> None:
        result = _strip_gutenberg_boilerplate(GUTENBERG_TEXT)

        assert "the body" in result

    def test_should_drop_license_after_end_marker(self) -> None:
        result = _strip_gutenberg_boilerplate(GUTENBERG_TEXT)

        assert "license" not in result

    def test_should_return_whole_text_when_markers_absent(self) -> None:
        text = "a plain document with no gutenberg markers at all"

        result = _strip_gutenberg_boilerplate(text)

        assert result == text


class TestExtractParagraphs:
    def test_should_keep_paragraphs_above_minimum_length(self) -> None:
        text = "tiny\n\n" + ("word " * 60)

        paragraphs = _extract_paragraphs(text, min_chars=200)

        assert len(paragraphs) == 1


class TestLoadCorpusParagraphs:
    def test_should_load_paragraphs_from_file(self, tmp_path: Path) -> None:
        corpus_file = tmp_path / "corpus.txt"
        corpus_file.write_text("alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota")

        paragraphs = _load_corpus_paragraphs(str(corpus_file), min_chars=5, limit=10)

        assert len(paragraphs) == 3


class TestBuildEncoder:
    @patch(f"{MODULE}.EncodeDocumentUseCase")
    @patch(f"{MODULE}.QuantizedColorMapper")
    @patch(f"{MODULE}.load_codebook")
    @patch(f"{MODULE}.PyTorchColorMapper")
    def test_should_return_codebook_when_it_exists(
        self,
        mock_mapper_class: Mock,
        mock_load_codebook: Mock,
        mock_quantized_class: Mock,
        mock_encode_class: Mock,
    ) -> None:
        sentinel_codebook = Mock()
        mock_load_codebook.return_value = sentinel_codebook

        _use_case, codebook = _build_encoder(Mock(), "model.pth", "codebook_4096")

        assert codebook is sentinel_codebook


class TestEncodeCorpora:
    @patch(f"{MODULE}.SentenceEmbeddingAdapter")
    @patch(f"{MODULE}._load_corpus_paragraphs")
    @patch(f"{MODULE}._build_encoder")
    def test_should_skip_paragraphs_without_sentences(
        self,
        mock_build_encoder: Mock,
        mock_load_paragraphs: Mock,
        mock_adapter_class: Mock,
    ) -> None:
        mock_use_case = Mock()
        mock_use_case.execute.return_value = Mock()
        mock_build_encoder.return_value = (mock_use_case, Mock())
        mock_load_paragraphs.return_value = ["empty paragraph", "good paragraph"]
        mock_adapter_class.return_value.encode_document_sentences.side_effect = [
            np.zeros((0, 384)),
            np.ones((2, 384)),
        ]
        args = VisualizeCorpusArgs(corpus_specs="A=x.txt")

        documents, _labels, _label_names, _codebook = _encode_corpora(args, Mock())

        assert len(documents) == 1

    @patch(f"{MODULE}.SentenceEmbeddingAdapter")
    @patch(f"{MODULE}._load_corpus_paragraphs")
    @patch(f"{MODULE}._build_encoder")
    def test_should_label_documents_by_corpus_index(
        self,
        mock_build_encoder: Mock,
        mock_load_paragraphs: Mock,
        mock_adapter_class: Mock,
    ) -> None:
        mock_build_encoder.return_value = (Mock(), Mock())
        mock_load_paragraphs.return_value = ["paragraph one"]
        mock_adapter_class.return_value.encode_document_sentences.return_value = np.ones((1, 384))
        args = VisualizeCorpusArgs(corpus_specs="A=x.txt,B=y.txt")

        _documents, labels, _label_names, _codebook = _encode_corpora(args, Mock())

        assert labels == [0, 1]

    @patch(f"{MODULE}.SentenceEmbeddingAdapter")
    @patch(f"{MODULE}._load_corpus_paragraphs")
    @patch(f"{MODULE}._build_encoder")
    def test_should_skip_corpus_that_yields_no_documents(
        self,
        mock_build_encoder: Mock,
        mock_load_paragraphs: Mock,
        mock_adapter_class: Mock,
    ) -> None:
        mock_build_encoder.return_value = (Mock(), Mock())
        mock_load_paragraphs.return_value = ["unusable paragraph"]
        mock_adapter_class.return_value.encode_document_sentences.return_value = np.zeros((0, 384))
        args = VisualizeCorpusArgs(corpus_specs="Empty=x.txt")

        _documents, _labels, label_names, _codebook = _encode_corpora(args, Mock())

        assert label_names == []


class TestVisualizeCorpusMain:
    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch(f"{MODULE}._encode_corpora")
    @patch(f"{MODULE}.SynestheticConfig")
    @patch("builtins.print")
    def test_should_render_corpus_signatures(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_encode_corpora: Mock,
        mock_renderer_class: Mock,
        mock_use_case_class: Mock,
    ) -> None:
        mock_encode_corpora.return_value = ([Mock()], [0], ["A"], Mock())
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        main(VisualizeCorpusArgs())

        mock_use_case.execute_corpus_signatures.assert_called_once()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch(f"{MODULE}._encode_corpora")
    @patch(f"{MODULE}.SynestheticConfig")
    @patch("builtins.print")
    def test_should_render_tsne_projection_when_multiple_documents(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_encode_corpora: Mock,
        mock_renderer_class: Mock,
        mock_use_case_class: Mock,
    ) -> None:
        mock_encode_corpora.return_value = ([Mock(), Mock()], [0, 1], ["A", "B"], Mock())
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        main(VisualizeCorpusArgs())

        mock_use_case.execute_projection.assert_called_once()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch(f"{MODULE}._encode_corpora")
    @patch(f"{MODULE}.SynestheticConfig")
    @patch("builtins.print")
    def test_should_skip_projection_when_single_document(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_encode_corpora: Mock,
        mock_renderer_class: Mock,
        mock_use_case_class: Mock,
    ) -> None:
        mock_encode_corpora.return_value = ([Mock()], [0], ["A"], Mock())
        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        main(VisualizeCorpusArgs())

        mock_use_case.execute_projection.assert_not_called()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch(f"{MODULE}._encode_corpora")
    @patch(f"{MODULE}.SynestheticConfig")
    @patch("builtins.print")
    def test_should_raise_when_no_documents_encoded(
        self,
        mock_print: Mock,
        mock_config_class: Mock,
        mock_encode_corpora: Mock,
        mock_renderer_class: Mock,
        mock_use_case_class: Mock,
    ) -> None:
        mock_encode_corpora.return_value = ([], [], [], Mock())

        with pytest.raises(ValueError, match="No documents"):
            main(VisualizeCorpusArgs())
