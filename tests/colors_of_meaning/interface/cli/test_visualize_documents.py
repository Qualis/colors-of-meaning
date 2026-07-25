from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.interface.cli.visualize_documents import (
    BookColour,
    VisualizeDocumentsArgs,
    _build_document_corpus,
    _build_encoder,
    _encode_book,
    _encode_books,
    _encode_paragraph,
    _encode_paragraph_documents,
    _leading_text,
    _render_figures,
    _resolve_paragraphs_per_work,
    main,
)

MODULE = "colors_of_meaning.interface.cli.visualize_documents"


def _book(author: str, work: str) -> BookColour:
    return BookColour(author=author, work=work, sheet_path=f"reports/figures/a4/{author}__{work}.png")


def _config(paragraphs_per_work: int = 60) -> Mock:
    config = Mock()
    config.dataset.paragraphs_per_work = paragraphs_per_work
    config.training.seed = 42
    return config


class TestVisualizeDocumentsArgs:
    def test_should_default_mapper_type_to_supervised(self) -> None:
        assert VisualizeDocumentsArgs().mapper_type == "supervised"

    def test_should_default_to_the_author_contrastive_projector(self) -> None:
        assert VisualizeDocumentsArgs().model_path == "artifacts/models/projector_documents_valsel.pth"


class TestBuildEncoder:
    @patch(f"{MODULE}.load_codebook")
    @patch(f"{MODULE}.create_color_mapper")
    def test_should_return_codebook_when_it_exists(self, _mapper: Mock, load_codebook: Mock) -> None:
        sentinel = Mock()
        load_codebook.return_value = sentinel

        _use_case, codebook = _build_encoder(VisualizeDocumentsArgs(), Mock())

        assert codebook is sentinel


class TestResolveParagraphsPerWork:
    def test_should_prefer_the_explicit_argument(self) -> None:
        assert _resolve_paragraphs_per_work(VisualizeDocumentsArgs(paragraphs_per_work=120), _config()) == 120

    def test_should_fall_back_to_the_config_default(self) -> None:
        assert _resolve_paragraphs_per_work(VisualizeDocumentsArgs(paragraphs_per_work=None), _config(300)) == 300


class TestBuildDocumentCorpus:
    @patch(f"{MODULE}.DocumentCorpusDatasetAdapter")
    def test_should_pass_the_resolved_cap_to_the_adapter(self, adapter: Mock) -> None:
        _build_document_corpus(VisualizeDocumentsArgs(paragraphs_per_work=90), _config())

        assert adapter.call_args.kwargs["paragraphs_per_work"] == 90


class TestEncodeParagraph:
    def _adapter(self, embeddings: np.ndarray) -> Mock:
        adapter = Mock()
        adapter.encode_document_sentences.return_value = embeddings
        return adapter

    def test_should_return_none_when_no_sentences(self) -> None:
        result = _encode_paragraph("text", 0, Mock(), self._adapter(np.zeros((0, 8))))

        assert result is None

    def test_should_encode_a_paragraph_with_sentences(self) -> None:
        encode_use_case = Mock()
        sentinel = Mock()
        encode_use_case.execute.return_value = sentinel

        result = _encode_paragraph("text", 0, encode_use_case, self._adapter(np.ones((2, 8))))

        assert result is sentinel


class TestEncodeParagraphDocuments:
    def _adapter(self, samples: list) -> Mock:
        adapter = Mock()
        adapter.get_samples.return_value = samples
        adapter.get_label_names.return_value = ["austen", "darwin"]
        return adapter

    @patch(f"{MODULE}._encode_paragraph")
    def test_should_collect_a_document_per_encoded_paragraph(self, encode_paragraph: Mock) -> None:
        encode_paragraph.side_effect = [Mock(), Mock()]
        adapter = self._adapter([Mock(text="a", label=0), Mock(text="b", label=1)])

        documents, _labels, _names = _encode_paragraph_documents(
            VisualizeDocumentsArgs(), _config(), adapter, Mock(), Mock()
        )

        assert len(documents) == 2

    @patch(f"{MODULE}._encode_paragraph")
    def test_should_align_labels_with_encoded_documents(self, encode_paragraph: Mock) -> None:
        encode_paragraph.side_effect = [None, Mock()]
        adapter = self._adapter([Mock(text="a", label=0), Mock(text="b", label=1)])

        _documents, labels, _names = _encode_paragraph_documents(
            VisualizeDocumentsArgs(), _config(), adapter, Mock(), Mock()
        )

        assert labels == [1]

    @patch(f"{MODULE}._encode_paragraph")
    def test_should_return_the_corpus_label_names(self, encode_paragraph: Mock) -> None:
        encode_paragraph.return_value = Mock()
        adapter = self._adapter([Mock(text="a", label=0)])

        _documents, _labels, label_names = _encode_paragraph_documents(
            VisualizeDocumentsArgs(), _config(), adapter, Mock(), Mock()
        )

        assert label_names == ["austen", "darwin"]


class TestLeadingText:
    def test_should_read_leading_paragraphs_from_work(self, tmp_path: Path) -> None:
        work = tmp_path / "emma.txt"
        work.write_text("alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota", encoding="utf-8")

        text = _leading_text(work, min_chars=5, count=2)

        assert text == "alpha beta gamma\n\ndelta epsilon zeta"


class TestEncodeBook:
    def _adapter(self, embeddings: np.ndarray) -> Mock:
        adapter = Mock()
        adapter.encode_document_sentences.return_value = embeddings
        return adapter

    def _work(self, tmp_path: Path) -> Path:
        work = tmp_path / "austen" / "emma.txt"
        work.parent.mkdir(parents=True)
        work.write_text("word " * 60, encoding="utf-8")
        return work

    def test_should_return_none_when_no_sentences(self, tmp_path: Path) -> None:
        result = _encode_book(self._work(tmp_path), VisualizeDocumentsArgs(), Mock(), self._adapter(np.zeros((0, 8))))

        assert result is None

    def test_should_render_the_signature_layout_for_a_book(self, tmp_path: Path) -> None:
        image_use_case = Mock()

        _encode_book(self._work(tmp_path), VisualizeDocumentsArgs(), image_use_case, self._adapter(np.ones((2, 8))))

        assert image_use_case.execute.call_args.kwargs["layout"] == "signature"

    def test_should_label_the_sheet_path_by_author_and_work(self, tmp_path: Path) -> None:
        result = _encode_book(self._work(tmp_path), VisualizeDocumentsArgs(), Mock(), self._adapter(np.ones((2, 8))))

        assert result is not None and result.sheet_path.endswith("a4/austen__emma.png")


class TestEncodeBooks:
    @patch(f"{MODULE}._encode_book")
    @patch(f"{MODULE}.discover_author_works")
    def test_should_skip_paths_that_are_not_files(self, discover: Mock, encode_book: Mock, tmp_path: Path) -> None:
        real_file = tmp_path / "austen" / "emma.txt"
        real_file.parent.mkdir(parents=True)
        real_file.write_text("body", encoding="utf-8")
        directory = tmp_path / "carroll" / "alice.txt"
        directory.mkdir(parents=True)
        discover.return_value = [real_file, directory]
        encode_book.return_value = _book("austen", "emma")

        _encode_books(VisualizeDocumentsArgs(), Mock(), Mock())

        encode_book.assert_called_once()

    @patch(f"{MODULE}._encode_book")
    @patch(f"{MODULE}.discover_author_works")
    def test_should_collect_encoded_books(self, discover: Mock, encode_book: Mock, tmp_path: Path) -> None:
        work = tmp_path / "austen" / "emma.txt"
        work.parent.mkdir(parents=True)
        work.write_text("body", encoding="utf-8")
        discover.return_value = [work]
        encode_book.return_value = _book("austen", "emma")

        books = _encode_books(VisualizeDocumentsArgs(), Mock(), Mock())

        assert len(books) == 1

    @patch(f"{MODULE}._encode_book")
    @patch(f"{MODULE}.discover_author_works")
    def test_should_drop_books_without_sentences(self, discover: Mock, encode_book: Mock, tmp_path: Path) -> None:
        work = tmp_path / "austen" / "emma.txt"
        work.parent.mkdir(parents=True)
        work.write_text("body", encoding="utf-8")
        discover.return_value = [work]
        encode_book.return_value = None

        books = _encode_books(VisualizeDocumentsArgs(), Mock(), Mock())

        assert books == []


def _render(use_case_class: Mock, documents: list, books: list) -> Mock:
    labels = list(range(len(documents)))
    _render_figures(VisualizeDocumentsArgs(output_dir="reports/figures"), documents, labels, ["a", "b"], books, Mock())
    return use_case_class.return_value


class TestRenderFigures:
    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_write_signatures_to_the_committed_path(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = _render(use_case_class, [Mock()], [_book("austen", "emma")])

        assert use_case.execute_corpus_signatures.call_args.args[4] == "reports/figures/documents_color_signatures.png"

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_write_the_tsne_projection_to_the_committed_path(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = _render(use_case_class, [Mock(), Mock()], [_book("austen", "emma")])

        assert use_case.execute_projection.call_args.args[3] == "reports/figures/documents_color_tsne.png"

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_skip_projection_when_single_document(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = _render(use_case_class, [Mock()], [_book("austen", "emma")])

        use_case.execute_projection.assert_not_called()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_write_the_gallery_to_the_committed_path(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = _render(use_case_class, [Mock()], [_book("austen", "emma")])

        assert use_case.execute_a4_gallery.call_args.args[1] == "reports/figures/documents_a4_gallery.png"


class TestVisualizeDocumentsMain:
    def _patch_main(self, mocker) -> Mock:
        mocker.patch(f"{MODULE}.SynestheticConfig").from_yaml.return_value = _config()
        mocker.patch(f"{MODULE}._build_encoder", return_value=(Mock(), Mock()))
        mocker.patch(f"{MODULE}.SentenceEmbeddingAdapter")
        mocker.patch(f"{MODULE}._build_document_corpus")
        mocker.patch(f"{MODULE}.EncodeDocumentToImageUseCase")
        return mocker.patch(f"{MODULE}._render_figures")

    def test_should_render_figures_when_documents_and_books_exist(self, mocker) -> None:
        render_figures = self._patch_main(mocker)
        mocker.patch(f"{MODULE}._encode_paragraph_documents", return_value=([Mock()], [0], ["austen"]))
        mocker.patch(f"{MODULE}._encode_books", return_value=[_book("austen", "emma")])

        main(VisualizeDocumentsArgs())

        render_figures.assert_called_once()

    def test_should_raise_when_no_paragraphs_are_encoded(self, mocker) -> None:
        self._patch_main(mocker)
        mocker.patch(f"{MODULE}._encode_paragraph_documents", return_value=([], [], []))
        mocker.patch(f"{MODULE}._encode_books", return_value=[_book("austen", "emma")])

        with pytest.raises(ValueError, match="No paragraphs"):
            main(VisualizeDocumentsArgs())

    def test_should_raise_when_no_books_are_encoded(self, mocker) -> None:
        self._patch_main(mocker)
        mocker.patch(f"{MODULE}._encode_paragraph_documents", return_value=([Mock()], [0], ["austen"]))
        mocker.patch(f"{MODULE}._encode_books", return_value=[])

        with pytest.raises(ValueError, match="No books"):
            main(VisualizeDocumentsArgs())
