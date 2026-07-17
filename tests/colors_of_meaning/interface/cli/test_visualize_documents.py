from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from colors_of_meaning.interface.cli.visualize_documents import (
    BookColour,
    VisualizeDocumentsArgs,
    _author_labels,
    _build_encoder,
    _encode_book,
    _encode_books,
    _leading_text,
    _render_figures,
    main,
)

MODULE = "colors_of_meaning.interface.cli.visualize_documents"


def _book(author: str, work: str) -> BookColour:
    return BookColour(author=author, work=work, document=Mock(), sheet_path=f"{author}__{work}.png")


class TestVisualizeDocumentsArgs:
    def test_should_default_mapper_type_to_supervised(self) -> None:
        assert VisualizeDocumentsArgs().mapper_type == "supervised"

    def test_should_default_config_to_documents_yaml(self) -> None:
        assert VisualizeDocumentsArgs().config == "configs/documents.yaml"


class TestBuildEncoder:
    @patch(f"{MODULE}.FileColorCodebookRepository")
    @patch(f"{MODULE}.create_color_mapper")
    def test_should_return_codebook_when_it_exists(self, _mapper: Mock, repository: Mock) -> None:
        sentinel = Mock()
        repository.return_value.load.return_value = sentinel

        _use_case, codebook = _build_encoder(VisualizeDocumentsArgs(), Mock())

        assert codebook is sentinel

    @patch(f"{MODULE}.FileColorCodebookRepository")
    @patch(f"{MODULE}.create_color_mapper")
    def test_should_raise_when_codebook_not_found(self, _mapper: Mock, repository: Mock) -> None:
        repository.return_value.load.return_value = None

        with pytest.raises(FileNotFoundError, match="Codebook not found"):
            _build_encoder(VisualizeDocumentsArgs(), Mock())


class TestLeadingText:
    def test_should_read_leading_paragraphs_from_work(self, tmp_path: Path) -> None:
        work = tmp_path / "emma.txt"
        work.write_text("alpha beta gamma\n\ndelta epsilon zeta\n\neta theta iota", encoding="utf-8")

        text = _leading_text(work, min_chars=5, count=2)

        assert text == "alpha beta gamma\n\ndelta epsilon zeta"


class TestEncodeBook:
    def _image_use_case(self) -> Mock:
        image_use_case = Mock()
        image_use_case.execute.return_value = Mock()
        return image_use_case

    def _adapter(self, embeddings: np.ndarray) -> Mock:
        adapter = Mock()
        adapter.encode_document_sentences.return_value = embeddings
        return adapter

    def test_should_return_none_when_no_sentences(self, tmp_path: Path) -> None:
        work = tmp_path / "austen" / "emma.txt"
        work.parent.mkdir(parents=True)
        work.write_text("word " * 60, encoding="utf-8")

        result = _encode_book(work, VisualizeDocumentsArgs(), self._image_use_case(), self._adapter(np.zeros((0, 8))))

        assert result is None

    def test_should_render_the_signature_layout_for_a_book(self, tmp_path: Path) -> None:
        work = tmp_path / "austen" / "emma.txt"
        work.parent.mkdir(parents=True)
        work.write_text("word " * 60, encoding="utf-8")
        image_use_case = self._image_use_case()

        _encode_book(work, VisualizeDocumentsArgs(), image_use_case, self._adapter(np.ones((2, 8))))

        assert image_use_case.execute.call_args.kwargs["layout"] == "signature"

    def test_should_label_the_sheet_path_by_author_and_work(self, tmp_path: Path) -> None:
        work = tmp_path / "austen" / "emma.txt"
        work.parent.mkdir(parents=True)
        work.write_text("word " * 60, encoding="utf-8")

        result = _encode_book(work, VisualizeDocumentsArgs(), self._image_use_case(), self._adapter(np.ones((2, 8))))

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


class TestAuthorLabels:
    def test_should_deduplicate_authors_preserving_order(self) -> None:
        books = [_book("austen", "emma"), _book("austen", "persuasion"), _book("carroll", "alice")]

        _labels, label_names = _author_labels(books)

        assert label_names == ["austen", "carroll"]

    def test_should_label_books_by_author_index(self) -> None:
        books = [_book("austen", "emma"), _book("carroll", "alice"), _book("austen", "persuasion")]

        labels, _label_names = _author_labels(books)

        assert labels == [0, 1, 0]


class TestRenderFigures:
    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_render_signatures(self, _print: Mock, _renderer: Mock, use_case_class: Mock) -> None:
        use_case = use_case_class.return_value

        _render_figures(VisualizeDocumentsArgs(), [_book("austen", "emma")], Mock())

        use_case.execute_corpus_signatures.assert_called_once()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_render_projection_when_multiple_books(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = use_case_class.return_value

        _render_figures(VisualizeDocumentsArgs(), [_book("austen", "emma"), _book("carroll", "alice")], Mock())

        use_case.execute_projection.assert_called_once()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_skip_projection_when_single_book(self, _print: Mock, _renderer: Mock, use_case_class: Mock) -> None:
        use_case = use_case_class.return_value

        _render_figures(VisualizeDocumentsArgs(), [_book("austen", "emma")], Mock())

        use_case.execute_projection.assert_not_called()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_render_the_a4_gallery(self, _print: Mock, _renderer: Mock, use_case_class: Mock) -> None:
        use_case = use_case_class.return_value

        _render_figures(VisualizeDocumentsArgs(), [_book("austen", "emma")], Mock())

        use_case.execute_a4_gallery.assert_called_once()

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_write_signatures_to_the_committed_path(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = use_case_class.return_value

        _render_figures(VisualizeDocumentsArgs(output_dir="reports/figures"), [_book("austen", "emma")], Mock())

        assert use_case.execute_corpus_signatures.call_args.args[4] == "reports/figures/documents_color_signatures.png"

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_write_the_tsne_projection_to_the_committed_path(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = use_case_class.return_value

        _render_figures(
            VisualizeDocumentsArgs(output_dir="reports/figures"),
            [_book("austen", "emma"), _book("carroll", "alice")],
            Mock(),
        )

        assert use_case.execute_projection.call_args.args[3] == "reports/figures/documents_color_tsne.png"

    @patch(f"{MODULE}.VisualizeDocumentsUseCase")
    @patch(f"{MODULE}.MatplotlibFigureRenderer")
    @patch("builtins.print")
    def test_should_write_the_gallery_to_the_committed_path(
        self, _print: Mock, _renderer: Mock, use_case_class: Mock
    ) -> None:
        use_case = use_case_class.return_value

        _render_figures(VisualizeDocumentsArgs(output_dir="reports/figures"), [_book("austen", "emma")], Mock())

        assert use_case.execute_a4_gallery.call_args.args[1] == "reports/figures/documents_a4_gallery.png"


class TestVisualizeDocumentsMain:
    @patch(f"{MODULE}._render_figures")
    @patch(f"{MODULE}._encode_books")
    @patch(f"{MODULE}.SentenceEmbeddingAdapter")
    @patch(f"{MODULE}.EncodeDocumentToImageUseCase")
    @patch(f"{MODULE}._build_encoder")
    @patch(f"{MODULE}.SynestheticConfig")
    def test_should_render_figures_when_books_are_encoded(
        self,
        _config: Mock,
        build_encoder: Mock,
        _image_use_case: Mock,
        _adapter: Mock,
        encode_books: Mock,
        render_figures: Mock,
    ) -> None:
        build_encoder.return_value = (Mock(), Mock())
        encode_books.return_value = [_book("austen", "emma")]

        main(VisualizeDocumentsArgs())

        render_figures.assert_called_once()

    @patch(f"{MODULE}._encode_books")
    @patch(f"{MODULE}.SentenceEmbeddingAdapter")
    @patch(f"{MODULE}.EncodeDocumentToImageUseCase")
    @patch(f"{MODULE}._build_encoder")
    @patch(f"{MODULE}.SynestheticConfig")
    def test_should_raise_when_no_books_are_encoded(
        self,
        _config: Mock,
        build_encoder: Mock,
        _image_use_case: Mock,
        _adapter: Mock,
        encode_books: Mock,
    ) -> None:
        build_encoder.return_value = (Mock(), Mock())
        encode_books.return_value = []

        with pytest.raises(ValueError, match="No books"):
            main(VisualizeDocumentsArgs())
