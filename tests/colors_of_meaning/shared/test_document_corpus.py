from pathlib import Path

from colors_of_meaning.shared.document_corpus import (
    discover_author_works,
    discover_book_images,
    extract_paragraphs,
    parse_author_work,
    parse_book_image_caption,
    strip_gutenberg_boilerplate,
)

GUTENBERG_TEXT = "header junk\n*** START OF EBOOK ***\nthe body\n*** END OF EBOOK ***\nlicense"


class TestStripGutenbergBoilerplate:
    def test_should_drop_content_before_the_start_marker(self) -> None:
        assert "header junk" not in strip_gutenberg_boilerplate(GUTENBERG_TEXT)

    def test_should_retain_the_body_between_the_markers(self) -> None:
        assert "the body" in strip_gutenberg_boilerplate(GUTENBERG_TEXT)

    def test_should_drop_the_license_after_the_end_marker(self) -> None:
        assert "license" not in strip_gutenberg_boilerplate(GUTENBERG_TEXT)

    def test_should_return_the_whole_text_when_markers_are_absent(self) -> None:
        text = "a plain document with no gutenberg markers at all"

        assert strip_gutenberg_boilerplate(text) == text


class TestExtractParagraphs:
    def test_should_keep_paragraphs_at_or_above_the_minimum_length(self) -> None:
        text = "tiny\n\n" + ("word " * 60)

        assert len(extract_paragraphs(text, min_chars=200)) == 1

    def test_should_drop_paragraphs_below_the_minimum_length(self) -> None:
        assert extract_paragraphs("tiny\n\nalso tiny", min_chars=200) == []

    def test_should_join_wrapped_lines_within_a_single_paragraph(self) -> None:
        assert extract_paragraphs("alpha\nbeta", min_chars=1) == ["alpha beta"]

    def test_should_normalise_windows_newlines_into_paragraph_breaks(self) -> None:
        assert extract_paragraphs("alpha\r\n\r\nbeta", min_chars=1) == ["alpha", "beta"]


class TestParseAuthorWork:
    def test_should_read_the_author_from_the_parent_directory(self) -> None:
        assert parse_author_work(Path("documents/austen/pride.txt"))[0] == "austen"

    def test_should_read_the_work_from_the_filename_stem(self) -> None:
        assert parse_author_work(Path("documents/austen/pride.txt"))[1] == "pride"


def _write(root: Path, author: str, work: str) -> None:
    (root / author).mkdir(parents=True, exist_ok=True)
    (root / author / f"{work}.txt").write_text("body", encoding="utf-8")


class TestDiscoverAuthorWorks:
    def test_should_return_works_ordered_by_author_then_work(self, tmp_path: Path) -> None:
        _write(tmp_path, "carroll", "alice")
        _write(tmp_path, "austen", "persuasion")
        _write(tmp_path, "austen", "emma")

        works = discover_author_works(tmp_path)

        assert [parse_author_work(path) for path in works] == [
            ("austen", "emma"),
            ("austen", "persuasion"),
            ("carroll", "alice"),
        ]

    def test_should_return_empty_when_root_is_not_a_directory(self, tmp_path: Path) -> None:
        assert discover_author_works(tmp_path / "missing") == []


def _write_image(root: Path, name: str) -> None:
    (root / name).write_bytes(b"png bytes")


class TestDiscoverBookImages:
    def test_should_return_images_ordered_by_name(self, tmp_path: Path) -> None:
        _write_image(tmp_path, "carroll__alice.png")
        _write_image(tmp_path, "austen__emma.png")

        images = discover_book_images(tmp_path)

        assert [path.name for path in images] == ["austen__emma.png", "carroll__alice.png"]

    def test_should_return_empty_when_root_is_not_a_directory(self, tmp_path: Path) -> None:
        assert discover_book_images(tmp_path / "missing") == []

    def test_should_ignore_files_that_are_not_png(self, tmp_path: Path) -> None:
        _write_image(tmp_path, "austen__emma.png")
        (tmp_path / "notes.txt").write_text("body", encoding="utf-8")

        assert [path.name for path in discover_book_images(tmp_path)] == ["austen__emma.png"]


class TestParseBookImageCaption:
    def test_should_render_the_author_and_work_as_a_path(self) -> None:
        assert parse_book_image_caption(Path("austen__emma.png")) == "austen/emma"

    def test_should_keep_the_page_suffix_for_multi_page_books(self) -> None:
        assert parse_book_image_caption(Path("darwin__the_descent_of_man_p02.png")) == "darwin/the_descent_of_man_p02"
