from pathlib import Path

import pytest

from colors_of_meaning.interface.cli.visualize_lossless import (
    EXACT_BYTES_CAPTION,
    MEANING_CAPTION,
    VisualizeLosslessArgs,
    main,
)

MODULE = "colors_of_meaning.interface.cli.visualize_lossless"


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png bytes")


def _corpus(tmp_path: Path, page_names: list, sheet_names: list) -> None:
    for name in page_names:
        _write_png(tmp_path / "lossless" / name)
    for name in sheet_names:
        _write_png(tmp_path / "a4" / name)


def _args(tmp_path: Path, **overrides) -> VisualizeLosslessArgs:
    return VisualizeLosslessArgs(
        lossless_dir=str(tmp_path / "lossless"),
        sheets_dir=str(tmp_path / "a4"),
        gallery_path=str(tmp_path / "gallery.png"),
        comparison_path=str(tmp_path / "two_ways.png"),
        **overrides,
    )


def _setup(mocker, tmp_path: Path, **overrides):
    mocker.patch(f"{MODULE}.MatplotlibFigureRenderer")
    mocker.patch("builtins.print")
    return mocker.patch(f"{MODULE}.VisualizeLosslessUseCase"), _args(tmp_path, **overrides)


class TestVisualizeLosslessSourceValidation:
    def test_should_raise_when_no_barcode_pages_are_found(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, [], ["darwin__origin_of_species.png"])
        _, args = _setup(mocker, tmp_path)

        with pytest.raises(ValueError, match="lossless-corpus"):
            main(args)

    def test_should_raise_when_the_semantic_sheet_is_missing(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png"], [])
        _, args = _setup(mocker, tmp_path)

        with pytest.raises(ValueError, match="documents-figures"):
            main(args)

    def test_should_raise_when_the_requested_book_has_no_barcode_page(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["austen__emma.png"], ["darwin__origin_of_species.png"])
        _, args = _setup(mocker, tmp_path)

        with pytest.raises(ValueError, match="Missing barcode page"):
            main(args)


class TestVisualizeLosslessGallery:
    def test_should_pass_every_barcode_page_in_name_order(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png", "austen__emma.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path)

        main(args)

        assert [Path(path).name for path in use_case.return_value.execute_gallery.call_args.args[0]] == [
            "austen__emma.png",
            "darwin__origin_of_species.png",
        ]

    def test_should_derive_captions_from_the_page_filenames(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png", "austen__emma.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path)

        main(args)

        assert use_case.return_value.execute_gallery.call_args.args[1] == [
            "austen/emma",
            "darwin/origin_of_species",
        ]

    def test_should_pass_the_requested_column_count(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path, columns=6)

        main(args)

        assert use_case.return_value.execute_gallery.call_args.args[4] == 6


class TestVisualizeLosslessComparison:
    def test_should_pair_the_semantic_sheet_with_the_barcode(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path)

        main(args)

        assert [caption for _, caption in use_case.return_value.execute_comparison.call_args.args[0]] == [
            MEANING_CAPTION,
            EXACT_BYTES_CAPTION,
        ]

    def test_should_use_the_first_numbered_page_for_a_multi_page_book(self, mocker, tmp_path: Path) -> None:
        pages = ["darwin__origin_of_species_p02.png", "darwin__origin_of_species_p01.png"]
        _corpus(tmp_path, pages, ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path)

        main(args)

        barcode_path = use_case.return_value.execute_comparison.call_args.args[0][1][0]
        assert Path(barcode_path).name == "darwin__origin_of_species_p01.png"

    def test_should_derive_the_title_from_the_book_name(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path)

        main(args)

        assert use_case.return_value.execute_comparison.call_args.args[1] == (
            "Origin of species — two ways to colour a book"
        )

    def test_should_prefer_an_explicit_title_over_the_derived_one(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path, title="On the Origin of Species")

        main(args)

        assert use_case.return_value.execute_comparison.call_args.args[1] == "On the Origin of Species"

    def test_should_write_the_comparison_to_the_requested_path(self, mocker, tmp_path: Path) -> None:
        _corpus(tmp_path, ["darwin__origin_of_species.png"], ["darwin__origin_of_species.png"])
        use_case, args = _setup(mocker, tmp_path)

        main(args)

        assert use_case.return_value.execute_comparison.call_args.args[2] == str(tmp_path / "two_ways.png")
