import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tyro

from colors_of_meaning.application.use_case.visualize_lossless_use_case import (
    VisualizeLosslessUseCase,
)
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.shared.document_corpus import (
    discover_book_images,
    parse_book_image_caption,
)

logger = logging.getLogger(__name__)

GALLERY_TITLE = "Per-book lossless colour barcodes"
MEANING_CAPTION = "meaning · lossy ~1024:1"
EXACT_BYTES_CAPTION = "exact bytes · lossless, byte-for-byte"
LOSSLESS_STAGE = "./bin/generate --only lossless-corpus"
SHEETS_STAGE = "./bin/generate --only documents-figures"


@dataclass
class VisualizeLosslessArgs:
    lossless_dir: str = "reports/figures/lossless"
    sheets_dir: str = "reports/figures/a4"
    book: str = "darwin__origin_of_species"
    title: str = ""
    columns: int = 12
    gallery_path: str = "reports/figures/documents_lossless_gallery.png"
    comparison_path: str = "reports/figures/documents_two_ways.png"


def _require_pages(lossless_dir: Path) -> List[Path]:
    pages = discover_book_images(lossless_dir)
    if not pages:
        raise ValueError(f"No barcode pages found in {lossless_dir}; run {LOSSLESS_STAGE} first")
    return pages


def _require_sheet(sheets_dir: Path, book: str) -> Path:
    sheet = sheets_dir / f"{book}.png"
    if not sheet.is_file():
        raise ValueError(f"Missing semantic sheet {sheet}; run {SHEETS_STAGE} first")
    return sheet


def _require_first_barcode_page(lossless_dir: Path, book: str) -> Path:
    single_page = lossless_dir / f"{book}.png"
    if single_page.is_file():
        return single_page
    pages = sorted(lossless_dir.glob(f"{book}_p*.png"))
    if not pages:
        raise ValueError(f"Missing barcode page for {book} in {lossless_dir}; run {LOSSLESS_STAGE} first")
    return pages[0]


def _comparison_title(args: VisualizeLosslessArgs) -> str:
    if args.title:
        return args.title
    work = args.book.split("__")[-1].replace("_", " ")
    return f"{work.capitalize()} — two ways to colour a book"


def _comparison_panels(sheet: Path, barcode: Path) -> List[Tuple[str, str]]:
    return [(str(sheet), MEANING_CAPTION), (str(barcode), EXACT_BYTES_CAPTION)]


def _log_startup(args: VisualizeLosslessArgs, page_count: int) -> None:
    logger.info(
        "Rendering lossless corpus figures",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "lossless_dir": args.lossless_dir,
            "pages": page_count,
            "book": args.book,
            "gallery_path": args.gallery_path,
            "comparison_path": args.comparison_path,
        },
    )


def main(args: VisualizeLosslessArgs) -> None:
    lossless_dir = Path(args.lossless_dir)
    pages = _require_pages(lossless_dir)
    sheet = _require_sheet(Path(args.sheets_dir), args.book)
    barcode = _require_first_barcode_page(lossless_dir, args.book)

    _log_startup(args, len(pages))
    use_case = VisualizeLosslessUseCase(MatplotlibFigureRenderer())

    use_case.execute_gallery(
        [str(page) for page in pages],
        [parse_book_image_caption(page) for page in pages],
        GALLERY_TITLE,
        args.gallery_path,
        args.columns,
    )
    print(f"Saved {args.gallery_path}")

    use_case.execute_comparison(_comparison_panels(sheet, barcode), _comparison_title(args), args.comparison_path)
    print(f"Saved {args.comparison_path}")


if __name__ == "__main__":
    main(tyro.cli(VisualizeLosslessArgs))
