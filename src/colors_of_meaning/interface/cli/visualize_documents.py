import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import tyro

from colors_of_meaning.application.use_case.encode_document_to_image_use_case import (
    EncodeDocumentToImageUseCase,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.visualize_documents_use_case import (
    VisualizeDocumentsUseCase,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.infrastructure.visualization.pillow_document_image_renderer import (
    PillowDocumentImageRenderer,
)
from colors_of_meaning.shared.document_corpus import (
    discover_author_works,
    extract_paragraphs,
    parse_author_work,
    strip_gutenberg_boilerplate,
)
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)


@dataclass
class VisualizeDocumentsArgs:
    config: str = "configs/documents.yaml"
    model_path: str = "artifacts/models/projector_documents_valsel.pth"
    codebook_name: str = "codebook_documents_valsel"
    mapper_type: str = "supervised"
    documents_dir: str = "./documents"
    min_paragraph_chars: int = 200
    leading_paragraphs: int = 40
    output_dir: str = "reports/figures"
    dpi: int = 150
    columns: int = 12
    top_colors: int = 24


@dataclass
class BookColour:
    author: str
    work: str
    document: ColoredDocument
    sheet_path: str


def _build_encoder(
    args: VisualizeDocumentsArgs, config: SynestheticConfig
) -> Tuple[EncodeDocumentUseCase, ColorCodebook]:
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    codebook = FileColorCodebookRepository().load(args.codebook_name)
    if codebook is None:
        raise FileNotFoundError(f"Codebook not found: {args.codebook_name}")
    return EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook)), codebook


def _leading_text(work_path: Path, min_chars: int, count: int) -> str:
    text = work_path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = extract_paragraphs(strip_gutenberg_boilerplate(text), min_chars)[:count]
    return "\n\n".join(paragraphs)


def _encode_book(
    work_path: Path,
    args: VisualizeDocumentsArgs,
    image_use_case: EncodeDocumentToImageUseCase,
    embedding_adapter: SentenceEmbeddingAdapter,
) -> Optional[BookColour]:
    author, work = parse_author_work(work_path)
    document_text = _leading_text(work_path, args.min_paragraph_chars, args.leading_paragraphs)
    embeddings = embedding_adapter.encode_document_sentences(document_text)
    if embeddings.shape[0] == 0:
        return None
    sheet_path = str(Path(args.output_dir) / "a4" / f"{author}__{work}.png")
    document = image_use_case.execute(
        embeddings, document_id=f"{author}__{work}", layout="signature", output_path=sheet_path, dpi=args.dpi
    )
    return BookColour(author=author, work=work, document=document, sheet_path=sheet_path)


def _encode_books(
    args: VisualizeDocumentsArgs,
    image_use_case: EncodeDocumentToImageUseCase,
    embedding_adapter: SentenceEmbeddingAdapter,
) -> List[BookColour]:
    books: List[BookColour] = []
    for work_path in discover_author_works(Path(args.documents_dir)):
        if not work_path.is_file():
            continue
        book = _encode_book(work_path, args, image_use_case, embedding_adapter)
        if book is not None:
            books.append(book)
    return books


def _author_labels(books: List[BookColour]) -> Tuple[List[int], List[str]]:
    label_names: List[str] = []
    for book in books:
        if book.author not in label_names:
            label_names.append(book.author)
    labels = [label_names.index(book.author) for book in books]
    return labels, label_names


def _render_figures(args: VisualizeDocumentsArgs, books: List[BookColour], codebook: ColorCodebook) -> None:
    use_case = VisualizeDocumentsUseCase(MatplotlibFigureRenderer())
    labels, label_names = _author_labels(books)
    documents = [book.document for book in books]
    output_dir = Path(args.output_dir)

    signatures_path = str(output_dir / "documents_color_signatures.png")
    use_case.execute_corpus_signatures(documents, labels, label_names, codebook, signatures_path, args.top_colors)
    print(f"Saved {signatures_path}")

    if len(documents) >= 2:
        projection_path = str(output_dir / "documents_color_tsne.png")
        use_case.execute_projection(documents, labels, label_names, projection_path)
        print(f"Saved {projection_path}")

    gallery_path = str(output_dir / "documents_a4_gallery.png")
    use_case.execute_a4_gallery([book.sheet_path for book in books], gallery_path, args.columns)
    print(f"Saved {gallery_path}")


def _log_startup(args: VisualizeDocumentsArgs, book_count: int) -> None:
    logger.info(
        "Rendering documents colour figures",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "documents_dir": args.documents_dir,
            "books": book_count,
            "output_dir": args.output_dir,
        },
    )


def main(args: VisualizeDocumentsArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    encode_use_case, codebook = _build_encoder(args, config)
    image_use_case = EncodeDocumentToImageUseCase(encode_use_case, PillowDocumentImageRenderer())
    books = _encode_books(args, image_use_case, SentenceEmbeddingAdapter())
    if not books:
        raise ValueError(f"No books were encoded from {args.documents_dir}")
    _log_startup(args, len(books))
    _render_figures(args, books, codebook)


if __name__ == "__main__":
    main(tyro.cli(VisualizeDocumentsArgs))
