import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

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
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
)
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.infrastructure.visualization.pillow_document_image_renderer import (
    PillowDocumentImageRenderer,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook
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
    paragraphs_per_work: Optional[int] = None
    split_strategy: Literal["work", "paragraph"] = "work"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    figure_split: str = "train"
    max_figure_samples: Optional[int] = 2000
    leading_paragraphs: int = 40
    output_dir: str = "reports/figures"
    dpi: int = 150
    columns: int = 12
    top_colors: int = 24


@dataclass
class BookColour:
    author: str
    work: str
    sheet_path: str


def _build_encoder(
    args: VisualizeDocumentsArgs, config: SynestheticConfig
) -> Tuple[EncodeDocumentUseCase, ColorCodebook]:
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    codebook = load_codebook(args.codebook_name)
    return EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook)), codebook


def _resolve_paragraphs_per_work(args: VisualizeDocumentsArgs, config: SynestheticConfig) -> int:
    if args.paragraphs_per_work is not None:
        return args.paragraphs_per_work
    return config.dataset.paragraphs_per_work


def _build_document_corpus(args: VisualizeDocumentsArgs, config: SynestheticConfig) -> DocumentCorpusDatasetAdapter:
    return DocumentCorpusDatasetAdapter(
        documents_dir=args.documents_dir,
        min_paragraph_chars=args.min_paragraph_chars,
        paragraphs_per_work=_resolve_paragraphs_per_work(args, config),
        split_strategy=args.split_strategy,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )


def _encode_paragraph(
    text: str,
    index: int,
    encode_use_case: EncodeDocumentUseCase,
    embedding_adapter: SentenceEmbeddingAdapter,
) -> Optional[ColoredDocument]:
    embeddings = embedding_adapter.encode_document_sentences(text)
    if embeddings.shape[0] == 0:
        return None
    return encode_use_case.execute(embeddings, document_id=f"paragraph_{index}")


def _encode_paragraph_documents(
    args: VisualizeDocumentsArgs,
    config: SynestheticConfig,
    adapter: DocumentCorpusDatasetAdapter,
    encode_use_case: EncodeDocumentUseCase,
    embedding_adapter: SentenceEmbeddingAdapter,
) -> Tuple[List[ColoredDocument], List[int], List[str]]:
    samples = adapter.get_samples(
        split=args.figure_split, max_samples=args.max_figure_samples, seed=config.training.seed
    )
    documents: List[ColoredDocument] = []
    labels: List[int] = []
    for index, sample in enumerate(samples):
        document = _encode_paragraph(sample.text, index, encode_use_case, embedding_adapter)
        if document is not None:
            documents.append(document)
            labels.append(sample.label)
    return documents, labels, adapter.get_label_names()


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
    image_use_case.execute(
        embeddings, document_id=f"{author}__{work}", layout="signature", output_path=sheet_path, dpi=args.dpi
    )
    return BookColour(author=author, work=work, sheet_path=sheet_path)


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


def _render_figures(
    args: VisualizeDocumentsArgs,
    documents: List[ColoredDocument],
    labels: List[int],
    label_names: List[str],
    books: List[BookColour],
    codebook: ColorCodebook,
) -> None:
    use_case = VisualizeDocumentsUseCase(MatplotlibFigureRenderer())
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


def _log_startup(args: VisualizeDocumentsArgs, paragraph_count: int, book_count: int) -> None:
    logger.info(
        "Rendering documents colour figures",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "documents_dir": args.documents_dir,
            "paragraphs": paragraph_count,
            "books": book_count,
            "output_dir": args.output_dir,
        },
    )


def main(args: VisualizeDocumentsArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    encode_use_case, codebook = _build_encoder(args, config)
    embedding_adapter = SentenceEmbeddingAdapter()
    adapter = _build_document_corpus(args, config)
    documents, labels, label_names = _encode_paragraph_documents(
        args, config, adapter, encode_use_case, embedding_adapter
    )
    if not documents:
        raise ValueError(f"No paragraphs were encoded from {args.documents_dir}")

    image_use_case = EncodeDocumentToImageUseCase(encode_use_case, PillowDocumentImageRenderer())
    books = _encode_books(args, image_use_case, embedding_adapter)
    if not books:
        raise ValueError(f"No books were encoded from {args.documents_dir}")

    _log_startup(args, len(documents), len(books))
    _render_figures(args, documents, labels, label_names, books, codebook)


if __name__ == "__main__":
    main(tyro.cli(VisualizeDocumentsArgs))
