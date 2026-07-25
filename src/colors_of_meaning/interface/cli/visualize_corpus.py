import tyro
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.shared.document_corpus import (
    extract_paragraphs as _extract_paragraphs,
    strip_gutenberg_boilerplate as _strip_gutenberg_boilerplate,
)
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.visualize_documents_use_case import (
    VisualizeDocumentsUseCase,
)
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook


@dataclass
class VisualizeCorpusArgs:
    corpus_specs: str = "sample=data/sample_train.txt"
    config: str = "configs/base.yaml"
    model_path: str = "artifacts/models/projector.pth"
    codebook_name: str = "codebook_4096"
    output_dir: str = "reports/figures"
    paragraphs_per_corpus: int = 60
    min_paragraph_chars: int = 200
    top_colors: int = 24


def _parse_corpus_specs(corpus_specs: str) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for item in corpus_specs.split(","):
        label, _, path = item.partition("=")
        specs.append((label.strip(), path.strip()))
    return specs


def _load_corpus_paragraphs(path: str, min_chars: int, limit: int) -> List[str]:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    paragraphs = _extract_paragraphs(_strip_gutenberg_boilerplate(raw), min_chars)
    start = len(paragraphs) // 5
    return paragraphs[start : start + limit]


def _build_encoder(
    config: SynestheticConfig, model_path: str, codebook_name: str
) -> Tuple[EncodeDocumentUseCase, ColorCodebook]:
    color_mapper = PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        device=config.training.device,
    )
    color_mapper.load_weights(model_path)
    codebook = load_codebook(codebook_name)
    return EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook)), codebook


def _encode_paragraphs(
    use_case: EncodeDocumentUseCase,
    adapter: SentenceEmbeddingAdapter,
    paragraphs: List[str],
    name: str,
) -> List[ColoredDocument]:
    documents: List[ColoredDocument] = []
    for index, paragraph in enumerate(paragraphs):
        embeddings = adapter.encode_document_sentences(paragraph)
        if embeddings.shape[0] == 0:
            continue
        documents.append(use_case.execute(embeddings, document_id=f"{name}_{index}"))
    return documents


def _encode_corpora(
    args: VisualizeCorpusArgs, config: SynestheticConfig
) -> Tuple[List[ColoredDocument], List[int], List[str], ColorCodebook]:
    use_case, codebook = _build_encoder(config, args.model_path, args.codebook_name)
    adapter = SentenceEmbeddingAdapter()

    documents: List[ColoredDocument] = []
    labels: List[int] = []
    label_names: List[str] = []

    for name, path in _parse_corpus_specs(args.corpus_specs):
        paragraphs = _load_corpus_paragraphs(path, args.min_paragraph_chars, args.paragraphs_per_corpus)
        corpus_documents = _encode_paragraphs(use_case, adapter, paragraphs, name)
        if not corpus_documents:
            continue
        label = len(label_names)
        documents.extend(corpus_documents)
        labels.extend([label] * len(corpus_documents))
        label_names.append(name)

    return documents, labels, label_names, codebook


def main(args: VisualizeCorpusArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    documents, labels, label_names, codebook = _encode_corpora(args, config)
    if not documents:
        raise ValueError("No documents were encoded from the provided corpora")

    use_case = VisualizeDocumentsUseCase(MatplotlibFigureRenderer())
    output_dir = Path(args.output_dir)

    signatures_path = str(output_dir / "corpus_color_signatures.png")
    use_case.execute_corpus_signatures(documents, labels, label_names, codebook, signatures_path, args.top_colors)
    print(f"Saved {signatures_path}")

    if len(documents) >= 2:
        projection_path = str(output_dir / "corpus_color_tsne.png")
        use_case.execute_projection(documents, labels, label_names, projection_path)
        print(f"Saved {projection_path}")


if __name__ == "__main__":
    main(tyro.cli(VisualizeCorpusArgs))
