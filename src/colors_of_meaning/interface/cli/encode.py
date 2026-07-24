import tyro
import pickle  # nosec B403
from dataclasses import dataclass
from pathlib import Path
from typing import List

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.color_codebook import ColorCodebook


@dataclass
class EncodeArgs:
    config: str = "configs/base.yaml"
    split: str = "test"
    dataset_path: str = "data/test.txt"
    model_path: str = "artifacts/models/projector.pth"
    codebook_name: str = "codebook_4096"
    output_path: str = "artifacts/encoded/test_documents.pkl"


def _load_documents(dataset_path: str) -> List[str]:
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _setup_color_mapper(config: SynestheticConfig, model_path: str) -> PyTorchColorMapper:
    color_mapper = PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        device=config.training.device,
    )
    color_mapper.load_weights(model_path)
    return color_mapper


def _load_codebook(codebook_name: str) -> ColorCodebook:
    codebook_repo = FileColorCodebookRepository()
    codebook = codebook_repo.load(codebook_name)
    if codebook is None:
        raise FileNotFoundError(f"Codebook not found: {codebook_name}")
    return codebook


def _create_use_case(config: SynestheticConfig, model_path: str, codebook_name: str) -> EncodeDocumentUseCase:
    color_mapper = _setup_color_mapper(config, model_path)
    codebook = _load_codebook(codebook_name)
    quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
    return EncodeDocumentUseCase(quantized_mapper)


def _encode_documents(
    documents: List[str],
    embedding_adapter: SentenceEmbeddingAdapter,
    use_case: EncodeDocumentUseCase,
    split_name: str,
) -> List[ColoredDocument]:
    colored_documents: List[ColoredDocument] = []

    for i, doc_text in enumerate(documents):
        sentence_embeddings = embedding_adapter.encode_document_sentences(doc_text)
        colored_doc = use_case.execute(sentence_embeddings, document_id=f"{split_name}_{i}")
        colored_documents.append(colored_doc)

        if (i + 1) % 100 == 0:
            print(f"Encoded {i + 1}/{len(documents)} documents")

    return colored_documents


def _save_documents(colored_documents: List[ColoredDocument], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(colored_documents, f)  # nosemgrep


def main(args: EncodeArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)

    print(f"Loading dataset from {args.dataset_path}...")
    documents = _load_documents(args.dataset_path)

    print("Loading color mapper and codebook...")
    use_case = _create_use_case(config, args.model_path, args.codebook_name)
    embedding_adapter = SentenceEmbeddingAdapter()

    print(f"Encoding {len(documents)} documents...")
    colored_documents = _encode_documents(documents, embedding_adapter, use_case, args.split)

    _save_documents(colored_documents, args.output_path)
    print(f"Encoded documents saved to {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(EncodeArgs))
