import tyro
from dataclasses import dataclass
from typing import List

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.visualization.pillow_document_image_renderer import (
    PillowDocumentImageRenderer,
)
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.domain.service.document_image_renderer import DocumentImageLayout
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.encode_document_to_image_use_case import (
    EncodeDocumentToImageUseCase,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook


@dataclass
class EncodeImageArgs:
    text: str = ""
    dataset_path: str = "data/sample_test.txt"
    index: int = 0
    layout: DocumentImageLayout = "score"
    dpi: int = 300
    config: str = "configs/base.yaml"
    model_path: str = "artifacts/models/projector.pth"
    codebook_name: str = "codebook_4096"
    output_path: str = "reports/figures/document_a4.png"


def _load_documents(dataset_path: str) -> List[str]:
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_text(args: EncodeImageArgs) -> str:
    if args.text:
        return args.text
    return _load_documents(args.dataset_path)[args.index]


def _build_encode_use_case(config: SynestheticConfig, model_path: str, codebook_name: str) -> EncodeDocumentUseCase:
    color_mapper = PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        device=config.training.device,
    )
    color_mapper.load_weights(model_path)
    codebook = load_codebook(codebook_name)
    return EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook))


def main(args: EncodeImageArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    document_text = _resolve_text(args)

    encode_use_case = _build_encode_use_case(config, args.model_path, args.codebook_name)
    use_case = EncodeDocumentToImageUseCase(encode_use_case, PillowDocumentImageRenderer())

    embeddings = SentenceEmbeddingAdapter().encode_document_sentences(document_text)
    document = use_case.execute(
        embeddings,
        document_id=f"encoded_{args.index}",
        layout=args.layout,
        output_path=args.output_path,
        dpi=args.dpi,
    )

    sentences = len(document.color_sequence or [])
    print(f"Saved {args.output_path} ({args.layout} layout, {args.dpi} DPI, {sentences} sentences)")


if __name__ == "__main__":
    main(tyro.cli(EncodeImageArgs))
