import tyro
from dataclasses import dataclass

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
)


@dataclass
class TrainArgs:
    config: str = "configs/base.yaml"
    dataset_path: str = "data/train.txt"
    output_model: str = "artifacts/models/projector.pth"
    output_codebook: str = "codebook_4096"
    mapper_type: str = "unconstrained"


def _create_color_mapper(args: TrainArgs, config: SynestheticConfig) -> ColorMapper:
    if args.mapper_type == "structured":
        structured_config = config.structured_mapper
        if structured_config is None:
            raise ValueError("structured_mapper config is required for structured mapper type")
        return StructuredPyTorchColorMapper(
            input_dim=config.projector.embedding_dim,
            hidden_dim_1=config.projector.hidden_dim_1,
            hidden_dim_2=config.projector.hidden_dim_2,
            dropout_rate=config.projector.dropout_rate,
            device=config.training.device,
            alpha=structured_config.alpha,
            beta=structured_config.beta,
            gamma=structured_config.gamma,
            num_clusters=structured_config.num_clusters,
            max_chroma=structured_config.max_chroma,
        )

    return PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
    )


def main(args: TrainArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)

    print(f"Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Encoding {len(texts)} texts with sentence embeddings...")
    embedding_adapter = SentenceEmbeddingAdapter()
    embeddings = embedding_adapter.encode_batch(texts, batch_size=config.training.batch_size, show_progress=True)

    print(f"Training {args.mapper_type} color projector for {config.training.epochs} epochs...")
    color_mapper = _create_color_mapper(args, config)

    codebook_repo = FileColorCodebookRepository()

    use_case = TrainColorMappingUseCase(color_mapper=color_mapper, codebook_repository=codebook_repo)

    use_case.execute(
        embeddings=embeddings,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        bins_per_dimension=config.codebook.bins_per_dimension,
        model_name=args.output_model,
        codebook_name=args.output_codebook,
    )

    print(f"Model saved to {args.output_model}")
    print(f"Codebook saved to artifacts/codebooks/{args.output_codebook}.pkl")


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
