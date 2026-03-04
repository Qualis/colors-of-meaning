import numpy as np
import numpy.typing as npt
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
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
)
from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import (
    AGNewsDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter import (
    IMDBDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter import (
    NewsgroupsDatasetAdapter,
)
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository


@dataclass
class TrainArgs:
    config: str = "configs/base.yaml"
    dataset_path: str = "data/train.txt"
    output_model: str = "artifacts/models/projector.pth"
    output_codebook: str = "codebook_4096"
    mapper_type: str = "unconstrained"


def _create_dataset_adapter(dataset_name: str) -> DatasetRepository:
    adapters: dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "20newsgroups": NewsgroupsDatasetAdapter,
    }
    adapter_class = adapters.get(dataset_name)
    if adapter_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(adapters.keys())}")
    return adapter_class()


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

    if args.mapper_type == "supervised":
        supervised_config = config.supervised_mapper
        if supervised_config is None:
            raise ValueError("supervised_mapper config is required for supervised mapper type")
        return SupervisedPyTorchColorMapper(
            input_dim=config.projector.embedding_dim,
            hidden_dim_1=config.projector.hidden_dim_1,
            hidden_dim_2=config.projector.hidden_dim_2,
            dropout_rate=config.projector.dropout_rate,
            device=config.training.device,
            num_classes=supervised_config.num_classes,
            classification_weight=supervised_config.classification_weight,
        )

    return PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
    )


def _load_supervised_data(config: SynestheticConfig) -> tuple:
    dataset_adapter = _create_dataset_adapter(config.dataset.name)
    samples = dataset_adapter.get_samples(
        split=config.dataset.train_split,
        max_samples=config.dataset.max_samples,
    )
    texts = [sample.text for sample in samples]
    labels = np.array([sample.label for sample in samples])
    return texts, labels


def _load_texts_from_file(dataset_path: str) -> list:
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _configure_supervised_mapper(
    color_mapper: ColorMapper,
    labels: npt.NDArray,
) -> None:
    if isinstance(color_mapper, SupervisedPyTorchColorMapper):
        color_mapper.set_training_labels(labels)


def main(args: TrainArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    labels = None

    if args.mapper_type == "supervised":
        print(f"Loading labeled dataset: {config.dataset.name}...")
        texts, labels = _load_supervised_data(config)
    else:
        print(f"Loading dataset from {args.dataset_path}...")
        texts = _load_texts_from_file(args.dataset_path)

    print(f"Encoding {len(texts)} texts with sentence embeddings...")
    embedding_adapter = SentenceEmbeddingAdapter()
    embeddings = embedding_adapter.encode_batch(texts, batch_size=config.training.batch_size, show_progress=True)

    color_mapper = _create_color_mapper(args, config)

    if labels is not None:
        _configure_supervised_mapper(color_mapper, labels)

    _execute_training(args, config, color_mapper, embeddings)


def _execute_training(
    args: TrainArgs,
    config: SynestheticConfig,
    color_mapper: ColorMapper,
    embeddings: npt.NDArray,
) -> None:
    print(f"Training {args.mapper_type} color projector for {config.training.epochs} epochs...")
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
