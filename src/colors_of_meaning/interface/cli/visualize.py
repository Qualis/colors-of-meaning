import tyro
from dataclasses import dataclass
from typing import List, Literal

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
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
from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.tfidf_classifier import (
    TFIDFClassifier,
)
from colors_of_meaning.infrastructure.evaluation.hnsw_classifier import (
    HNSWClassifier,
)
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.visualize_codebook_use_case import (
    VisualizeCodebookUseCase,
)
from colors_of_meaning.application.use_case.visualize_documents_use_case import (
    VisualizeDocumentsUseCase,
)
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.interface.cli.codebook_loading import load_codebook


@dataclass
class VisualizeArgs:
    visualization_type: Literal["codebook", "histograms", "projection", "confusion_matrix"] = "codebook"
    config: str = "configs/base.yaml"
    dataset: Literal["ag_news", "imdb", "newsgroups"] = "ag_news"
    method: Literal["color", "tfidf", "hnsw"] = "color"
    model_path: str = "artifacts/models/projector.pth"
    codebook_name: str = "codebook_4096"
    k_neighbors: int = 5
    output_dir: str = "reports/figures"
    max_samples: int = 500


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    dataset_adapters = {
        "ag_news": (AGNewsDatasetAdapter, "Loading AG News dataset..."),
        "imdb": (IMDBDatasetAdapter, "Loading IMDB dataset..."),
        "newsgroups": (NewsgroupsDatasetAdapter, "Loading 20 Newsgroups dataset..."),
    }
    adapter_class, message = dataset_adapters[dataset_name]
    print(message)
    return adapter_class()


def _encode_samples(
    samples: List[EvaluationSample],
    config: SynestheticConfig,
    model_path: str,
    codebook_name: str,
) -> List[ColoredDocument]:
    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
    )
    color_mapper.load_weights(model_path)
    codebook = load_codebook(codebook_name)
    quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
    encode_use_case = EncodeDocumentUseCase(quantized_mapper)

    documents: List[ColoredDocument] = []
    for i, sample in enumerate(samples):
        embeddings = embedding_adapter.encode_document_sentences(sample.text)
        doc = encode_use_case.execute(embeddings, document_id=f"viz_{i}")
        documents.append(doc)
        if (i + 1) % 100 == 0:
            print(f"  Encoded {i + 1}/{len(samples)} documents")
    return documents


def _run_codebook_visualization(args: VisualizeArgs) -> None:
    renderer = MatplotlibFigureRenderer()
    codebook_repo = FileColorCodebookRepository()
    use_case = VisualizeCodebookUseCase(codebook_repo, renderer)
    output_path = f"{args.output_dir}/codebook_palette.png"
    print(f"Rendering codebook palette to {output_path}...")
    use_case.execute(args.codebook_name, output_path)
    print("Done.")


def _run_histograms_visualization(args: VisualizeArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    dataset_repo = _setup_dataset(args.dataset)
    samples = dataset_repo.get_samples(split="test", max_samples=args.max_samples)
    label_names = dataset_repo.get_label_names()

    print(f"Encoding {len(samples)} documents...")
    documents = _encode_samples(samples, config, args.model_path, args.codebook_name)
    labels = [s.label for s in samples]

    renderer = MatplotlibFigureRenderer()
    use_case = VisualizeDocumentsUseCase(renderer)
    output_path = f"{args.output_dir}/document_histograms.png"
    print(f"Rendering document histograms to {output_path}...")
    use_case.execute_histograms(documents, labels, label_names, output_path)
    print("Done.")


def _run_projection_visualization(args: VisualizeArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    dataset_repo = _setup_dataset(args.dataset)
    samples = dataset_repo.get_samples(split="test", max_samples=args.max_samples)
    label_names = dataset_repo.get_label_names()

    print(f"Encoding {len(samples)} documents...")
    documents = _encode_samples(samples, config, args.model_path, args.codebook_name)
    labels = [s.label for s in samples]

    renderer = MatplotlibFigureRenderer()
    use_case = VisualizeDocumentsUseCase(renderer)
    output_path = f"{args.output_dir}/tsne_projection.png"
    print(f"Rendering t-SNE projection to {output_path}...")
    use_case.execute_projection(documents, labels, label_names, output_path)
    print("Done.")


def _run_confusion_matrix_visualization(args: VisualizeArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    dataset_repo = _setup_dataset(args.dataset)
    label_names = dataset_repo.get_label_names()

    train_samples = dataset_repo.get_samples(split="train", max_samples=args.max_samples)
    test_samples = dataset_repo.get_samples(split="test", max_samples=args.max_samples)

    classifier = _create_classifier(args, config)
    print(f"Fitting {args.method} classifier on {len(train_samples)} samples...")
    classifier.fit(train_samples)
    print(f"Predicting on {len(test_samples)} samples...")
    y_true = [s.label for s in test_samples]
    y_pred = classifier.predict(test_samples)

    renderer = MatplotlibFigureRenderer()
    use_case = VisualizeDocumentsUseCase(renderer)
    output_path = f"{args.output_dir}/confusion_matrix_{args.method}.png"
    print(f"Rendering confusion matrix to {output_path}...")
    use_case.execute_confusion_matrix(y_true, y_pred, label_names, output_path)
    print("Done.")


def _create_classifier(args: VisualizeArgs, config: SynestheticConfig) -> Classifier:
    if args.method == "color":
        embedding_adapter = SentenceEmbeddingAdapter()
        color_mapper = PyTorchColorMapper(
            input_dim=config.projector.embedding_dim,
            hidden_dim_1=config.projector.hidden_dim_1,
            hidden_dim_2=config.projector.hidden_dim_2,
            dropout_rate=config.projector.dropout_rate,
            device=config.training.device,
        )
        color_mapper.load_weights(args.model_path)
        codebook = load_codebook(args.codebook_name)
        quantized_mapper = QuantizedColorMapper(color_mapper, codebook)
        encode_use_case = EncodeDocumentUseCase(quantized_mapper)
        distance_calculator = WassersteinDistanceCalculator(
            codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg
        )
        return ColorHistogramClassifier(embedding_adapter, encode_use_case, distance_calculator, k=args.k_neighbors)
    elif args.method == "tfidf":
        return TFIDFClassifier()
    elif args.method == "hnsw":
        return HNSWClassifier(SentenceEmbeddingAdapter(), k=args.k_neighbors)
    raise ValueError(f"Unknown method: {args.method}")


def main(args: VisualizeArgs) -> None:
    visualization_handlers = {
        "codebook": _run_codebook_visualization,
        "histograms": _run_histograms_visualization,
        "projection": _run_projection_visualization,
        "confusion_matrix": _run_confusion_matrix_visualization,
    }
    handler = visualization_handlers.get(args.visualization_type)
    if handler is None:
        raise ValueError(f"Unknown visualization type: {args.visualization_type}")
    handler(args)


if __name__ == "__main__":
    main(tyro.cli(VisualizeArgs))
