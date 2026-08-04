import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional

import tyro

from colors_of_meaning.application.use_case.authorship_report_use_case import (
    AuthorshipReportUseCase,
)
from colors_of_meaning.application.use_case.encode_document_use_case import (
    EncodeDocumentUseCase,
)
from colors_of_meaning.application.use_case.evaluate_use_case import EvaluateUseCase
from colors_of_meaning.domain.model.authorship_report import (
    AuthorshipCorpus,
    AuthorshipReport,
    AuthorshipScalingPoint,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.service.classifier import Classifier
from colors_of_meaning.domain.service.color_mapper import ColorMapper, QuantizedColorMapper
from colors_of_meaning.infrastructure.dataset.document_corpus_dataset_adapter import (
    DocumentCorpusDatasetAdapter,
)
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.evaluation.color_histogram_classifier import (
    ColorHistogramClassifier,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_metrics_calculator import (
    SklearnMetricsCalculator,
)
from colors_of_meaning.infrastructure.evaluation.tfidf_classifier import TFIDFClassifier
from colors_of_meaning.infrastructure.ml.authorship_scaling_sweep import (
    AuthorshipScalingSweep,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.persistence.json_authorship_scaling_repository import (
    JsonAuthorshipScalingRepository,
)
from colors_of_meaning.interface.cli.codebook_loading import load_codebook
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

logger = logging.getLogger(__name__)

COLOR_BITS_PER_TOKEN = 12.0


@dataclass
class AuthorshipArgs:
    config: str = "configs/documents.yaml"
    model_path: str = "artifacts/models/projector_documents_valsel.pth"
    codebook_name: str = "codebook_documents_valsel"
    mapper_type: str = "supervised"
    documents_dir: str = "./documents"
    min_paragraph_chars: int = 200
    paragraphs_per_work: int = 60
    split_strategy: Literal["work", "paragraph"] = "work"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    distance: str = "jensen_shannon"
    k_neighbors: int = 5
    methods: List[str] = field(default_factory=lambda: ["color", "tfidf"])
    refresh_scaling: bool = False
    scaling_seeds: int = 8
    scaling_caps: List[int] = field(default_factory=lambda: [60, 150, 300])
    scaling_scratch_dir: str = "artifacts/scaling_sweep"
    scaling_workers: int = 1
    scaling_manifest: str = "reports/data/authorship_scaling.json"
    output_path: str = "reports/documents_authorship.md"


def _build_document_corpus(args: AuthorshipArgs, cap: int) -> DocumentCorpusDatasetAdapter:
    return DocumentCorpusDatasetAdapter(
        documents_dir=args.documents_dir,
        min_paragraph_chars=args.min_paragraph_chars,
        paragraphs_per_work=cap,
        split_strategy=args.split_strategy,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )


def _split_count(adapter: DocumentCorpusDatasetAdapter, split: str, seed: int) -> int:
    return len(adapter.get_samples(split=split, seed=seed))


def _corpus_summary(adapter: DocumentCorpusDatasetAdapter, seed: int) -> AuthorshipCorpus:
    split_counts = [(split, _split_count(adapter, split, seed)) for split in ("train", "validation", "test")]
    return AuthorshipCorpus(authors_works=adapter.works_per_author(), split_counts=split_counts)


def _build_color_mapper(args: AuthorshipArgs, config: SynestheticConfig) -> ColorMapper:
    color_mapper = create_color_mapper(args.mapper_type, config)
    color_mapper.load_weights(args.model_path)
    return color_mapper


def _build_color_classifier(
    args: AuthorshipArgs,
    config: SynestheticConfig,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    codebook: ColorCodebook,
) -> ColorHistogramClassifier:
    encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(color_mapper, codebook))
    distance_calculator = JensenShannonDistanceCalculator(smoothing_epsilon=config.distance.smoothing_epsilon)
    return ColorHistogramClassifier(embedding_adapter, encode_use_case, distance_calculator, k=args.k_neighbors)


def _build_classifier(
    method: str,
    args: AuthorshipArgs,
    config: SynestheticConfig,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    codebook: ColorCodebook,
) -> Classifier:
    if method == "color":
        return _build_color_classifier(args, config, embedding_adapter, color_mapper, codebook)
    if method == "tfidf":
        return TFIDFClassifier()
    raise ValueError(f"Unknown method: {method}")


def _build_evaluate_factory(
    args: AuthorshipArgs,
    config: SynestheticConfig,
    reference_adapter: DocumentCorpusDatasetAdapter,
    embedding_adapter: SentenceEmbeddingAdapter,
    color_mapper: ColorMapper,
    codebook: ColorCodebook,
) -> Callable[[str], EvaluateUseCase]:
    def build(method: str) -> EvaluateUseCase:
        classifier = _build_classifier(method, args, config, embedding_adapter, color_mapper, codebook)
        return EvaluateUseCase(classifier, SklearnMetricsCalculator(), reference_adapter)

    return build


def _bits_per_token(methods: List[str]) -> Dict[str, Optional[float]]:
    return {method: (COLOR_BITS_PER_TOKEN if method == "color" else None) for method in methods}


def _build_scaling_sweep(
    args: AuthorshipArgs,
    config: SynestheticConfig,
    reference_adapter: DocumentCorpusDatasetAdapter,
    embedding_adapter: SentenceEmbeddingAdapter,
) -> Callable[[], List[AuthorshipScalingPoint]]:
    sweep = AuthorshipScalingSweep(
        config=config,
        reference_adapter=reference_adapter,
        embedding_adapter=embedding_adapter,
        documents_dir=args.documents_dir,
        min_paragraph_chars=args.min_paragraph_chars,
        split_strategy=args.split_strategy,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        distance_smoothing=config.distance.smoothing_epsilon,
        k_neighbors=args.k_neighbors,
        caps=args.scaling_caps,
        seeds=args.scaling_seeds,
        scratch_dir=args.scaling_scratch_dir,
        scaling_workers=args.scaling_workers,
    )
    return sweep.run


def _format_bits(bits_per_token: Optional[float]) -> str:
    return "—" if bits_per_token is None else f"{bits_per_token:.0f}"


def _authors_works_rows(corpus: AuthorshipCorpus) -> List[str]:
    rows = ["| author | works |", "|---|---|"]
    for author, works in corpus.authors_works:
        rows.append(f"| {author} | {works} |")
    return rows


def _split_rows(corpus: AuthorshipCorpus) -> List[str]:
    rows = ["| split | paragraphs |", "|---|---|"]
    for split, count in corpus.split_counts:
        rows.append(f"| {split} | {count} |")
    return rows


def _held_out_rows(report: AuthorshipReport) -> List[str]:
    rows = ["| method | bits/token | test accuracy | macro-F1 |", "|---|---|---|---|"]
    for row in report.held_out:
        rows.append(f"| {row.method} | {_format_bits(row.bits_per_token)} | {row.accuracy:.4f} | {row.macro_f1:.4f} |")
    rows.append(f"| chance | — | {report.chance:.4f} | — |")
    return rows


def _scaling_rows(report: AuthorshipReport) -> List[str]:
    rows = ["| paragraphs_per_work | train paragraphs | test accuracy (mean ± std) | × chance |", "|---|---|---|---|"]
    for point in report.scaling:
        rows.append(
            f"| {point.paragraphs_per_work} | {point.train_paragraphs} | "
            f"{point.mean_accuracy:.3f} ± {point.std_accuracy:.3f} | {point.times_chance(report.chance):.2f}× |"
        )
    return rows


def _provenance_line() -> str:
    import numpy
    import sklearn  # type: ignore

    return f"Library versions: numpy {numpy.__version__}, scikit-learn {sklearn.__version__}."


def _reproduce_command(args: AuthorshipArgs) -> str:
    return (
        f"tox -e authorship -- --config {args.config} --model-path {args.model_path} "
        f"--codebook-name {args.codebook_name} --mapper-type {args.mapper_type} "
        f"--documents-dir {args.documents_dir} --paragraphs-per-work {args.paragraphs_per_work} "
        f"--distance {args.distance}"
    )


def _visualize_command(args: AuthorshipArgs) -> str:
    return (
        f"tox -e visualize_documents -- --config {args.config} --model-path {args.model_path} "
        f"--codebook-name {args.codebook_name} --mapper-type {args.mapper_type}"
    )


def _figures_section(args: AuthorshipArgs) -> List[str]:
    return [
        "## The texts as colours",
        "",
        "Each book's leading paragraphs, encoded to a colour distribution by the trained projector and",
        "projected to 2-D — authors settle into colour neighbourhoods:",
        "",
        "![Document colour t-SNE](figures/documents_color_tsne.png)",
        "",
        "Per-author colour signatures (the dominant palette colours for each author):",
        "",
        "![Document colour signatures](figures/documents_color_signatures.png)",
        "",
        "### A4 colour image per book",
        "",
        "Each work is rendered as an A4 colours-of-meaning sheet — horizontal bands of the book's palette",
        "colours, sized by how often the projector maps the book's sentences to each colour (the `signature`",
        "layout over the book's leading paragraphs). The contact sheet below tiles all of them, ordered by",
        "author, so an author's works sit together (the per-book sheets are the `figures/a4/`",
        "`<author>__<work>.png` files):",
        "",
        "![Per-book A4 colour signatures](figures/documents_a4_gallery.png)",
        "",
        "These figures regenerate deterministically (byte-identical across runs) with:",
        "",
        "```bash",
        _visualize_command(args),
        "```",
        "",
        "### Lossless A4 colour-barcode representation",
        "",
        "The signatures above are a *lossy, semantic* rendering. Separately, the lossless codec",
        "(`encode_lossless` / `decode_lossless`) stores each book's **exact** text as printable A4",
        "colour-barcode page(s) that decode back byte-for-byte. Those barcode images are dense data and",
        "are **not** committed — they are git-ignored, regenerable local artifacts.",
        "",
    ]


def _report_lines(args: AuthorshipArgs, report: AuthorshipReport) -> List[str]:
    corpus = report.corpus
    return [
        "# Authorship by colour: training, evaluating and testing against `documents/`",
        "",
        "This report trains the semantic-colour projector on a real authored-document corpus and",
        "measures how much authorship signal survives compressing each paragraph's meaning into a",
        "distribution over a 4,096-colour palette (12 bits/token). It is generated from the local,",
        "git-ignored `./documents/` corpus and is **not** CI-reproducible; regenerate it locally with",
        "the command at the end. The computed tables below are machine-written; only the prose is fixed.",
        "",
        _provenance_line(),
        "",
        "## Corpus",
        "",
        f"`documents/<author>/<work>.txt` — {corpus.author_count()} authors, {corpus.total_works()} works. "
        "The author is the class label; works are stripped of Gutenberg boilerplate, split into paragraphs,",
        "globally de-duplicated, and partitioned by a work-level three-way holdout.",
        "",
        *_authors_works_rows(corpus),
        "",
        f"Per-split paragraph counts at the `paragraphs_per_work` cap of {args.paragraphs_per_work}:",
        "",
        *_split_rows(corpus),
        "",
        "## Held-out test results",
        "",
        f"Authorship on the held-out test works (chance = 1/{corpus.author_count()} = {report.chance:.4f}). "
        "The colour method is a supervised author-contrastive projector; TF-IDF keeps every word.",
        "",
        *_held_out_rows(report),
        "",
        *_figures_section(args),
        "## Data scaling: does more training data help?",
        "",
        "The `paragraphs_per_work` cap controls how much of each book enters training. Training the",
        "val-selected supervised projector at increasing caps and measuring held-out authorship accuracy",
        "on the same fixed test set (multiple seeds each; mean ± population std):",
        "",
        *_scaling_rows(report),
        "",
        "## Reproduce (local; requires `./documents/`)",
        "",
        "```bash",
        _reproduce_command(args),
        "```",
        "",
    ]


def _write_report(args: AuthorshipArgs, report: AuthorshipReport) -> None:
    destination = Path(args.output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, report)), encoding="utf-8")
    print(f"Saved {args.output_path}")


def _print_tables(report: AuthorshipReport) -> None:
    print("\n=== Held-out authorship ===")
    for line in _held_out_rows(report):
        print(line)


def _log_startup(args: AuthorshipArgs, config: SynestheticConfig, corpus: AuthorshipCorpus) -> None:
    logger.info(
        "Generating authorship report",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "authors": corpus.author_count(),
            "works": corpus.total_works(),
            "split_counts": dict(corpus.split_counts),
            "refresh_scaling": args.refresh_scaling,
            "seed": config.training.seed,
        },
    )


def main(args: AuthorshipArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    reference_adapter = _build_document_corpus(args, args.paragraphs_per_work)
    corpus = _corpus_summary(reference_adapter, config.training.seed)
    if not corpus.authors_works:
        raise ValueError(f"No qualifying authors found in {args.documents_dir}")

    embedding_adapter = SentenceEmbeddingAdapter()
    color_mapper = _build_color_mapper(args, config)
    codebook = load_codebook(args.codebook_name)
    _log_startup(args, config, corpus)

    factory = _build_evaluate_factory(args, config, reference_adapter, embedding_adapter, color_mapper, codebook)
    scaling_sweep = _build_scaling_sweep(args, config, reference_adapter, embedding_adapter)
    use_case = AuthorshipReportUseCase(factory, JsonAuthorshipScalingRepository(args.scaling_manifest))

    report = use_case.execute(
        corpus=corpus,
        methods=args.methods,
        bits_per_token=_bits_per_token(args.methods),
        max_samples=None,
        seed=config.training.seed,
        refresh_scaling=args.refresh_scaling,
        scaling_sweep=scaling_sweep,
    )
    _print_tables(report)
    _write_report(args, report)


if __name__ == "__main__":
    main(tyro.cli(AuthorshipArgs))
