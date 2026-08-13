# Design

## Overview

Text is mapped to color and documents are represented as color distributions. A sentence
embedding is projected into CIE Lab, quantized against a fixed palette, and a document becomes a
histogram over that palette. Comparison is then a distance between distributions rather than
between 384-dimensional vectors. The approach is named for synesthesia: one modality, text,
represented through another, color.

Two independent paths share the repository and share almost no machinery:

- a **lossy semantic path**, `text → embedding → projector → Lab → palette code → histogram`, at
  an exact 1024:1 compression ratio, used for classification, retrieval, interpretation, and
  visualization;
- a **lossless codec**, `text → DEFLATE → CRC-framed pages → A4 color barcode`, which involves no
  model at all and returns the exact bytes.

## Architecture

Hexagonal architecture (ports and adapters) with domain-driven design. The dependency rules below
are enforced by `pytest-archon` tests, not by convention:

| Layer | May import from | Responsibility |
|---|---|---|
| `domain` | `shared` only | Entities, value objects, and the port interfaces (ABCs). No frameworks, no I/O |
| `application` | `domain`, `shared` | Use cases that orchestrate domain services, dependencies injected |
| `infrastructure` | `domain`, `application`, `shared` | Adapters implementing the ports |
| `interface` | all | FastAPI controllers with Pydantic DTOs, and `tyro` CLIs |
| `shared` | — | Configuration, Lab utilities, determinism, corpus and outline parsing |

### Ports and their adapters

The domain declares an interface; infrastructure supplies the implementation. This is what keeps
`scikit-learn`, `scipy`, `torch`, `POT`, `hnswlib`, `Pillow`, `matplotlib`, and the Anthropic SDK
out of the domain layer entirely.

| Port (`domain/service/`) | Adapters (`infrastructure/`) |
|---|---|
| `ColorMapper` | `pytorch_color_mapper`, `structured_pytorch_color_mapper`, `supervised_pytorch_color_mapper` |
| `DistanceCalculator` | `wasserstein`, `sliced_wasserstein`, `jensen_shannon`, `cosine_histogram` |
| `Classifier` | `color_histogram_classifier`, `tfidf_classifier`, `hnsw_classifier` |
| `Retriever` | `color_histogram_retriever`, `embedding_retriever` |
| `CompressionBaseline` | `color_vq`, `gzip`, `pq` |
| `MetricsCalculator`, `RankCorrelationCalculator` | `sklearn_metrics_calculator`, `spearman_rank_correlation_calculator` |
| `StructurePreservationEvaluator`, `InterpretabilityEvaluator` | `structure_preservation_evaluator`, `sklearn_interpretability_evaluator` |
| `FigureRenderer`, `DocumentImageRenderer`, `DataImageCodec` | `matplotlib_figure_renderer`, `pillow_document_image_renderer`, `pillow_data_image_codec` |
| `ColorCodebookFactory` | `learned_color_codebook_factory` (k-means), plus the uniform grid on the entity |
| `ConcretenessLexicon` | `brysbaert_concreteness_lexicon` (bundled, offline) |
| `TextGenerator` | `anthropic_text_generator_adapter` |
| `ColorCodebookRepository`, `DatasetRepository`, `AuthorshipScalingRepository` | file, in-memory, JSON, and the four dataset adapters |

Dependency injection uses [Lagom](https://github.com/meadsteve/lagom): components receive their
dependencies and depend on the abstractions, so a test container can substitute any adapter.

## Module map

```
colors_of_meaning/
  domain/
    model/            LabColor, ColorCodebook, ColoredDocument, EvaluationSample,
                      EvaluationResult, RetrievalEvaluation, AblationResult,
                      RateDistortionPoint, DistanceFidelity, InterpretabilityReport,
                      NarrativeArc, GroundingReport, AuthorshipReport,
                      BookSpecification, GeneratedBook
    repository/       ColorCodebookRepository, DatasetRepository,
                      AuthorshipScalingRepository
    service/          the ports listed above, plus CellGeometry and DataPayload
    health/           HealthChecker, HealthStatus
  application/
    use_case/         train, encode, compare, compress, compression comparison,
                      query by palette, evaluate, retrieval evaluate, evaluation
                      suite, distance fidelity, ablation sweep, rate-distortion
                      sweep, interpretability, grounding, narrative arc, book
                      generation, authorship report, visualization, image codecs,
                      health
  infrastructure/
    ml/               color mappers, codebook factory, compression baselines,
                      distance calculators, concreteness lexicon, scaling sweep
    dataset/          AG News, IMDB, 20 Newsgroups, document corpus, seeded sampler
    embedding/        sentence-transformers adapter
    evaluation/       classifiers, retrievers, metrics, rank correlation,
                      structure preservation, interpretability, checkpoint selection
    generation/       Anthropic text generator
    visualization/    matplotlib figures, Pillow document images, Pillow data codec
    persistence/      file codebook repository, JSON scaling manifest, in-memory
    system/           health checks
  interface/
    api/              FastAPI app, health and palette-query controllers, Pydantic DTOs
    cli/              one tyro module per command (see docs/cli.md)
  shared/             configuration, synesthetic config, Lab utilities, determinism,
                      document corpus parsing, outline parsing
```

## Design rationale

**Why CIE Lab.** It is perceptually uniform, so Euclidean distance approximates perceived color
difference. That makes a distance between colors meaningful rather than arbitrary, which is what
lets a distance between color *distributions* stand in for semantic distance. Its three axes also
map naturally onto the interpretable axes the structured mapper targets.

**Why 4,096 colors.** 16 bins per dimension is fine enough to separate meanings and coarse enough
to keep the code at 12 bits, giving the exact 1024:1 ratio against a 12,288-bit embedding. The
rate–distortion sweep measures what other budgets cost: mean ΔE falls steeply from 3 to 9 bits
and downstream accuracy peaks at 9, so 12 bits buys perceptual fidelity rather than task accuracy.

**Why a neural projector.** The mapping from semantic space to perceptual space is learned rather
than hand-specified, and the bottleneck architecture (384→128→64→3) enforces the compression. The
training objective distils embedding-space similarity structure into color-space distances; the
supervised variant additionally optimizes a classification head, which is discarded afterwards.

**Why a histogram.** A document as a bag of colors is orderless, like a bag of words, and is
already a normalized probability distribution, so it works directly with Wasserstein and
Jensen-Shannon distances. It also survives aggregation in a way a mean color does not: averaging
Lab across sentences cancels chroma through `sqrt(a² + b²)` and destroys the concreteness axis.

**Why Wasserstein-1.** Earth-Mover distance over a perceptual ground cost respects the geometry of
the palette: moving mass between two similar colors is cheap and between distant colors is
expensive, which bin-index distance would miss entirely. It is computed with a Euclidean Lab
ground cost via `ot.emd2` and no final square root; squaring the ground cost and taking a final
root would give the order-2 variant instead.

**Why a sliced proxy.** Exact EMD costs ~92 ms per call over the 4,096×4,096 cost matrix, which
makes a full test set hours of compute. Sliced-Wasserstein over the same fixed support is ~200×
faster and is a metric, restricted to the union of each pair's non-zero bins. It is trusted only
behind a fidelity gate that compares it against exact distances and fails the run if rank
correlation or accuracy drift past their thresholds.

## Evaluation strategy

**Datasets.** AG News (4-class topic), IMDB (binary sentiment), 20 Newsgroups (20-class topic),
and an authored document corpus where the author is the label and whole works are held out.

**Baselines.** TF-IDF with logistic regression, and sentence embeddings with HNSW k-NN
(`hnswlib` — FAISS-free and ARM64-safe). Both are run at matched sample budgets and matched bit
budgets, so a compression claim is never compared against a differently-scoped number.

**Metrics.** Accuracy and macro-F1 for classification; label-based recall@k and MRR for
retrieval; bits per token for compression; ΔE and MSE for distortion; Spearman rank correlation
for structure preservation and for the fidelity gate; normalized mutual information and
point-biserial or Spearman correlation for the interpretable axes.

**Falsifiability.** The interpretability claim is measured on a held-out split against a negative
control — an untrained noise projector, or a mapper never trained toward those axes — and an axis
is only asserted where its margin over the control clears a documented threshold. An axis that
fails is reported as a falsification rather than dropped.

**Determinism.** Training honours a configured seed and regenerates bit-identical weights on the
reference environment. Evaluation is deterministic given fixed weights: seeded single-thread HNSW
build, deterministic forward pass, exact EMD. Reported figures carry a stated tolerance.

## Limitations

- The 384→3 projection is lossy by construction, so fine semantic distinctions collapse into
  nearby colors. The measured structure-preservation correlation is ρ = −0.3904: real structure
  survives, and a meaningful share does not.
- Classification accuracy trails both baselines at every matched budget measured.
- The committed projector is trained on AG News; transfer to sentiment is weak and to 20-class
  topic modest. Per-dataset retraining is not done.
- Sentence splitting is simplistic and would benefit from proper tokenization.
- No online learning or incremental index updates.
- English only, via the sentence-transformers model in use.
- The A4 semantic image round-trips only approximately on the full palette, because it includes
  out-of-gamut Lab points that `lab_to_rgb` clips.

## Future work

- Strengthen the interpretable axes. All three clear their negative control on held-out data, but
  with modest absolute effect sizes (hue↔topic NMI 0.129, chroma↔concreteness 0.331,
  lightness↔sentiment 0.572 and only with its opt-in sentiment head). Making all three hold
  strongly and simultaneously is open.
- Per-dataset projectors, and a projector trained jointly across datasets.
- Remaining ablations: continuous Lab against quantized bins, and histogram against histogram
  plus temporal statistics. The distance ablation (Wasserstein against Jensen-Shannon against
  cosine) and the codebook ablation are implemented in `ablate`.
- Other perceptual modalities: sound, touch, taste.
- Hierarchical color spaces for variable-resolution encoding.
- Integration with a vector database for large-scale retrieval.
- Print-and-scan robustness for the lossless codec: fiducials, color calibration, and error
  correction are out of scope for the current version.
