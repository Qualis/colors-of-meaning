# Colors of Meaning

[![Build Builder](https://github.com/Qualis/colors-of-meaning/actions/workflows/builder.yml/badge.svg)](https://github.com/Qualis/colors-of-meaning/actions/workflows/builder.yml)
[![Build Development](https://github.com/Qualis/colors-of-meaning/actions/workflows/development.yml/badge.svg)](https://github.com/Qualis/colors-of-meaning/actions/workflows/development.yml)
[![Build Service](https://github.com/Qualis/colors-of-meaning/actions/workflows/service.yml/badge.svg)](https://github.com/Qualis/colors-of-meaning/actions/workflows/service.yml)

**Machine synesthesia for semantic compression and retrieval**

## Project Overview

This project implements the **Colors of Meaning** experiment described in [the research article](https://www.qual.is/posts/colors-of-meaning), which explores a novel approach to semantic understanding through synesthesia.

The core idea is to map semantic embeddings into perceptual color space (CIE Lab), creating a synesthetic representation where documents are color distributions rather than high-dimensional vectors. Rather than working with abstract 384-dimensional embeddings, we translate semantic meaning into the intuitive language of color: hue, lightness, and saturation. This approach achieves extreme compression (1000x+) while maintaining interpretable semantic structure, opening new possibilities for understanding how meaning relates to perception.

### Key Features

- **Semantic Color Mapping**: Neural projector maps 384-dim embeddings to 3-dim Lab color space
- **Structured Color Mapping**: Self-supervised projector variant where hue encodes semantic clusters, lightness encodes sentiment, and chroma encodes concreteness
- **Extreme Compression**: 12,288 bits/sentence to 12 bits (4,096-color palette)
- **Compression Baselines**: Compare color VQ against gzip and Product Quantization at matched bit budgets
- **Interpretable Structure**: Hue clusters synonyms, lightness encodes sentiment, saturation distinguishes abstract/concrete
- **Query by Color Palette**: Retrieve documents by specifying Lab colors with weights (CLI and API)
- **Fast Retrieval**: Wasserstein distance on color histograms for document comparison
- **Portable Implementation**: ARM64-safe using hnswlib (no FAISS segfaults)

### Current Performance (AG News Dataset)

| Method | Accuracy | Macro F1 | Notes |
|--------|----------|----------|-------|
| TF-IDF Baseline | 90.63% | 90.61% | Traditional bag-of-words |
| HNSW Baseline | 91.99% | 91.97% | Semantic embeddings + k-NN |
| Color Method | TBD | TBD | Train model first |

### Project Status

- ✅ **Phase 1**: Core infrastructure (color mapping, encoding, compression)
- ✅ **Phase 2**: Evaluation pipeline and baselines
- ✅ **Phase 3**: Visualizations (codebook palette, histograms, t-SNE projections, confusion matrices)
- ✅ **Phase 4**: Structured color mapping, compression comparison baselines, query by color palette

**Quality Gates**: All passing (flake8, black, bandit, semgrep, xenon, mypy, pip-audit, 100% coverage, 680 tests)

## Quick Start

### Evaluate Baselines (No Training Required)

```bash
# TF-IDF baseline (fast, ~12 seconds)
tox -e eval -- --dataset ag_news --method tfidf

# HNSW embedding baseline (slower, ~30 minutes)
tox -e eval -- --dataset ag_news --method hnsw --k-neighbors 5
```

### Train and Evaluate Color Method

```bash
# Step 1: Train the unconstrained color mapper
tox -e train -- --config configs/base.yaml

# Step 2: Evaluate color method
tox -e eval -- --dataset ag_news --method color
```

### Train the Structured Color Mapper

The structured mapper uses explicit heads for hue (semantic clusters), lightness (sentiment), and chroma (concreteness), trained with a multi-objective self-supervised loss.

```bash
tox -e train -- --config configs/structured.yaml --mapper-type structured
```

### Query by Color Palette

Retrieve documents matching a color distribution specified as Lab colors with weights:

```bash
tox -e query -- \
  --palette-json '[{"l": 75, "a": 20, "b": -10, "weight": 1.0}]' \
  --codebook-name codebook_4096 \
  --encoded-documents artifacts/encoded/test_documents.pkl \
  --k 5
```

### Compare Compression Baselines

Compare color VQ compression against gzip and Product Quantization:

```bash
tox -e compress -- --compare-baselines --embeddings-path artifacts/encoded/test_embeddings.npy
```

### Generate Visualizations

```bash
# Codebook palette
tox -e visualize -- --visualization-type codebook

# Document color histograms
tox -e visualize -- --visualization-type histograms --dataset ag_news --max-samples 50

# t-SNE projection of color distributions
tox -e visualize -- --visualization-type projection --dataset ag_news --max-samples 200

# Confusion matrix
tox -e visualize -- --visualization-type confusion_matrix --dataset ag_news --method color
```

## CLI Reference

All CLI tools are available via tox environments or as Python modules directly.

### Training

Train the embedding-to-color projector:

```bash
# Unconstrained projector (default)
tox -e train -- --config configs/base.yaml

# Structured projector (hue=semantics, lightness=sentiment, chroma=concreteness)
tox -e train -- --config configs/structured.yaml --mapper-type structured
```

**Arguments**:
- `--config`: Configuration YAML file (default: `configs/base.yaml`)
- `--dataset-path`: Path to training text file (default: `data/train.txt`)
- `--output-model`: Output path for trained model (default: `artifacts/models/projector.pth`)
- `--output-codebook`: Output name for color codebook (default: `codebook_4096`)
- `--mapper-type`: Projector variant to train: `unconstrained` (default) or `structured`

The structured mapper trains with a multi-objective loss combining angular hue loss, MSE lightness loss, and MSE chroma loss, weighted by alpha/beta/gamma from the config. Self-supervised targets are derived from the embeddings themselves: k-means clusters for hue, mean activation for lightness, and variance for chroma.

### Encoding

Encode documents to color histograms:

```bash
tox -e encode -- --config configs/base.yaml --split test
```

**Arguments**:
- `--config`: Configuration YAML file (default: `configs/base.yaml`)
- `--split`: Dataset split (`train`, `test`)
- `--model-path`: Path to trained model (default: `artifacts/models/projector.pth`)
- `--codebook-name`: Color codebook name (default: `codebook_4096`)
- `--output-path`: Output path for encoded documents (default: `artifacts/encoded/test_documents.pkl`)

### Comparison

Compare documents via color distance:

```bash
tox -e compare -- --config configs/base.yaml --k 5
```

**Arguments**:
- `--config`: Configuration YAML file (default: `configs/base.yaml`)
- `--encoded-documents`: Path to encoded documents (default: `artifacts/encoded/test_documents.pkl`)
- `--k`: Number of nearest neighbors to find (default: 5)
- `--query-index`: Index of query document (default: 0)

### Compression

Analyze compression ratios and compare against baselines:

```bash
# Color VQ compression analysis
tox -e compress -- --config configs/base.yaml --method vq

# Compare against gzip and Product Quantization baselines
tox -e compress -- --compare-baselines --embeddings-path artifacts/encoded/test_embeddings.npy
```

**Arguments**:
- `--config`: Configuration YAML file (default: `configs/base.yaml`)
- `--encoded-documents`: Path to encoded documents (default: `artifacts/encoded/test_documents.pkl`)
- `--method`: Compression method (default: `vq`)
- `--compare-baselines`: Run gzip and Product Quantization baselines instead of VQ analysis (default: `false`)
- `--embeddings-path`: Path to raw embeddings for baseline comparison (default: `artifacts/encoded/test_embeddings.npy`)

When `--compare-baselines` is enabled, the tool loads raw float32 embeddings and runs gzip (lossless) and Product Quantization (M=48 subspaces, k=256 centroids) baselines, reporting compression ratio, bits per token, and reconstruction error (MSE) for each method.

### Evaluation

Evaluate classification performance with different methods:

```bash
# Evaluate on AG News (4-class topic classification)
tox -e eval -- --dataset ag_news --method tfidf

# Evaluate on IMDB (binary sentiment classification)
tox -e eval -- --dataset imdb --method tfidf

# Evaluate on 20 Newsgroups (20-class topic classification)
tox -e eval -- --dataset newsgroups --method hnsw --k-neighbors 10
```

**Arguments**:
- `--dataset`: Dataset to evaluate on (`ag_news`, `imdb`, `newsgroups`)
- `--method`: Classification method (`tfidf`, `hnsw`, `color`)
- `--k-neighbors`: Number of neighbors for k-NN methods (default: 5)
- `--config`: Configuration file (default: `configs/base.yaml`)
- `--model-path`: Path to trained projector model (default: `artifacts/models/projector.pth`)
- `--codebook-path`: Color codebook name (default: `codebook_4096`)

### Visualization

Generate visualizations of the color mapping pipeline. Output is saved to `reports/figures/`.

```bash
# Codebook palette (4,096-color swatch grid)
tox -e visualize -- --visualization-type codebook

# Document color histograms by class
tox -e visualize -- --visualization-type histograms --dataset ag_news --max-samples 50

# t-SNE projection of color distributions
tox -e visualize -- --visualization-type projection --dataset ag_news --max-samples 200

# Confusion matrix for a classification method
tox -e visualize -- --visualization-type confusion_matrix --dataset ag_news --method color
```

**Arguments**:
- `--visualization-type`: Type of visualization (`codebook`, `histograms`, `projection`, `confusion_matrix`)
- `--config`: Configuration YAML file (default: `configs/base.yaml`)
- `--dataset`: Dataset to visualize (`ag_news`, `imdb`, `newsgroups`)
- `--method`: Classification method for confusion matrix (`color`, `tfidf`, `hnsw`)
- `--model-path`: Path to trained model (default: `artifacts/models/projector.pth`)
- `--codebook-name`: Color codebook name (default: `codebook_4096`)
- `--k-neighbors`: Number of neighbors for k-NN methods (default: 5)
- `--output-dir`: Output directory for figures (default: `reports/figures`)
- `--max-samples`: Maximum number of samples to process (default: 500)

### Query by Color Palette

Retrieve documents matching a specified color distribution:

```bash
tox -e query -- \
  --palette-json '[{"l": 75, "a": 20, "b": -10, "weight": 1.0}, {"l": 30, "a": -5, "b": 40, "weight": 0.5}]' \
  --codebook-name codebook_4096 \
  --k 5
```

**Arguments**:
- `--palette-json`: JSON array of Lab colors with optional weights (default: `[{"l": 50, "a": 0, "b": 0, "weight": 1.0}]`)
- `--encoded-documents`: Path to encoded document corpus (default: `artifacts/encoded/test_documents.pkl`)
- `--codebook-name`: Color codebook name (default: `codebook_4096`)
- `--k`: Number of nearest neighbors to return (default: 5)

Each color in the palette is specified with CIE Lab coordinates: `l` (lightness, 0-100), `a` (green-red, -128 to 127), `b` (blue-yellow, -128 to 127), and an optional `weight` (default: 1.0). The system quantizes the palette into a color histogram and finds the closest documents by Wasserstein distance.

An API endpoint is also available at `POST /query/palette` when the service is running, accepting the same palette specification as a JSON request body.

## Development

### Tested Configuration

* `vagrant`: 2.4.3
* `ansible`: 2.18.3
* `colima`: 0.8.1
* `docker`: 28.0.2

## Instructions

```
colima start --cpu 2 --memory 2 --vm-type=vz --mount-type=virtiofs --dns 8.8.8.8
vagrant up
vagrant provision
vagrant ssh
```

### Tox Environments

| Command | Description |
|---------|-------------|
| `tox` | Run all quality gates and tests |
| `tox -e format` | Format code with black |
| `tox -e watch` | Watch for changes and run tests |
| `tox -e train -- [args]` | Train the embedding-to-color projector |
| `tox -e encode -- [args]` | Encode documents to color histograms |
| `tox -e compare -- [args]` | Compare documents via color distance |
| `tox -e compress -- [args]` | Analyze compression ratios and baselines |
| `tox -e eval -- [args]` | Evaluate classification performance |
| `tox -e visualize -- [args]` | Generate visualizations |
| `tox -e query -- [args]` | Query documents by color palette |

### Test

#### Shell Script

```
shellcheck -x ./*.sh bin/*
```

#### Ansible

```
ansible-lint -p infrastructure/ansible/playbook-*.yml
```

#### Python

```
tox
```

##### Watch

```
tox -e watch
```

### Code Formatter

```
tox -e format
```

### Run

```
./bin/run-local -c
```

## Architecture

This project follows Hexagonal Architecture (Ports and Adapters) combined with Domain-Driven Design principles:

- **Domain Layer**: Pure business logic, entities, repository interfaces (ABCs), and service interfaces
- **Application Layer**: Use case orchestration, no direct framework or IO dependencies
- **Infrastructure Layer**: Technical adapters implementing domain interfaces (ML, datasets, persistence, visualization)
- **Interface Layer**: FastAPI controllers, Pydantic DTOs, and CLI tools
- **Shared Layer**: Cross-cutting concerns (configuration, Lab utilities)

For comprehensive architectural rules, coding standards, testing requirements, and development guidelines, see [`.claude/CLAUDE.md`](.claude/CLAUDE.md).

### Project Structure Overview

```
colors_of_meaning/

 application/                       # Use cases
    use_case/
        train_color_mapping_use_case.py
        encode_document_use_case.py
        compare_documents_use_case.py
        compress_document_use_case.py
        compression_comparison_use_case.py
        query_by_palette_use_case.py
        evaluate_use_case.py
        visualize_codebook_use_case.py
        visualize_documents_use_case.py

 domain/                            # Business logic
    model/
        lab_color.py
        color_codebook.py
        colored_document.py
        evaluation_sample.py
        evaluation_result.py
    repository/
        color_codebook_repository.py
        dataset_repository.py
    service/
        color_mapper.py
        compression_baseline.py
        distance_calculator.py
        classifier.py
        retriever.py
        metrics_calculator.py
        figure_renderer.py

 infrastructure/                    # Adapters, drivers
    dataset/
        ag_news_dataset_adapter.py
        imdb_dataset_adapter.py
        newsgroups_dataset_adapter.py
    embedding/
        sentence_embedding_adapter.py
    evaluation/
        color_histogram_classifier.py
        hnsw_classifier.py
        tfidf_classifier.py
        sklearn_metrics_calculator.py
    ml/
        pytorch_color_mapper.py
        structured_lab_projector_network.py
        structured_pytorch_color_mapper.py
        gzip_compression_baseline.py
        pq_compression_baseline.py
        wasserstein_distance_calculator.py
        jensen_shannon_distance_calculator.py
    persistence/
        file_color_codebook_repository.py
        in_memory/
    visualization/
        matplotlib_figure_renderer.py
    security/
        basic_authentication.py
    system/
        health_checker.py

 interface/                         # APIs and CLI
    api/
        main.py
        controller/
            health_controller.py
            query_controller.py
        data_transfer_object/
            palette_query_dto.py
    cli/
        train.py
        encode.py
        compare.py
        compress.py
        eval.py
        visualize.py
        query.py

 shared/                            # Cross-cutting concerns
    configuration.py
    lab_utils.py
    synesthetic_config.py
```

### Design Principles

#### Domain

- `model/` contain pure domain entities (e.g., `LabColor`, `ColorCodebook`, `ColoredDocument`)
- `repository/` define interfaces for persistence (abstract base classes)
- `service/` define domain service interfaces (e.g., `ColorMapper`, `DistanceCalculator`, `Classifier`)

#### Application

- Use cases (`_use_case.py`) orchestrate logic/delegate
- Should not directly depend on e.g. `FastAPI`, database, or file system

#### Infrastructure

- Implements interfaces from `domain.repository` and `domain.service`
- `ml/` contains PyTorch color mappers (unconstrained and structured), compression baselines (gzip, Product Quantization), and distance calculators
- `evaluation/` provides classifier implementations (TF-IDF, HNSW, color histogram)
- `dataset/` adapts external datasets (AG News, IMDB, 20 Newsgroups)
- `embedding/` wraps sentence-transformers for text encoding
- `visualization/` renders figures using matplotlib (codebook palettes, histograms, projections, confusion matrices)
- `security/` provides token-based authentication
- `system/` contains diagnostics (e.g., disk, database checks)

#### Interface

- `api/controller/` expose `FastAPI` routes and depend on use cases (health, coconut, query by palette)
- `api/data_transfer_object/` define `Pydantic` models for request/response shaping (coconut DTOs, palette query DTOs)
- `cli/` provides command-line tools for training, encoding, evaluation, compression, visualization, and query by palette

#### Shared

- `lab_utils.py` provides RGB/Lab conversion and Delta E distance
- `synesthetic_config.py` loads YAML-based experiment configuration (projector, codebook, training, distance, dataset, structured mapper settings)
- `configuration.py` loads application settings

## Engineering Practices

- 100% test coverage (680 tests)
- Dependency injection using [Lagom](https://github.com/meadsteve/lagom)
- Structured logging with correlation IDs
- Consumer-Driven Contract Testing (CDCT)
- Security scanning and static analysis (flake8, black, bandit, xenon, mypy, semgrep, pip-audit)
- Docker containerization with multi-arch builds (amd64, arm64)
- Infrastructure as Code (Ansible)

## Claude Code Skills

This project includes [Claude Code skills](.claude/skills/README.md) for development assistance:

- **Hexagonal Architecture Scaffolder** - Scaffold features across all architecture layers
- **Test Generator** - Generate tests following the one-assertion-per-test rule
- **Self-Documenting Refactor** - Enforce the no-comments policy
- **Property-Based Testing** - Identify and implement property-based tests
- **Test-Driven Development** - Red-Green-Refactor cycle enforcement
- **Systematic Debugging** - Root cause investigation methodology
- **Verification Before Completion** - Evidence-based completion verification

## Contributing

When contributing to this project:

1. Review [`.claude/CLAUDE.md`](.claude/CLAUDE.md) for detailed coding standards and architectural rules
2. Ensure all tests pass: `tox`
3. Format code before committing: `tox -e format`
4. Maintain 100% test coverage
5. Follow the one-assertion-per-test rule
6. Respect layer boundaries (see [`.claude/CLAUDE.md`](.claude/CLAUDE.md))
