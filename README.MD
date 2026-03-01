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
- **Extreme Compression**: 12,288 bits/sentence → 12 bits (4,096-color palette)
- **Interpretable Structure**: Hue clusters synonyms, lightness encodes sentiment, saturation distinguishes abstract/concrete
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
- ⏳ **Phase 3**: Visualizations and interpretability analysis

**Test Coverage**: 100% (476 tests passing)
**Quality Gates**: All passing (flake8, black, bandit, semgrep, xenon, mypy, pip-audit)

## Quick Start

### Evaluate Baselines (No Training Required)

```bash
# TF-IDF baseline (fast, ~12 seconds)
python -m colors_of_meaning.interface.cli.eval \
  --dataset ag_news \
  --method tfidf

# HNSW embedding baseline (slower, ~30 minutes)
python -m colors_of_meaning.interface.cli.eval \
  --dataset ag_news \
  --method hnsw \
  --k-neighbors 5
```

### Train and Evaluate Color Method

```bash
# Step 1: Train the color mapper
python -m colors_of_meaning.interface.cli.train \
  --config configs/base.yaml

# Step 2: Evaluate color method
python -m colors_of_meaning.interface.cli.eval \
  --dataset ag_news \
  --method color
```

## CLI Reference

### Evaluation

Evaluate classification performance with different methods:

```bash
# Evaluate on AG News (4-class topic classification)
python -m colors_of_meaning.interface.cli.eval \
  --dataset ag_news \
  --method [tfidf|hnsw|color] \
  --k-neighbors 5

# Evaluate on IMDB (binary sentiment classification)
python -m colors_of_meaning.interface.cli.eval \
  --dataset imdb \
  --method tfidf

# Evaluate on 20 Newsgroups (20-class topic classification)
python -m colors_of_meaning.interface.cli.eval \
  --dataset newsgroups \
  --method hnsw \
  --k-neighbors 10
```

**Arguments**:
- `--dataset`: Dataset to evaluate on (`ag_news`, `imdb`, `newsgroups`)
- `--method`: Classification method (`tfidf`, `hnsw`, `color`)
- `--k-neighbors`: Number of neighbors for k-NN methods (default: 5)
- `--config`: Configuration file (default: `configs/base.yaml`)
- `--model-path`: Path to trained projector model (default: `artifacts/models/projector.pth`)
- `--codebook-path`: Color codebook name (default: `codebook_4096`, resolved from `artifacts/codebooks/`)

### Training

Train the embedding-to-color projector:

```bash
python -m colors_of_meaning.interface.cli.train \
  --config configs/base.yaml
```

### Encoding

Encode documents to color histograms:

```bash
python -m colors_of_meaning.interface.cli.encode \
  --config configs/base.yaml \
  --split test
```

### Comparison

Compare documents via color distance:

```bash
python -m colors_of_meaning.interface.cli.compare \
  --config configs/base.yaml \
  --k 5 \
  --distance wasserstein
```

**Arguments**:
- `--k`: Number of nearest neighbors to find
- `--distance`: Distance metric (`wasserstein` or `jensen_shannon`)

### Compression

Analyze compression ratios:

```bash
python -m colors_of_meaning.interface.cli.compress \
  --config configs/base.yaml \
  --method vq
```

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

The architectural structure and design rules for this project uses a combination of Hexagonal Architecture and Domain-Driven Design (`DDD`) principles. The architecture is organized into clearly separated domain, application, infrastructure, and interface layers, following clean architecture principles. The domain layer encapsulates entities and business rules, the application layer manages orchestration through use cases, infrastructure contains data persistence implementations with repositories, and the interface layer provides `RESTful` `API` routes. The structure emphasizes high test coverage, maintainability, and security by design.

---

### Project Structure Overview

```
colors_of_meaning/

 application/                       # Use cases
    use_case/
        train_color_mapping_use_case.py
        encode_document_use_case.py
        compare_documents_use_case.py
        compress_document_use_case.py
        evaluate_use_case.py

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
        distance_calculator.py
        classifier.py
        retriever.py
        metrics_calculator.py

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
        wasserstein_distance_calculator.py
        jensen_shannon_distance_calculator.py
    persistence/
        file_color_codebook_repository.py
        in_memory/
    security/
        basic_authentication.py
    system/
        health_checker.py

 interface/                         # APIs and CLI
    api/
        main.py
        controller/
            health_controller.py
        data_transfer_object/
    cli/
        train.py
        encode.py
        compare.py
        compress.py
        eval.py

 shared/                            # Cross-cutting concerns
    configuration.py
    lab_utils.py
    synesthetic_config.py
```

---

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
- `ml/` contains PyTorch color mapper and distance calculators
- `evaluation/` provides classifier implementations (TF-IDF, HNSW, color histogram)
- `dataset/` adapts external datasets (AG News, IMDB, 20 Newsgroups)
- `embedding/` wraps sentence-transformers for text encoding
- `security/` provides token-based authentication
- `system/` contains diagnostics (e.g., disk, database checks)

#### Interface

- `api/controller/` expose `FastAPI` routes and depend on use cases
- `api/data_transfer_object/` define `Pydantic` model for request/response shaping
- `cli/` provides command-line tools for training, encoding, evaluation, and compression

#### Shared

- `lab_utils.py` provides RGB/Lab conversion and Delta E distance
- `synesthetic_config.py` loads YAML-based experiment configuration
- `configuration.py` loads application settings

---

### Dependency Injection

This project uses the dependency injection pattern to enhance modularity, testability, and maintainability, with [Lagom](https://github.com/meadsteve/lagom) as the dependency injection framework.

#### Key Principles

- **Inversion of Control**: Dependencies are provided to components rather than created internally.
- **Interface-based Programming**: Components depend on abstractions (interfaces) rather than concrete implementations.
- **Single Responsibility**: Each component focuses on its core functionality, with dependencies injected as needed.
- **Testability**: Allows easy replacement of dependencies with test doubles (mocks, stubs) in unit tests.

#### Using `Lagom` for Dependency Injection

`Lagom` provides a lightweight, Pythonic approach to dependency injection with features that align well with our architecture:

- **Type-based Resolution**: Dependencies are resolved based on type annotations.
- **Singletons and Transients**: Support for both singleton (shared) and transient (new instance per request) dependencies.
- **Contextual Containers**: Create specialized containers for different application contexts (e.g., testing vs. production).

---

### Enhancing System Quality

#### Performance and Scalability

- Implement caching strategies (`Redis`) for frequently accessed data.
- Use message queues (`Pub/Sub`) for asynchronous tasks.

#### Reliability and Fault Tolerance

- Explicitly define retry and circuit breaker strategies.
- Clearly document error handling and recovery procedures.

#### Maintainability and Modularity

- Clearly define module boundaries and use explicit interfaces (`ABC`).

#### Observability and Monitoring

- Structured logging with `correlation-id`.
- Metrics collection and distributed tracing.

#### Security

- Auditing of key domain events.
- Secure management of secrets (`Vault`).

#### Availability

- Explicit fall-back or degraded-service strategies.
- Robust health-check mechanisms.

#### Testability

- Include integration and end-to-end tests for core functionality.
- Contract testing for integrations.

#### Portability

- Containerization strategy (`Docker`).
- Infrastructure as code (`Terraform`, `Ansible`, `Packer`).

---

### Coding Conventions

#### General

- No comments: code should be self-documenting and expressive
- Descriptive naming: variables, functions, and classes should clearly communicate their intent
- Include `__init__.py` files in each Python package to clearly define module boundaries and support tools like `linters`, test runners, and code navigation.

#### Testing

- 100% test coverage is expected
- Tests should be meaningful and not just aim to increase coverage
- Mocks/stubs should be used where necessary to isolate behaviour
- Tests should include only one assertion per test case
- Test names should be phrased as a sentence clearly reflecting the behaviour being asserted
  - **Example:** `test_should_return_404_when_resource_is_not_found()`
- Consumer Driven Contract Testing (`CDCT`) consumer tests are required for any internal service called, and producer tests for `API` routes provided by this project.
- Architectural Unit Testing should be used to assert that the code aligns with the principles defined in this document.

#### Static Analysis

Code must pass the following tools before merging:

| Tool                | Purpose                                            |
|---------------------|----------------------------------------------------|
| `flake8`            | `Linting` and style checking                       |
| `black`             | Code formatting                                    |
| `bandit`            | Security scanning for common issues                |
| `xenon`             | Maintainability and complexity limits              |
| `mypy`              | Static type checking                               |
| `semgrep`           | Static analysis for code patterns and security     |
| `pip-audit`         | Dependency vulnerability scanning                  |

---

By following these conventions, the code remains:

- Clean and easy to reason about
- Secure by default
- Readable by both humans and tools
- Easy to `refactor` and scale with confidence
- Loosely coupled through dependency injection
