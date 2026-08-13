# Python Sprint Zero - Claude Code Instructions

## Project Purpose

This project implements the **Colors of Meaning** experiment ([research article](https://www.qual.is/posts/colors-of-meaning)), exploring machine synesthesia for semantic compression and retrieval. The core idea is mapping 384-dimensional semantic embeddings into 3-dimensional CIE Lab perceptual color space, achieving an exact ~1024:1 compression (a 384-dim float32 embedding is 12,288 bits; one 4,096-color code is 12 bits) while maintaining interpretable semantic structure. Documents become color distributions (histograms over a quantized palette) rather than high-dimensional vectors.

### Core Domain Concepts

- **Lab Color**: CIE Lab perceptual color (L=lightness 0-100, a=green-red -128 to 127, b=blue-yellow -128 to 127)
- **Color Codebook**: 4,096-color fixed uniform Lab grid; Lab colors quantize to the nearest grid point (not learned vector quantization until `007-p1-1-learned-vq-codebook`)
- **Colored Document**: Document represented as a histogram over codebook colors
- **Semantic Color Mapping**: Neural projector from 384-dim sentence-transformers embeddings to 3-dim Lab space
- **Structured Mapping**: Self-supervised variant where hue encodes semantic clusters, lightness encodes sentiment, chroma encodes concreteness
- **Wasserstein Distance**: Earth mover's distance on color histograms for document comparison

### Processing Pipeline

```
text → sentence-transformers embedding (384-dim) → neural projector → Lab color → codebook quantization → color histogram → Wasserstein distance comparison
```

### Evaluation Baselines (AG News)

| Method | Accuracy | Macro F1 |
|--------|----------|----------|
| TF-IDF | 90.63% | 90.61% |
| HNSW k-NN | 91.99% | 91.97% |

Supported datasets: AG News (4-class topic), IMDB (binary sentiment), 20 Newsgroups (20-class topic)

## Absolute Non-Negotiables

These rules are **MANDATORY** and violations will break the project:

### 1. NO COMMENTS
- Code MUST be self-documenting through expressive naming
- NEVER add comments to any code
- If code needs explanation, refactor it to be clearer instead

### 2. ONE ASSERTION PER TEST
- Each test function SHOULD contain one logical assertion
- Do NOT use pytest subtests or unrelated multiple assertions
- Split tests with multiple unrelated assertions into separate test functions
- For ML/numerical tests, related assertions on the same result (e.g., checking shape and value ranges) are acceptable in a single test
- Use `assertpy` (`assert_that`) for base entity tests; plain `assert` and `pytest.raises` are acceptable for ML and domain-specific tests
- **Example:**
  ```python
  # WRONG - Multiple unrelated assertions
  def test_user_creation(self):
      user = create_user()
      assert_that(user.name).is_equal_to("John")
      assert_that(user.email).is_equal_to("john@example.com")

  # CORRECT - One assertion per test
  def test_should_set_user_name_when_user_is_created(self):
      user = create_user()
      assert_that(user.name).is_equal_to("John")

  def test_should_set_user_email_when_user_is_created(self):
      user = create_user()
      assert_that(user.email).is_equal_to("john@example.com")

  # ALSO CORRECT - Related assertions on same ML result
  def test_should_produce_valid_lab_output(self):
      output = network.forward(input_tensor)
      assert output.shape == (2, 3)
      assert torch.all(output[:, 0] >= 0) and torch.all(output[:, 0] <= 100)
  ```

### 3. LAYER BOUNDARY VIOLATIONS FORBIDDEN
- **Domain** MUST NOT import from: `application`, `infrastructure`, `interface`
- **Application** MUST NOT import from: `infrastructure`, `interface`
- **Infrastructure** MAY import from: `domain`, `application`
- **Interface** MAY import from: `domain`, `application`, `infrastructure`
- **Shared** MAY be imported by any layer

### 4. 100% TEST COVERAGE REQUIRED
- Every function, class, and method MUST have tests
- Tests MUST be meaningful, not just coverage-seeking
- Use `tox` to verify coverage before marking work complete

### 5. PREFER EDITING OVER CREATING
- ALWAYS prefer editing existing files to creating new ones
- Only create new files when absolutely necessary
- Do NOT create documentation files unless explicitly requested

## Architectural Layer Rules

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
    service/

 domain/                            # Business logic
    health/
        health_status.py
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
    system/
        health_checker.py

 interface/                         # APIs and CLI
    api/
        main.py
        controller/
            health_controller.py
            query_controller.py
        data_transfer_object/
            health_dto.py
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

### Domain Layer (`domain/`)

**Purpose:** Pure business logic and entities

**Rules:**
- Define abstract repository interfaces using `ABC`
- Implement domain entities as dataclasses or Pydantic models
- Contain stateless domain services with business rules
- MUST NOT depend on external frameworks (FastAPI, databases, etc.)
- MUST NOT have side effects (no I/O, no external calls)

**Structure:**
- `model/` - Domain entities (e.g., `LabColor`, `ColorCodebook`, `ColoredDocument`, `EvaluationResult`)
- `repository/` - Repository interfaces (abstract base classes)
- `service/` - Domain service interfaces (e.g., `ColorMapper`, `DistanceCalculator`, `Classifier`, `Retriever`, `MetricsCalculator`, `FigureRenderer`, `CompressionBaseline`)
- `health/` - Health status domain model

### Application Layer (`application/`)

**Purpose:** Orchestrate use cases and coordinate domain logic

**Rules:**
- Use cases orchestrate and delegate to domain services
- MUST NOT depend on FastAPI, databases, or file systems directly
- Use dependency injection (Lagom) to receive dependencies
- Focus on workflow orchestration, not business logic
- Handle application-level concerns (transaction boundaries, etc.)

**Structure:**
- `use_case/` - Use case implementations (e.g., `TrainColorMappingUseCase`, `EncodeDocumentUseCase`, `EvaluateUseCase`, `QueryByPaletteUseCase`)
- `service/` - Application-level services

### Infrastructure Layer (`infrastructure/`)

**Purpose:** Implement technical adapters and integrations

**Rules:**
- Implement repository interfaces from `domain.repository` and service interfaces from `domain.service`
- Handle all external integrations (datasets, ML frameworks, APIs)
- Provide concrete implementations of domain abstractions
- Log through the standard-library logger with a `correlation-id` (there is no metrics or tracing backend — see Observability Requirements)

**Structure:**
- `ml/` - PyTorch color mappers (unconstrained and structured), compression baselines (gzip, Product Quantization), and distance calculators (Wasserstein, Jensen-Shannon)
- `dataset/` - Dataset adapters (AG News, IMDB, 20 Newsgroups)
- `embedding/` - Sentence-transformers embedding adapter
- `evaluation/` - Classifier implementations (TF-IDF, HNSW, color histogram) and metrics calculator
- `visualization/` - Matplotlib figure renderer for codebook palettes, histograms, projections, confusion matrices
- `persistence/` - Repository implementations (file-based codebook, JSON scaling manifest, in-memory)
- `generation/` - Anthropic text generator adapter behind the `TextGenerator` port
- `system/` - Health checks and diagnostics

### Interface Layer (`interface/`)

**Purpose:** Expose APIs and handle external communication

**Rules:**
- Controllers expose FastAPI routes
- **MUST use Pydantic DTOs for ALL endpoint responses** - never return plain dictionaries
- Use Pydantic models for DTOs (request/response shaping)
- Depend on use cases from application layer
- Handle HTTP-specific concerns (status codes, headers, etc.)
- Controllers are built by a factory that receives its use case and closes over it; the container is composed once at import in `api/main.py`

**Structure:**
- `api/main.py` - FastAPI application setup
- `api/controller/` - API route controllers (health, query by palette)
- `api/data_transfer_object/` - Pydantic DTOs (health, palette query)
- `cli/` - Command-line tools (train, encode, compare, compress, eval, visualize, query)

**Example Controller Pattern:**
```python
from fastapi import APIRouter, status

from colors_of_meaning.application.use_case.query_by_palette_use_case import QueryByPaletteUseCase
from colors_of_meaning.interface.api.data_transfer_object.palette_query_dto import (
    PaletteQueryRequestDTO,
    PaletteQueryResponseDTO,
)


def create_query_controller(query_use_case: QueryByPaletteUseCase) -> APIRouter:
    router = APIRouter(tags=["query"])

    async def query_by_palette(request: PaletteQueryRequestDTO) -> PaletteQueryResponseDTO:
        matches = query_use_case.execute(palette=request.colors, k=request.k)
        return PaletteQueryResponseDTO(matches=matches, query_colors=len(request.colors))

    router.add_api_route(
        "/query/palette",
        query_by_palette,
        methods=["POST"],
        status_code=status.HTTP_200_OK,
    )

    return router
```

### Shared Layer (`shared/`)

**Purpose:** Cross-cutting concerns and utilities

**Rules:**
- Contains reusable utilities accessible from all layers
- Includes configuration management
- Provides color space utilities and experiment configuration

**Structure:**
- `configuration.py` - Application settings and config loading
- `lab_utils.py` - RGB/Lab conversion and Delta E distance utilities
- `synesthetic_config.py` - YAML-based experiment configuration (projector, codebook, training, distance, dataset, structured mapper settings)

## Testing Requirements

### Test Naming Convention

Test names MUST be phrased as descriptive sentences using the pattern:
```
test_should_[expected_behavior]_when_[condition]
```

**Examples:**
- `test_should_return_404_when_resource_is_not_found()`
- `test_should_create_user_when_valid_data_is_provided()`
- `test_should_raise_validation_error_when_email_is_invalid()`
- `test_should_increment_counter_when_event_is_processed()`

### Test Structure

Base entity tests use `assertpy` (`assert_that`). ML and domain-specific tests may use plain `assert` and `pytest.raises`.

```python
from assertpy import assert_that

def test_should_return_codebook_when_name_exists(self):
    # Arrange
    repository = InMemoryColorCodebookRepository()
    codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=4)
    repository.save(codebook, "codebook_64")

    # Act
    result = repository.load("codebook_64")

    # Assert
    assert_that(result.num_bins).is_equal_to(64)
```

```python
def test_should_embed_to_lab(self):
    mapper = PyTorchColorMapper(input_dim=10, device="cpu")
    embedding = np.array([1.0] * 10, dtype=np.float32)

    result = mapper.embed_to_lab(embedding)

    assert isinstance(result, LabColor)
```

### Consumer Driven Contract Testing (CDCT)

**Required for:**
- Any internal service your project calls (consumer tests)
- Any API routes your project provides (producer tests)

**Consumer Test Example:**
```python
def test_should_return_expected_user_schema_when_calling_user_service(self):
    # Test that external service returns expected contract
    pass
```

**Producer Test Example:**
```python
def test_should_return_palette_match_schema_in_query_endpoint_response(self):
    # Test that your API returns expected contract
    pass
```

### Architectural Unit Testing

MUST include tests that validate architectural rules:

```python
def test_should_not_import_infrastructure_in_domain_layer(self):
    # Verify domain doesn't import from infrastructure
    pass

def test_should_define_repository_interfaces_in_domain(self):
    # Verify repository abstractions exist in domain
    pass
```

### Mocking and Test Isolation

- Use mocks/stubs to isolate behavior under test
- Prefer dependency injection for testability
- Mock external services, datasets, and I/O (no network calls in unit tests)
- Keep tests fast and independent
- Use `assertpy` (`assert_that`) for base entity tests; plain `assert` and `pytest.raises` are acceptable for ML/domain-specific tests

## Dependency Injection with Lagom

**ALWAYS use dependency injection** - never directly instantiate dependencies.

### Principles
- Components receive dependencies rather than creating them
- Depend on abstractions (interfaces) not concrete implementations
- Use Lagom's type-based resolution
- Configure containers for different contexts (test vs. production)

### Pattern
```python
from lagom import Container

# Define interface in domain
class ColorCodebookRepository(ABC):
    @abstractmethod
    def load(self, name: str) -> Optional[ColorCodebook]:
        raise NotImplementedError

# Implement in infrastructure
class InMemoryColorCodebookRepository(ColorCodebookRepository):
    def load(self, name: str) -> Optional[ColorCodebook]:
        return self.codebooks.get(name)

# Configure container
container = Container()
container[ColorCodebookRepository] = InMemoryColorCodebookRepository

# Inject in use case
class VisualizeCodebookUseCase:
    def __init__(self, codebook_repository: ColorCodebookRepository, figure_renderer: FigureRenderer):
        self.codebook_repository = codebook_repository
        self.figure_renderer = figure_renderer
```

## Observability Requirements

### Structured Logging (implemented)
- Include a `correlation-id` in log entries
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR) using the standard-library logger

### Metrics & Distributed Tracing (not implemented)
- The project does **not** ship a metrics backend or distributed tracing; correlation-id structured logging is the current observability surface
- If a feature needs metrics or tracing, add it behind a `domain/service` port with an adapter in `infrastructure/observability/` — do not assume it already exists

## Security Requirements

### Authentication & Authorization (not implemented)
- The API is unauthenticated: every endpoint is open, and there is no `Authenticator` port or security adapter
- Run it locally or on a trusted network; it is a research service, not a multi-tenant one
- If authentication is added, attach the dependency to the routers in `create_app()` and cover it with a test that an uncredentialed request receives `401` — an adapter that is registered but never wired protects nothing

### Auditing
- Log key domain events (with `correlation-id`) for an audit trail
- Include user context in audit logs where available
- Tamper-proof / append-only audit storage is not implemented; the audit trail is structured logging

### Secrets Management
- Load secrets from the environment (today the only one is `ANTHROPIC_API_KEY`, used by book generation) or a git-ignored `.env`
- Never commit credentials or secrets, and never hardcode them in code
- A dedicated secret manager (e.g. Vault) is not used; the environment-based approach above is the current mechanism

## System Qualities

The qualities below describe the intended architecture. Items are marked **(implemented)** where they exist in the codebase today and **(aspirational)** where they are target patterns the hexagonal structure would support but which are **not yet built** — do not assume aspirational items exist.

### Maintainability and Modularity (implemented)

- Explicit module boundaries and `ABC` interfaces, enforced by the `pytest-archon` architecture tests.

### Observability and Monitoring

- Structured logging with `correlation-id` (implemented).
- Metrics collection and distributed tracing (aspirational — not implemented).

### Security (partial)

- Auditing of key domain events via structured logging (implemented).
- Secrets loaded from the environment; a secret manager such as Vault is aspirational.
- API authentication is **not** implemented — every endpoint is open.

### Availability (implemented)

- Robust health checks (readiness verifies codebook/model artifacts; liveness reads heap state).
- Fall-back / degraded-service behaviour where applicable.

### Testability (implemented)

- 100% test coverage, integration tests, and consumer/producer contract tests.

### Portability

- Containerization via Docker images built with Packer and Ansible (implemented).
- Terraform-based infrastructure-as-code (aspirational — Packer and Ansible are used today; Terraform is not).

### Performance and Scalability (aspirational — not implemented)

- Caching (e.g. `Redis`) and asynchronous messaging (e.g. `Pub/Sub`) are target patterns, not current features.

### Reliability and Fault Tolerance (aspirational — not implemented)

- Retry and circuit-breaker strategies are target patterns; document error handling and recovery as features add them.

## Code Quality Standards

### Static Analysis Tools

Before completing any work, code MUST pass:

| Tool | Purpose | Command |
|------|---------|---------|
| `ruff check` | Linting and style | `tox` |
| `ruff format` | Code formatting | `tox -e format` |
| `bandit` | Security scanning | `tox` |
| `radon` | Complexity reporting | `tox` |
| `xenon` | Complexity limits | `tox` |
| `mypy` | Type checking | `tox` |
| `semgrep` | Pattern/security analysis | `tox` |
| `pip-audit` | Dependency vulnerabilities | `tox` |

### Dependency Management (uv)

Dependencies are managed with **uv**. `pyproject.toml` holds abstract requirements plus all
project metadata (PEP 621 `[project]`); `uv.lock` holds the exact cross-platform closure.
Both are committed. There is no `setup.cfg`, `setup.py`, or `requirements.lock`.

| Task | Command |
|------|---------|
| Create/refresh the local environment | `uv sync --locked --extra testing` |
| Add or change a dependency | edit `pyproject.toml`, then `uv lock` |
| Install the gate runner | `uv tool install --python 3.11 --with tox-uv tox` |

- ALWAYS commit `uv.lock` alongside a `pyproject.toml` dependency change. `tox` runs
  `uv sync --locked`, which FAILS if the lock is stale — do not work around it with `--frozen`.
- `torch` resolves from an explicit PyTorch CPU index via `[tool.uv.sources]`; do NOT add
  `nvidia-*` or `triton` pins to make the lock portable.
- `tox` requires **tox 4** with the `tox-uv` plugin. Every environment installs from `uv.lock` —
  `build`/`publish` via `only_groups` against `[dependency-groups]`, the rest via `extras`. Only
  `clean` opts out (`runner = uv-venv-runner`), because it installs nothing at all. Never add a
  bare `deps` to a lock-runner env: it is silently ignored. The `--python 3.11` above is what
  makes a local gate run match CI.

### Module Structure

- Include `__init__.py` in EVERY Python package
- This supports linters, test runners, and code navigation
- Defines clear module boundaries

### Naming Conventions

- Use descriptive names that communicate intent
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

### Type Hints

- Use type hints for all function signatures
- Use `typing` module for complex types
- Enable `mypy` strict mode compliance

## Development Workflow

### Before Starting Work
1. Understand the architectural layer you're working in
2. Identify existing files to edit rather than creating new ones
3. Plan your tests before implementation

### During Development
1. Write tests first (TDD approach encouraged)
2. Implement with one assertion per test
3. Use dependency injection (Lagom)
4. Add structured logging with a `correlation-id`
5. Ensure no comments - make code self-documenting

### Before Completing Work
1. Run `tox` to verify all tests pass and coverage is 100%
2. Run `tox -e format` to format code with ruff
3. Verify all static analysis tools pass
4. Review for layer boundary violations
5. Confirm architectural unit tests pass

### Running Tests

**IMPORTANT: Always use `tox` for final verification, NOT `pytest` directly**

Running `pytest` directly bypasses 8 critical quality gates:
- ruff check (linting/style)
- ruff format (code formatting)
- bandit (security scanning)
- semgrep (pattern/security analysis)
- pip-audit (dependency vulnerabilities)
- radon (cyclomatic complexity)
- xenon (complexity enforcement)
- mypy (type checking)

This creates a false sense of completion. Tests may pass locally but fail in CI/CD.

```bash
# ✅ CORRECT - Full verification with all quality gates
tox

# ✅ CORRECT - Quick iteration during TDD (runs specific test with all quality gates)
tox -- tests/specific_test.py

# ❌ WRONG - Bypasses quality gates
pytest tests/specific_test.py

# Run tests in watch mode
tox -e watch

# Format code
tox -e format

# Run locally
./bin/run-local -c
```

**Rule of thumb:** Always use `tox` (or `tox --` for specific tests), NEVER `pytest` directly.

## When Uncertain

### ASK rather than guess when:
- Unclear which layer should contain logic
- Uncertain about dependency direction
- Need clarification on requirements
- Unsure if creating a new file is necessary

### DO NOT:
- Create files without necessity
- Add comments to explain unclear code (refactor instead)
- Violate layer boundaries "just this once"
- Write tests with multiple unrelated assertions
- Skip running `tox` before completion

## Common Pitfalls to Avoid

1. **Importing infrastructure in domain** - Domain must be pure
2. **Multiple unrelated assertions in one test** - Split into separate tests (related assertions on same ML result are acceptable)
3. **Returning plain dicts from endpoints** - MUST use Pydantic DTOs
4. **Adding comments** - Make code self-documenting instead
5. **Direct instantiation** - Use dependency injection
6. **Missing `__init__.py`** - Add to all packages
7. **Wrong test names** - Follow sentence pattern
8. **Skipping CDCT tests** - Required for service interactions
9. **Missing observability** - Add logging with correlation-id
10. **Using pytest instead of tox for final verification** - Bypasses 8 quality gates (ruff check, ruff format, bandit, semgrep, mypy, xenon, radon, pip-audit)
11. **Creating new files unnecessarily** - Prefer editing existing
12. **Network calls in unit tests** - Use synthetic data and mocks instead

## Success Criteria

Work is complete when:
- [ ] All tests pass with 100% coverage (`tox`)
- [ ] All static analysis passes (ruff check, ruff format, bandit, xenon, mypy, semgrep, pip-audit)
- [ ] Each test has one logical assertion (related assertions on same result are acceptable)
- [ ] Test names follow sentence pattern
- [ ] No comments exist in code
- [ ] Layer boundaries are respected
- [ ] Dependency injection is used throughout
- [ ] CDCT tests exist for service interactions
- [ ] Architectural unit tests validate structure
- [ ] Structured logging with a `correlation-id` is in place
- [ ] No secrets are committed
- [ ] `__init__.py` files exist in all packages
