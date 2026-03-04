# Python Sprint Zero - Claude Code Instructions

## Project Purpose

This project implements the **Colors of Meaning** experiment ([research article](https://www.qual.is/posts/colors-of-meaning)), exploring machine synesthesia for semantic compression and retrieval. The core idea is mapping 384-dimensional semantic embeddings into 3-dimensional CIE Lab perceptual color space, achieving extreme compression (1000x+) while maintaining interpretable semantic structure. Documents become color distributions (histograms over a quantized palette) rather than high-dimensional vectors.

### Core Domain Concepts

- **Lab Color**: CIE Lab perceptual color (L=lightness 0-100, a=green-red -128 to 127, b=blue-yellow -128 to 127)
- **Color Codebook**: 4,096-color palette used for vector quantization of Lab colors
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
        coconut_use_case.py
    service/

 domain/                            # Business logic
    authentication/
        authenticator.py
    health/
        health_status.py
    model/
        lab_color.py
        color_codebook.py
        colored_document.py
        evaluation_sample.py
        evaluation_result.py
        coconut.py
    repository/
        color_codebook_repository.py
        dataset_repository.py
        coconut_repository.py
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
    observability/
        logger.py
        metrics.py
        tracing.py
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
            coconut_controller.py
            query_controller.py
        data_transfer_object/
            coconut_data_transfer_object.py
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
- `model/` - Domain entities (e.g., `LabColor`, `ColorCodebook`, `ColoredDocument`, `EvaluationResult`, `Coconut`)
- `repository/` - Repository interfaces (abstract base classes)
- `service/` - Domain service interfaces (e.g., `ColorMapper`, `DistanceCalculator`, `Classifier`, `Retriever`, `MetricsCalculator`, `FigureRenderer`, `CompressionBaseline`)
- `authentication/` - Authentication domain logic
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
- `use_case/` - Use case implementations (e.g., `TrainColorMappingUseCase`, `EncodeDocumentUseCase`, `EvaluateUseCase`, `QueryByPaletteUseCase`, `CoconutUseCase`)
- `service/` - Application-level services

### Infrastructure Layer (`infrastructure/`)

**Purpose:** Implement technical adapters and integrations

**Rules:**
- Implement repository interfaces from `domain.repository` and service interfaces from `domain.service`
- Handle all external integrations (datasets, ML frameworks, APIs)
- Provide concrete implementations of domain abstractions
- Include observability implementations (logging, metrics, tracing)
- Manage security implementations (authentication, authorization)

**Structure:**
- `ml/` - PyTorch color mappers (unconstrained and structured), compression baselines (gzip, Product Quantization), and distance calculators (Wasserstein, Jensen-Shannon)
- `dataset/` - Dataset adapters (AG News, IMDB, 20 Newsgroups)
- `embedding/` - Sentence-transformers embedding adapter
- `evaluation/` - Classifier implementations (TF-IDF, HNSW, color histogram) and metrics calculator
- `visualization/` - Matplotlib figure renderer for codebook palettes, histograms, projections, confusion matrices
- `persistence/` - Repository implementations (file-based codebook, in-memory)
- `observability/` - Logging, metrics, tracing
- `security/` - Authentication and authorization implementations
- `system/` - Health checks and diagnostics

### Interface Layer (`interface/`)

**Purpose:** Expose APIs and handle external communication

**Rules:**
- Controllers expose FastAPI routes
- **MUST use Pydantic DTOs for ALL endpoint responses** - never return plain dictionaries
- Use Pydantic models for DTOs (request/response shaping)
- Depend on use cases from application layer
- Handle HTTP-specific concerns (status codes, headers, etc.)
- Use FastAPI's `Depends()` alongside Lagom for dependency injection

**Structure:**
- `api/main.py` - FastAPI application setup
- `api/controller/` - API route controllers (health, coconut, query by palette)
- `api/data_transfer_object/` - Pydantic DTOs (coconut, palette query)
- `cli/` - Command-line tools (train, encode, compare, compress, eval, visualize, query)

**Example Controller Pattern:**
```python
from fastapi import APIRouter, Depends
from typing import Annotated
from application.use_case.coconut_use_case import CoconutUseCase
from interface.api.data_transfer_object.coconut_dto import CoconutResponse

router = APIRouter()

def get_use_case() -> CoconutUseCase:
    # Lagom container resolution here
    pass

@router.get("/coconuts/{id}")
async def get_coconut(
    id: str,
    use_case: Annotated[CoconutUseCase, Depends(get_use_case)]
) -> CoconutResponse:
    coconut = use_case.get_coconut(id)
    return CoconutResponse.model_validate(coconut)
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

def test_should_return_coconut_when_id_exists(self):
    # Arrange
    repository = InMemoryCoconutRepository()
    use_case = CoconutUseCase(repository)
    coconut_id = "test-id"

    # Act
    result = use_case.get_coconut(coconut_id)

    # Assert
    assert_that(result.id).is_equal_to(coconut_id)
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
def test_should_return_coconut_schema_in_get_endpoint_response(self):
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
class CoconutRepository(ABC):
    @abstractmethod
    def get(self, id: str) -> Coconut:
        pass

# Implement in infrastructure
class InMemoryCoconutRepository(CoconutRepository):
    def get(self, id: str) -> Coconut:
        # Implementation
        pass

# Configure container
container = Container()
container[CoconutRepository] = InMemoryCoconutRepository

# Inject in use case
class CoconutUseCase:
    def __init__(self, repository: CoconutRepository):
        self.repository = repository
```

## Observability Requirements

### Structured Logging
- Include `correlation-id` in all log entries
- Use structured logging format (JSON)
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)

### Metrics Collection
- Track key business metrics
- Monitor performance indicators
- Use appropriate metric types (counters, gauges, histograms)

### Distributed Tracing
- Implement tracing decorators for use cases
- Propagate trace context across service boundaries
- Track request flows through the system

## Security Requirements

### Authentication & Authorization
- Implement token-based authentication in `infrastructure/security/`
- Define authentication domain logic in `domain/authentication/`
- Never commit credentials or secrets

### Auditing
- Log key domain events for audit trail
- Include user context in audit logs
- Implement tamper-proof audit logging

### Secrets Management
- Use Vault or equivalent for secret storage
- Never hardcode secrets in code
- Load secrets from environment or secret manager

## Enhancing System Quality

### Performance and Scalability

- Implement caching strategies (`Redis`) for frequently accessed data.
- Use message queues (`Pub/Sub`) for asynchronous tasks.

### Reliability and Fault Tolerance

- Explicitly define retry and circuit breaker strategies.
- Clearly document error handling and recovery procedures.

### Maintainability and Modularity

- Clearly define module boundaries and use explicit interfaces (`ABC`).

### Observability and Monitoring

- Structured logging with `correlation-id`.
- Metrics collection and distributed tracing.

### Security

- Auditing of key domain events.
- Secure management of secrets (`Vault`).

### Availability

- Explicit fall-back or degraded-service strategies.
- Robust health-check mechanisms.

### Testability

- Include integration and end-to-end tests for core functionality.
- Contract testing for integrations.

### Portability

- Containerization strategy (`Docker`).
- Infrastructure as code (`Terraform`, `Ansible`, `Packer`).

## Code Quality Standards

### Static Analysis Tools

Before completing any work, code MUST pass:

| Tool | Purpose | Command |
|------|---------|---------|
| `flake8` | Linting and style | `tox` |
| `black` | Code formatting | `tox -e format` |
| `bandit` | Security scanning | `tox` |
| `xenon` | Complexity limits | `tox` |
| `mypy` | Type checking | `tox` |
| `semgrep` | Pattern/security analysis | `tox` |
| `pip-audit` | Dependency vulnerabilities | `tox` |

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
4. Add observability (logging, metrics)
5. Ensure no comments - make code self-documenting

### Before Completing Work
1. Run `tox` to verify all tests pass and coverage is 100%
2. Run `tox -e format` to format code with black
3. Verify all static analysis tools pass
4. Review for layer boundary violations
5. Confirm architectural unit tests pass

### Running Tests

**IMPORTANT: Always use `tox` for final verification, NOT `pytest` directly**

Running `pytest` directly bypasses 8 critical quality gates:
- flake8 (linting/style)
- black (code formatting)
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
10. **Using pytest instead of tox for final verification** - Bypasses 8 quality gates (flake8, black, bandit, semgrep, mypy, xenon, radon, pip-audit)
11. **Creating new files unnecessarily** - Prefer editing existing
12. **Network calls in unit tests** - Use synthetic data and mocks instead

## Success Criteria

Work is complete when:
- [ ] All tests pass with 100% coverage (`tox`)
- [ ] All static analysis passes (flake8, black, bandit, xenon, mypy, semgrep, pip-audit)
- [ ] Each test has one logical assertion (related assertions on same result are acceptable)
- [ ] Test names follow sentence pattern
- [ ] No comments exist in code
- [ ] Layer boundaries are respected
- [ ] Dependency injection is used throughout
- [ ] CDCT tests exist for service interactions
- [ ] Architectural unit tests validate structure
- [ ] Observability is implemented (logging, metrics, tracing)
- [ ] No secrets are committed
- [ ] `__init__.py` files exist in all packages
