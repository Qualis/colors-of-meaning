# Python Sprint Zero - Claude Code Instructions

## Absolute Non-Negotiables

These rules are **MANDATORY** and violations will break the project:

### 1. NO COMMENTS
- Code MUST be self-documenting through expressive naming
- NEVER add comments to any code
- If code needs explanation, refactor it to be clearer instead

### 2. ONE ASSERTION PER TEST
- Each test function MUST contain exactly ONE assertion
- Do NOT use pytest subtests or multiple assertions
- Split tests with multiple assertions into separate test functions
- **MUST use assertpy** (`assert_that`) - never bare Python `assert` statements
- **Example:**
  ```python
  # WRONG - Multiple assertions
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
project/

 application/                       # Use cases
    use_case/
        coconut_use_case.py
    service/
        example_service.py

 domain/                            # Business logic
    authentication/
        authentitator.py
    health/
        health_status.py
    model/
        coconut.py
    repository/
        coconut_repository.py
    service/
        example_data_import_service.py

 infrastructure/                    # Adapters, drivers
    observability/
        logger.py
        metrics.py
        tracing.py
    persistence/
        in_memory_coconut_command_repository.py
        in_memory_coconut_query_repository.py
    cache/
        cache_provider.py
    message/
        message_producer.py
        message_consumer.py
    importer/
        example_importer.py
        base_importer.py
    security/
        basic_authentication.py
    system/
        health_checker.py

 interface/                         # APIs and CLI
    api/
        main.py
        controller/
            coconut_controller.py
            health_controller.py
        data_transfer_object/
            coconut_data_transfer_object.py
            health_status_data_transfer_object.py

 shared/                            # Cross-cutting concerns
    configuration.py
    resilience/
        retry.py
        circuit_breaker.py
    formatter/
        example_formatter.py
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
- `model/` - Domain entities (e.g., `coconut.py`)
- `repository/` - Repository interfaces (abstract base classes)
- `service/` - Domain logic services
- `authentication/` - Authentication domain logic

### Application Layer (`application/`)

**Purpose:** Orchestrate use cases and coordinate domain logic

**Rules:**
- Use cases orchestrate and delegate to domain services
- MUST NOT depend on FastAPI, databases, or file systems directly
- Use dependency injection (Lagom) to receive dependencies
- Focus on workflow orchestration, not business logic
- Handle application-level concerns (transaction boundaries, etc.)

**Structure:**
- `use_case/` - Use case implementations (e.g., `coconut_use_case.py`)
- `service/` - Application-level services

### Infrastructure Layer (`infrastructure/`)

**Purpose:** Implement technical adapters and integrations

**Rules:**
- Implement repository interfaces from `domain.repository`
- Handle all external integrations (databases, APIs, message queues)
- Provide concrete implementations of domain abstractions
- Include observability implementations (logging, metrics, tracing)
- Manage security implementations (authentication, authorization)

**Structure:**
- `persistence/` - Repository implementations (in-memory, database, etc.)
- `observability/` - Logging, metrics, tracing
- `security/` - Authentication and authorization implementations
- `cache/` - Caching providers (Redis, etc.)
- `message/` - Message queue producers/consumers
- `importer/` - Data import implementations
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
- `api/controller/` - API route controllers
- `api/data_transfer_object/` - Pydantic DTOs

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
- Provides resilience patterns (retry, circuit breaker)
- Contains formatters and common utilities

**Structure:**
- `configuration.py` - Settings and config loading
- `resilience/` - Retry and circuit breaker patterns
- `formatter/` - Reusable formatting utilities

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

All tests MUST use `assertpy` library (`assert_that`) - never bare `assert` statements.

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
- Mock external services, databases, and I/O
- Keep tests fast and independent
- **ALWAYS use `assert_that` from assertpy** - bare `assert` statements are forbidden

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
- Write tests with multiple assertions
- Skip running `tox` before completion

## Common Pitfalls to Avoid

1. **Importing infrastructure in domain** - Domain must be pure
2. **Multiple assertions in one test** - Split into separate tests
3. **Using bare assert statements** - ALWAYS use `assert_that` from assertpy
4. **Returning plain dicts from endpoints** - MUST use Pydantic DTOs
5. **Adding comments** - Make code self-documenting instead
6. **Direct instantiation** - Use dependency injection
7. **Missing `__init__.py`** - Add to all packages
8. **Wrong test names** - Follow sentence pattern
9. **Skipping CDCT tests** - Required for service interactions
10. **Missing observability** - Add logging with correlation-id
11. **Using pytest instead of tox for final verification** - Bypasses 8 quality gates (flake8, black, bandit, semgrep, mypy, xenon, radon, pip-audit)
12. **Creating new files unnecessarily** - Prefer editing existing

## Success Criteria

Work is complete when:
- [ ] All tests pass with 100% coverage (`tox`)
- [ ] All static analysis passes (flake8, black, bandit, xenon, mypy, semgrep, pip-audit)
- [ ] Each test has exactly one assertion
- [ ] Test names follow sentence pattern
- [ ] No comments exist in code
- [ ] Layer boundaries are respected
- [ ] Dependency injection is used throughout
- [ ] CDCT tests exist for service interactions
- [ ] Architectural unit tests validate structure
- [ ] Observability is implemented (logging, metrics, tracing)
- [ ] No secrets are committed
- [ ] `__init__.py` files exist in all packages
