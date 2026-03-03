---
name: Test Generator with One-Assertion Rule
description: This skill should be used when the user asks to "generate tests", "create tests for", "write test cases", "add test coverage", "write tests", or "test this code". This skill generates tests following the project's strict one-assertion-per-test rule with proper naming and structure.
version: 1.0.0
---

# Test Generator with One-Assertion Rule

## Overview

This skill generates tests following the project's mandatory testing standards:
- **One assertion per test** (critical rule - violations will break the build)
- **Descriptive test names** using `test_should_[expected]_when_[condition]` pattern
- **Arrange-Act-Assert (AAA) structure** for clarity
- **Proper mocking** to isolate behavior under test
- **100% coverage** requirement

## When to Use This Skill

Use this skill when:
- Creating tests for new code
- Adding missing test coverage
- Refactoring tests that violate the one-assertion rule
- Converting multi-assertion tests into separate test functions

## Critical Rule: ONE ASSERTION PER TEST

Each test function MUST contain exactly ONE assertion. If you need multiple assertions, create separate test functions.

### Wrong (Multiple Assertions)
```python
def test_user_creation(self):
    user = create_user()
    assert_that(user.name).is_equal_to("John")  # ❌ Multiple assertions
    assert_that(user.email).is_equal_to("john@example.com")  # ❌
```

### Correct (One Assertion Each)
```python
def test_should_set_user_name_when_user_is_created(self):
    user = create_user()
    assert_that(user.name).is_equal_to("John")  # ✓ Single assertion

def test_should_set_user_email_when_user_is_created(self):
    user = create_user()
    assert_that(user.email).is_equal_to("john@example.com")  # ✓ Single assertion
```

## Test Naming Convention

Test names must be descriptive sentences following this pattern:

```
test_should_[expected_behavior]_when_[condition]
```

### Examples
- `test_should_return_404_when_resource_is_not_found()`
- `test_should_create_user_when_valid_data_is_provided()`
- `test_should_raise_validation_error_when_email_is_invalid()`
- `test_should_call_repository_read_method_when_use_case_executes()`
- `test_should_increment_counter_when_event_is_processed()`

### Naming Tips
- Start with `test_should_`
- Describe the expected outcome (what should happen)
- End with `_when_` followed by the condition or trigger
- Be specific and descriptive
- Use full words, not abbreviations

## Test Structure: Arrange-Act-Assert (AAA)

Every test should follow this three-part structure:

```python
def test_should_return_coconut_when_id_exists(self):
    # Arrange - Set up test data and mocks
    repository = Mock()
    use_case = GetCoconutUseCase(repository)
    coconut_id = uuid.uuid4()
    expected_coconut = Coconut(id=coconut_id)
    repository.read.return_value = expected_coconut

    # Act - Execute the behavior being tested
    result = use_case.execute(coconut_id)

    # Assert - Verify the expected outcome (ONE assertion only)
    assert_that(result).is_equal_to(expected_coconut)
```

### Section Guidelines

**Arrange:**
- Create test data
- Set up mocks and their return values
- Initialize the object under test with dependencies
- Configure the test scenario

**Act:**
- Execute the single method or behavior being tested
- Store the result if needed for assertion
- Should typically be one line

**Assert:**
- Verify ONE expected outcome
- **MUST use `assertpy` library** (`assert_that`) - bare `assert` statements are FORBIDDEN
- Use `pytest.raises` for exception testing

## Testing by Layer

### Domain Model Tests

Test Pydantic models and domain entities:

```python
from python_sprint_zero.domain.model.coconut import Coconut

def test_should_create_coconut_with_id(self):
    # Arrange
    coconut_id = uuid.uuid4()

    # Act
    coconut = Coconut(id=coconut_id)

    # Assert
    assert_that(coconut.id).is_equal_to(coconut_id)

def test_should_support_none_id(self):
    # Arrange & Act
    coconut = Coconut(id=None)

    # Assert
    assert_that(coconut.id).is_none()
```

### Repository Interface Tests (Domain)

Test the interface definition exists and has correct signatures:

```python
from python_sprint_zero.domain.repository.coconut_repository import CoconutQueryRepository

def test_should_define_read_method(self):
    # Arrange & Act
    method = getattr(CoconutQueryRepository, 'read', None)

    # Assert
    assert_that(method).is_not_none()
```

### Repository Implementation Tests (Infrastructure)

Test concrete repository implementations with proper setup:

```python
from python_sprint_zero.infrastructure.persistence.in_memory.in_memory_coconut_query_repository import (
    InMemoryCoconutQueryRepository
)

class TestInMemoryCoconutQueryRepository:
    @pytest.fixture
    def storage(self):
        return {}

    @pytest.fixture
    def repository(self, storage):
        return InMemoryCoconutQueryRepository(storage)

    def test_should_return_coconut_when_id_exists(self, repository, storage):
        # Arrange
        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)
        storage[coconut_id] = coconut

        # Act
        result = repository.read(coconut_id)

        # Assert
        assert_that(result).is_equal_to(coconut)

    def test_should_raise_exception_when_id_not_found(self, repository):
        # Arrange
        coconut_id = uuid.uuid4()

        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            repository.read(coconut_id)

        assert_that(str(excinfo.value)).contains("not found")
```

### Use Case Tests (Application)

Test use cases with mocked dependencies:

```python
from unittest.mock import Mock
from python_sprint_zero.application.use_case.coconut_use_case import GetCoconutUseCase

class TestGetCoconutUseCase:
    @pytest.fixture
    def mock_repository(self):
        return Mock()

    @pytest.fixture
    def use_case(self, mock_repository):
        return GetCoconutUseCase(mock_repository)

    def test_should_call_repository_read_method(self, use_case, mock_repository):
        # Arrange
        coconut_id = uuid.uuid4()
        mock_repository.read.return_value = Coconut(id=coconut_id)

        # Act
        use_case.execute(coconut_id)

        # Assert
        mock_repository.read.assert_called_once_with(coconut_id)

    def test_should_return_coconut_from_repository(self, use_case, mock_repository):
        # Arrange
        coconut_id = uuid.uuid4()
        expected_coconut = Coconut(id=coconut_id)
        mock_repository.read.return_value = expected_coconut

        # Act
        result = use_case.execute(coconut_id)

        # Assert
        assert_that(result).is_equal_to(expected_coconut)
```

### Controller Tests (Interface)

Test API controllers with TestClient:

```python
from fastapi.testclient import TestClient
from unittest.mock import Mock

class TestCoconutController:
    @pytest.fixture
    def mock_use_case(self):
        return Mock()

    @pytest.fixture
    def controller(self, mock_use_case):
        return CoconutController(
            get_use_case=mock_use_case,
            authentication_dependency=lambda x: None
        )

    @pytest.fixture
    def client(self, controller):
        app = FastAPI()
        app.include_router(controller.router)
        return TestClient(app)

    def test_should_return_200_when_coconut_exists(self, client, mock_use_case):
        # Arrange
        coconut_id = uuid.uuid4()
        mock_use_case.execute.return_value = Coconut(id=coconut_id)

        # Act
        response = client.get(f"/coconut/{coconut_id}")

        # Assert
        assert_that(response.status_code).is_equal_to(200)

    def test_should_return_coconut_data_when_requested(self, client, mock_use_case):
        # Arrange
        coconut_id = uuid.uuid4()
        mock_use_case.execute.return_value = Coconut(id=coconut_id)

        # Act
        response = client.get(f"/coconut/{coconut_id}")

        # Assert
        assert_that(response.json()["id"]).is_equal_to(str(coconut_id))
```

## Mocking Guidelines

### When to Mock

- **External dependencies** (databases, APIs, file systems)
- **Repository implementations** in use case tests
- **Use cases** in controller tests
- **Authentication dependencies**

### What NOT to Mock

- **The object under test** itself
- **Simple domain models** (Pydantic models)
- **Data transfer objects** (DTOs)

### Mock Setup

```python
from unittest.mock import Mock

# Create mock
mock_repository = Mock()

# Configure return value
mock_repository.read.return_value = Coconut(id=uuid.uuid4())

# Configure exception
mock_repository.read.side_effect = Exception("Not found")

# Verify call
mock_repository.read.assert_called_once_with(coconut_id)

# Verify call with any arguments
mock_repository.read.assert_called_once()
```

## Common Testing Patterns

### Testing Exceptions

```python
def test_should_raise_exception_when_not_found(self, use_case, mock_repository):
    # Arrange
    coconut_id = uuid.uuid4()
    mock_repository.read.side_effect = Exception("Not found")

    # Act & Assert
    with pytest.raises(Exception) as excinfo:
        use_case.execute(coconut_id)

    assert_that(str(excinfo.value)).contains("Not found")
```

### Testing with Fixtures

```python
class TestMyFeature:
    @pytest.fixture
    def sample_id(self):
        return uuid.uuid4()

    @pytest.fixture
    def repository(self):
        return InMemoryRepository({})

    def test_should_use_fixture(self, repository, sample_id):
        # Use fixtures in test
        pass
```

### Testing HTTP Status Codes

```python
def test_should_return_404_when_not_found(self, client, mock_use_case):
    # Arrange
    coconut_id = uuid.uuid4()
    mock_use_case.execute.side_effect = Exception("Not found")

    # Act
    response = client.get(f"/coconut/{coconut_id}")

    # Assert
    assert_that(response.status_code).is_equal_to(404)
```

## Test File Organization

### File Naming
- Test files mirror source structure exactly
- Prefix with `test_`: `test_coconut_use_case.py`

### Class Organization
```python
class TestGetCoconutUseCase:
    # Group related tests in classes
    # Use fixtures for setup
    # One behavior = one class

    def test_should_call_repository(self):
        pass

    def test_should_return_result(self):
        pass

class TestCreateCoconutUseCase:
    # Separate class for separate use case
    pass
```

## Splitting Multi-Assertion Tests

When you encounter tests with multiple assertions, split them:

### Before (Violates Rule)
```python
def test_user_creation(self):
    user = create_user("John", "john@example.com")
    assert_that(user.name).is_equal_to("John")
    assert_that(user.email).is_equal_to("john@example.com")
    assert_that(user.is_active).is_true()
```

### After (Follows Rule)
```python
def test_should_set_name_when_user_created(self):
    # Arrange & Act
    user = create_user("John", "john@example.com")

    # Assert
    assert_that(user.name).is_equal_to("John")

def test_should_set_email_when_user_created(self):
    # Arrange & Act
    user = create_user("John", "john@example.com")

    # Assert
    assert_that(user.email).is_equal_to("john@example.com")

def test_should_activate_user_when_created(self):
    # Arrange & Act
    user = create_user("John", "john@example.com")

    # Assert
    assert_that(user.is_active).is_true()
```

**Note:** Duplicating the Arrange-Act code is acceptable and encouraged. Each test should be independently readable.

## Assertion Libraries

### Using assertpy (Preferred)

```python
from assertpy import assert_that

# Equality
assert_that(result).is_equal_to(expected)

# None checks
assert_that(value).is_none()
assert_that(value).is_not_none()

# Boolean
assert_that(value).is_true()
assert_that(value).is_false()

# Contains
assert_that(string).contains("substring")
assert_that(list).contains(item)

# Type checks
assert_that(value).is_instance_of(MyClass)

# Comparisons
assert_that(value).is_greater_than(5)
assert_that(value).is_less_than(10)
```

### Using pytest.raises

```python
import pytest

with pytest.raises(ValueError) as excinfo:
    function_that_raises()

assert_that(str(excinfo.value)).contains("expected message")
```

## Coverage Requirements

- **100% test coverage** is mandatory
- Every function, class, and method must have tests
- Tests must be meaningful, not just coverage-seeking
- Use `tox` to verify coverage

```bash
# Run all tests with coverage
tox

# Run specific tests during TDD iteration
tox -- tests/specific_test.py

# View coverage report
coverage report
```

## Test Checklist

Before completing test generation, verify:

- [ ] Each test has exactly ONE assertion
- [ ] Test names follow `test_should_X_when_Y` pattern
- [ ] Tests use Arrange-Act-Assert structure
- [ ] Appropriate use of mocks for dependencies
- [ ] Tests are isolated and independent
- [ ] All edge cases covered (happy path, errors, edge cases)
- [ ] Tests verified with `tox` (NOT just `pytest` - must pass all quality gates)
- [ ] Coverage is 100%: confirmed by `tox` output

## Common Mistakes to Avoid

1. **Multiple assertions** - Split into separate tests
2. **Using bare assert statements** - ALWAYS use `assert_that` from assertpy
3. **Vague test names** - Use descriptive names with should/when
4. **Testing implementation details** - Test behavior, not internals
5. **Not using mocks** - Mock external dependencies
6. **Shared state between tests** - Each test should be independent
7. **Missing edge cases** - Test happy path, errors, and boundaries
8. **No AAA structure** - Always use Arrange-Act-Assert
9. **Testing multiple behaviors** - One test = one behavior
10. **Using pytest for final verification** - Use `tox` to verify all quality gates pass

## References

See the `references/` directory for:
- Complete test examples from the codebase
- Test patterns by layer
- Common testing scenarios
