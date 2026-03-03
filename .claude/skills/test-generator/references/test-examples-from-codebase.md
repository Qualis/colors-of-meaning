# Test Examples from Codebase

This document contains actual test examples from the project demonstrating proper patterns.

## Use Case Test Example

From `tests/python_sprint_zero/application/use_case/test_coconut_use_case.py`:

```python
class TestGetCoconutUseCase:
    @pytest.fixture
    def mock_query_repository(self):
        return Mock()

    @pytest.fixture
    def use_case(self, mock_query_repository):
        return GetCoconutUseCase(mock_query_repository)

    def test_should_call_repository_read_method(self, use_case, mock_query_repository, sample_coconut_id):
        mock_query_repository.read.return_value = Coconut(id=sample_coconut_id)

        use_case.execute(sample_coconut_id)

        mock_query_repository.read.assert_called_once_with(sample_coconut_id)

    def test_should_return_coconut_from_repository(self, use_case, mock_query_repository, sample_coconut_id):
        expected_coconut = Coconut(id=sample_coconut_id)
        mock_query_repository.read.return_value = expected_coconut

        result = use_case.execute(sample_coconut_id)

        assert_that(result).is_equal_to(expected_coconut)

    def test_should_propagate_not_found_exception(self, use_case, mock_query_repository, sample_coconut_id):
        mock_query_repository.read.side_effect = Exception("Coconut not found")

        with pytest.raises(Exception) as excinfo:
            use_case.execute(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to("Coconut not found")
```

**Observations:**
- Each test has ONE assertion
- Fixtures provide clean test setup
- Mocks isolate the use case from repository implementation
- Test names clearly describe behavior
- Separate tests for different behaviors (call verification, return value, exception handling)

## Controller Test Example

From `tests/python_sprint_zero/interface/api/controller/test_coconut_controller.py`:

```python
class TestCoconutController:
    @pytest.fixture
    def mock_get_coconut_use_case(self) -> Mock:
        mock = Mock(spec=GetCoconutUseCase)
        return mock

    @pytest.fixture
    def controller(self, mock_get_coconut_use_case, authentication_dependency) -> CoconutController:
        return CoconutController(
            get_coconut_use_case=mock_get_coconut_use_case,
            authentication_dependency=authentication_dependency,
        )

    @pytest.fixture
    def client(self, app) -> TestClient:
        return TestClient(app)

    def test_should_be_200(self, client, mock_get_coconut_use_case, sample_coconut_id, authentication_headers):
        coconut = Coconut(id=sample_coconut_id)
        mock_get_coconut_use_case.execute.return_value = coconut

        response = client.get(f"/coconut/{sample_coconut_id}", headers=authentication_headers)

        assert_that(response.status_code).is_equal_to(200)

    def test_should_get_coconut(self, client, mock_get_coconut_use_case, sample_coconut_id, authentication_headers):
        coconut = Coconut(id=sample_coconut_id)
        mock_get_coconut_use_case.execute.return_value = coconut

        response = client.get(f"/coconut/{sample_coconut_id}", headers=authentication_headers)

        assert_that(response.json()["id"]).is_equal_to(str(sample_coconut_id))

    def test_should_be_404(self, client, mock_get_coconut_use_case, sample_coconut_id, authentication_headers):
        mock_get_coconut_use_case.execute.side_effect = Exception("Coconut not found")

        response = client.get(f"/coconut/{sample_coconut_id}", headers=authentication_headers)

        assert_that(response.status_code).is_equal_to(404)
```

**Observations:**
- HTTP status code checks in separate tests from body checks
- Authentication fixtures for security testing
- FastAPI TestClient for integration testing
- Mock use cases, not repositories (layer separation)
- Each aspect tested separately: status code, response body, error handling

## DTO Test Example

From `tests/python_sprint_zero/interface/api/data_transfer_object/data_transfer_object/test_coconut_data_transfer_object.py`:

```python
class TestCoconutDataTransferObject:
    def test_should_have_id_when_creating_request_from_domain_model(self):
        coconut_id = uuid.uuid4()
        coconut = Coconut(id=coconut_id)

        data_transfer_object = CoconutApiRequestDataTransferObject.from_domain_model(coconut)

        assert_that(data_transfer_object.id).is_equal_to(coconut_id)

    def test_should_support_none_id_when_creating_request_from_domain_model(self):
        coconut = Coconut(id=None)

        data_transfer_object = CoconutApiRequestDataTransferObject.from_domain_model(coconut)

        assert_that(data_transfer_object.id).is_none()

    def test_should_raise_error_if_none_id_when_creating_response_from_domain_model(self):
        coconut = Coconut(id=None)

        with pytest.raises(ValueError) as excinfo:
            CoconutApiResponseDataTransferObject.from_domain_model(coconut)

        assert_that(str(excinfo.value)).contains("cannot be None")
```

**Observations:**
- Tests conversion logic in DTOs
- Separate tests for different scenarios (valid id, None id, error cases)
- Tests both request and response DTOs
- Exception messages verified

## Repository Implementation Test

From `tests/python_sprint_zero/infrastructure/persistence/in_memory/test_in_memory_coconut_query_repository.py`:

```python
class TestInMemoryCoconutQueryRepository:
    @pytest.fixture
    def storage(self):
        return {}

    @pytest.fixture
    def repository(self, storage):
        return InMemoryCoconutQueryRepository(storage)

    def test_should_return_coconut_when_exists_in_storage(self, repository, storage, sample_coconut_id):
        coconut = Coconut(id=sample_coconut_id)
        storage[sample_coconut_id] = coconut

        result = repository.read(sample_coconut_id)

        assert_that(result).is_equal_to(coconut)

    def test_should_raise_exception_when_coconut_not_in_storage(self, repository, sample_coconut_id):
        with pytest.raises(Exception) as excinfo:
            repository.read(sample_coconut_id)

        assert_that(str(excinfo.value)).is_equal_to(f"Coconut with id {sample_coconut_id} not found")
```

**Observations:**
- Storage fixture provides isolation
- Tests both success and failure paths
- Exception messages verified
- Real implementation tested, not mocked

## Pattern: Testing Multiple Behaviors Requires Multiple Tests

**WRONG - Multiple Assertions:**
```python
def test_create_coconut(self):
    use_case = CreateCoconutUseCase(repository)
    result_id = use_case.execute(sample_id)

    assert result_id == sample_id  # ❌ Multiple assertions
    repository.create.assert_called_once()  # ❌
    assert repository.call_args[0][0].id == sample_id  # ❌
```

**CORRECT - Separate Tests:**
```python
def test_should_return_id_from_repository(self, use_case, mock_repository, sample_id):
    mock_repository.create.return_value = sample_id

    result = use_case.execute(sample_id)

    assert_that(result).is_equal_to(sample_id)

def test_should_call_repository_create_method(self, use_case, mock_repository, sample_id):
    mock_repository.create.return_value = sample_id

    use_case.execute(sample_id)

    mock_repository.create.assert_called_once()

def test_should_pass_coconut_with_correct_id_to_repository(self, use_case, mock_repository, sample_id):
    mock_repository.create.return_value = sample_id

    use_case.execute(sample_id)

    arg = mock_repository.create.call_args[0][0]
    assert_that(arg.id).is_equal_to(sample_id)
```

## Common Fixtures

From `tests/python_sprint_zero/conftest.py`:

```python
@pytest.fixture
def sample_coconut_id():
    return uuid.UUID("12345678-1234-1234-1234-123456789012")
```

Shared fixtures help maintain consistency across tests while keeping tests readable.
