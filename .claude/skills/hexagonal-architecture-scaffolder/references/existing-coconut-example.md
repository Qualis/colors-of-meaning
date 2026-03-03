# Existing Coconut Feature Example

This document shows the complete structure of the existing `coconut` feature as a reference implementation of the hexagonal architecture pattern.

## File Structure

```
src/python_sprint_zero/
├── domain/
│   ├── model/
│   │   └── coconut.py                                    # Domain entity
│   └── repository/
│       └── coconut_repository.py                         # Repository interfaces (ABC)
├── infrastructure/
│   └── persistence/
│       └── in_memory/
│           ├── in_memory_coconut_command_repository.py  # Command repo implementation
│           ├── in_memory_coconut_query_repository.py    # Query repo implementation
│           └── shared_storage.py                         # Shared storage
├── application/
│   └── use_case/
│       └── coconut_use_case.py                          # Use cases (Get, Create)
└── interface/
    └── api/
        ├── controller/
        │   └── coconut_controller.py                     # FastAPI controller
        └── data_transfer_object/
            └── coconut_data_transfer_object.py           # Request/Response DTOs

tests/python_sprint_zero/
├── domain/
│   ├── model/
│   │   └── test_coconut.py
│   └── repository/
│       ├── test_coconut_command_repository.py
│       └── test_coconut_query_repository.py
├── infrastructure/
│   └── persistence/
│       └── in_memory/
│           ├── test_in_memory_coconut_command_repository.py
│           ├── test_in_memory_coconut_query_repository.py
│           └── test_shared_storage.py
├── application/
│   └── use_case/
│       └── test_coconut_use_case.py
└── interface/
    └── api/
        ├── controller/
        │   └── test_coconut_controller.py
        └── data_transfer_object/
            └── data_transfer_object/
                └── test_coconut_data_transfer_object.py
```

## Key Implementation Details

### Domain Layer
- **coconut.py**: Simple Pydantic model with `id` field
- **coconut_repository.py**: Defines `CoconutQueryRepository` and `CoconutCommandRepository` as ABCs

### Infrastructure Layer
- Separate query and command repository implementations
- Uses `shared_storage.py` for in-memory persistence
- Both repositories inject storage dict via constructor

### Application Layer
- `GetCoconutUseCase`: Takes query repository, orchestrates read operation
- `CreateCoconutUseCase`: Takes command repository, orchestrates create operation
- Both use constructor injection for dependencies

### Interface Layer
- **DTOs**: Separate request (Optional id) and response (required id) models
- **Controller**: Uses FastAPI APIRouter, injects both use cases and authentication
- Error handling converts domain exceptions to appropriate HTTP status codes

## Pattern Observations

1. **CQRS Separation**: Query and command repositories are separate
2. **Shared Storage**: In-memory implementations share a dict for storage
3. **Use Case Separation**: Separate use case classes for each operation
4. **Error Translation**: Controller translates domain exceptions to HTTP exceptions
5. **Authentication**: Injected as a callable dependency
6. **Location Header**: POST returns Location header with new resource path

## Test Patterns

- Tests mirror source structure exactly
- Each test has one assertion
- Test names follow `test_should_X_when_Y` pattern
- Use mocks for dependencies (unittest.mock)
- Use assertpy for assertions
- Arrange-Act-Assert structure

## Import Flow

```
Interface → Application → Domain
    ↓           ↓
Infrastructure
```

No upward imports (infrastructure/interface cannot be imported by domain/application).
