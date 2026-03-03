---
name: Hexagonal Architecture Feature Scaffolder
description: This skill should be used when the user asks to "create a new feature", "scaffold a feature", "add a new domain entity", "create a new API endpoint", "add a new resource", or "implement a new use case". This skill guides Claude through creating features following the project's hexagonal architecture with proper layer separation.
version: 1.0.0
---

# Hexagonal Architecture Feature Scaffolder

## Overview

This skill guides you through scaffolding complete features across all layers of the hexagonal architecture used in this Python Sprint Zero project. It ensures proper layer separation, dependency injection, and adherence to project standards.

## When to Use This Skill

Use this skill when implementing any new feature that requires:
- New domain entities or business logic
- New API endpoints
- New use cases or workflows
- Complete vertical slices through the architecture

## Architecture Layers

The project follows hexagonal architecture with these layers:

1. **Domain Layer** (`domain/`) - Pure business logic, no external dependencies
2. **Application Layer** (`application/`) - Use case orchestration
3. **Infrastructure Layer** (`infrastructure/`) - Technical implementations (persistence, security, etc.)
4. **Interface Layer** (`interface/`) - API controllers and DTOs
5. **Shared Layer** (`shared/`) - Cross-cutting concerns

## Step-by-Step Scaffolding Process

### Step 1: Create Domain Model

**Location:** `src/python_sprint_zero/domain/model/{feature_name}.py`

Create a Pydantic model representing your domain entity:

```python
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

class FeatureName(BaseModel):
    id: Optional[UUID] = Field(default=None)
    # Add your domain fields here
```

**Rules:**
- Use Pydantic BaseModel for validation
- Keep pure business logic only
- No imports from application, infrastructure, or interface layers
- No external framework dependencies (no FastAPI, databases, etc.)

### Step 2: Create Repository Interface

**Location:** `src/python_sprint_zero/domain/repository/{feature_name}_repository.py`

Define the abstract repository interface:

```python
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID
from python_sprint_zero.domain.model.{feature_name} import FeatureName

class FeatureNameQueryRepository(ABC):
    @abstractmethod
    def read(self, id: UUID) -> FeatureName:
        pass

class FeatureNameCommandRepository(ABC):
    @abstractmethod
    def create(self, entity: FeatureName) -> UUID:
        pass
```

**Rules:**
- Separate query and command repositories (CQRS pattern)
- Use ABC for abstract base classes
- Define only method signatures, no implementation

### Step 3: Create Repository Implementation

**Location:** `src/python_sprint_zero/infrastructure/persistence/in_memory/in_memory_{feature_name}_repository.py`

Implement the repository interface:

```python
from typing import Optional
from uuid import UUID
from python_sprint_zero.domain.model.{feature_name} import FeatureName
from python_sprint_zero.domain.repository.{feature_name}_repository import (
    FeatureNameQueryRepository,
    FeatureNameCommandRepository
)

class InMemoryFeatureNameQueryRepository(FeatureNameQueryRepository):
    def __init__(self, storage: dict):
        self._storage = storage

    def read(self, id: UUID) -> FeatureName:
        if id not in self._storage:
            raise Exception(f"FeatureName with id {id} not found")
        return self._storage[id]

class InMemoryFeatureNameCommandRepository(FeatureNameCommandRepository):
    def __init__(self, storage: dict):
        self._storage = storage

    def create(self, entity: FeatureName) -> UUID:
        if entity.id in self._storage:
            raise Exception(f"FeatureName with id {entity.id} already exists")
        self._storage[entity.id] = entity
        return entity.id
```

**Rules:**
- Implement the domain repository interfaces
- Handle technical concerns (storage, caching, etc.)
- May import from domain and application layers

### Step 4: Create Use Cases

**Location:** `src/python_sprint_zero/application/use_case/{feature_name}_use_case.py`

Create use cases that orchestrate the workflow:

```python
from uuid import UUID
from python_sprint_zero.domain.model.{feature_name} import FeatureName
from python_sprint_zero.domain.repository.{feature_name}_repository import (
    FeatureNameQueryRepository,
    FeatureNameCommandRepository
)

class GetFeatureNameUseCase:
    def __init__(self, query_repository: FeatureNameQueryRepository):
        self._query_repository = query_repository

    def execute(self, id: UUID) -> FeatureName:
        return self._query_repository.read(id)

class CreateFeatureNameUseCase:
    def __init__(self, command_repository: FeatureNameCommandRepository):
        self._command_repository = command_repository

    def execute(self, id: Optional[UUID] = None) -> UUID:
        import uuid
        if id is None:
            id = uuid.uuid4()
        entity = FeatureName(id=id)
        return self._command_repository.create(entity)
```

**Rules:**
- Orchestrate domain logic, don't implement it
- Use dependency injection (constructor injection)
- Depend on repository interfaces, not implementations
- No direct database or framework dependencies

### Step 5: Create Data Transfer Objects (DTOs)

**Location:** `src/python_sprint_zero/interface/api/data_transfer_object/{feature_name}_data_transfer_object.py`

Create Pydantic models for API requests/responses:

```python
from pydantic import UUID4, BaseModel, Field
from typing import Optional, Any

class FeatureNameApiRequestDataTransferObject(BaseModel):
    id: Optional[UUID4] = Field(default=None)

    @classmethod
    def from_domain_model(cls, domain_model: Any) -> "FeatureNameApiRequestDataTransferObject":
        return cls(id=getattr(domain_model, "id", None))

class FeatureNameApiResponseDataTransferObject(BaseModel):
    id: UUID4 = Field(...)

    @classmethod
    def from_domain_model(cls, domain_model: Any) -> "FeatureNameApiResponseDataTransferObject":
        id_value = getattr(domain_model, "id", None)
        if id_value is None:
            raise ValueError("Domain model id cannot be None")
        return cls(id=id_value)
```

**Rules:**
- Use Pydantic for validation
- Separate request and response DTOs
- Provide `from_domain_model` class methods for conversion
- Handle validation errors appropriately

### Step 6: Create Controller

**Location:** `src/python_sprint_zero/interface/api/controller/{feature_name}_controller.py`

Create FastAPI controller with routes:

```python
from typing import Annotated, Callable, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBasicCredentials

from python_sprint_zero.application.use_case.{feature_name}_use_case import (
    GetFeatureNameUseCase,
    CreateFeatureNameUseCase
)
from python_sprint_zero.interface.api.data_transfer_object.{feature_name}_data_transfer_object import (
    FeatureNameApiRequestDataTransferObject,
    FeatureNameApiResponseDataTransferObject
)

class FeatureNameController:
    def __init__(
        self,
        get_use_case: GetFeatureNameUseCase,
        create_use_case: CreateFeatureNameUseCase,
        authentication_dependency: Callable[[Optional[HTTPBasicCredentials]], None]
    ):
        self.router = APIRouter(prefix="/{feature_name}", tags=["{feature_name}"])
        self._get_use_case = get_use_case
        self._create_use_case = create_use_case
        self._authentication_dependency = authentication_dependency

        self.router.add_api_route(
            "/{id}",
            self.get,
            methods=["GET"],
            response_model=FeatureNameApiResponseDataTransferObject,
            dependencies=[Depends(authentication_dependency)]
        )

        self.router.add_api_route(
            "/",
            self.create,
            methods=["POST"],
            status_code=status.HTTP_201_CREATED,
            dependencies=[Depends(authentication_dependency)]
        )

    async def get(self, id: UUID):
        try:
            result = self._get_use_case.execute(id)
            return FeatureNameApiResponseDataTransferObject.from_domain_model(result)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"FeatureName with id {id} not found"
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving feature"
            )

    async def create(self, request: FeatureNameApiRequestDataTransferObject):
        try:
            result_id = self._create_use_case.execute(request.id)
            return {"Location": f"/{feature_name}/{result_id}"}
        except Exception as e:
            if "already exists" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="FeatureName already exists"
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Error creating feature"
            )
```

**Rules:**
- Use FastAPI's APIRouter
- Inject use cases via constructor
- Handle errors and return appropriate HTTP status codes
- Add authentication dependencies
- Use DTOs for request/response transformation

### Step 7: Add __init__.py Files

Ensure each new package directory has an `__init__.py` file:

```bash
# If creating new directories, add __init__.py
touch src/python_sprint_zero/domain/model/__init__.py
touch src/python_sprint_zero/domain/repository/__init__.py
touch src/python_sprint_zero/infrastructure/persistence/in_memory/__init__.py
touch src/python_sprint_zero/application/use_case/__init__.py
touch src/python_sprint_zero/interface/api/controller/__init__.py
touch src/python_sprint_zero/interface/api/data_transfer_object/__init__.py
```

### Step 8: Create Corresponding Test Files

For each source file created, create a corresponding test file with matching structure:

**Test structure mirrors source structure:**
- `tests/python_sprint_zero/domain/model/test_{feature_name}.py`
- `tests/python_sprint_zero/domain/repository/test_{feature_name}_repository.py`
- `tests/python_sprint_zero/infrastructure/persistence/in_memory/test_in_memory_{feature_name}_repository.py`
- `tests/python_sprint_zero/application/use_case/test_{feature_name}_use_case.py`
- `tests/python_sprint_zero/interface/api/controller/test_{feature_name}_controller.py`
- `tests/python_sprint_zero/interface/api/data_transfer_object/data_transfer_object/test_{feature_name}_data_transfer_object.py`

**Important:** Use the Test Generator skill to create tests following the one-assertion-per-test rule.

### Step 9: Update Dependency Injection Container

Wire up the new components in the Lagom container (typically in `interface/api/main.py` or a dedicated container setup file):

```python
from lagom import Container
from python_sprint_zero.domain.repository.{feature_name}_repository import (
    FeatureNameQueryRepository,
    FeatureNameCommandRepository
)
from python_sprint_zero.infrastructure.persistence.in_memory.in_memory_{feature_name}_repository import (
    InMemoryFeatureNameQueryRepository,
    InMemoryFeatureNameCommandRepository
)

container = Container()
container[FeatureNameQueryRepository] = InMemoryFeatureNameQueryRepository
container[FeatureNameCommandRepository] = InMemoryFeatureNameCommandRepository
```

### Step 10: Register Controller Routes

Add the new controller to the FastAPI application:

```python
from python_sprint_zero.interface.api.controller.{feature_name}_controller import FeatureNameController

# In main.py
controller = FeatureNameController(
    get_use_case=container[GetFeatureNameUseCase],
    create_use_case=container[CreateFeatureNameUseCase],
    authentication_dependency=security_dependency.authentication_dependency()
)
app.include_router(controller.router)
```

## Critical Rules to Follow

1. **NO COMMENTS** - Code must be self-documenting through expressive naming
2. **Layer Boundaries** - Respect import restrictions:
   - Domain CANNOT import from application, infrastructure, or interface
   - Application CANNOT import from infrastructure or interface
   - Infrastructure CAN import from domain and application
   - Interface CAN import from all layers
3. **Dependency Injection** - Always use constructor injection, never direct instantiation
4. **One Assertion Per Test** - When creating tests, ensure each test has exactly one assertion
5. **100% Test Coverage** - Every function must have tests
6. **Type Hints** - Use type hints for all function signatures

## Verification Checklist

After scaffolding, verify:

- [ ] All files created in correct layer directories
- [ ] No layer boundary violations (check imports)
- [ ] All `__init__.py` files exist
- [ ] Repository interface in domain, implementation in infrastructure
- [ ] Use cases use dependency injection
- [ ] DTOs separate from domain models
- [ ] Controller uses use cases, not repositories directly
- [ ] Tests created for all components
- [ ] Tests follow one-assertion-per-test rule
- [ ] All tests pass: `tox`
- [ ] Code formatted: `tox -e format`

## References

See the `references/` directory for:
- Example features from the codebase
- Layer dependency rules
- Common patterns and anti-patterns

## Common Mistakes to Avoid

1. **Direct instantiation** - Always use DI container
2. **Domain importing infrastructure** - Keep domain pure
3. **DTOs in domain** - DTOs belong in interface layer
4. **Business logic in controllers** - Logic belongs in domain/application
5. **Multiple assertions in tests** - Split into separate test functions
6. **Missing __init__.py** - Required for Python packages
7. **Skipping tests** - 100% coverage is mandatory
8. **Adding comments** - Refactor to self-documenting code instead
