# Layer Dependency Rules

## Allowed Import Directions

```
┌─────────────┐
│   Shared    │ ← Can be imported by ANY layer
└─────────────┘

┌─────────────┐
│   Domain    │ ← NO imports from other layers (pure business logic)
└─────────────┘
       ↑
       │
┌─────────────┐
│ Application │ ← Can import: Domain, Shared
└─────────────┘   CANNOT import: Infrastructure, Interface
       ↑
       │
┌─────────────────────────────┐
│ Infrastructure │ Interface  │ ← Can import: Domain, Application, Shared
└─────────────────────────────┘   (Infrastructure and Interface are peers)
```

## Specific Rules

### Domain Layer (`domain/`)
**CAN import:**
- Python standard library
- Pydantic (for models)
- typing module
- Other domain modules
- shared/ modules

**CANNOT import:**
- application/
- infrastructure/
- interface/
- FastAPI
- Database libraries
- External APIs

**Why:** Domain must remain pure business logic, free from framework coupling.

### Application Layer (`application/`)
**CAN import:**
- domain/
- shared/
- Python standard library
- typing module

**CANNOT import:**
- infrastructure/
- interface/
- FastAPI
- Database libraries

**Why:** Use cases orchestrate domain logic but don't depend on technical implementation details.

### Infrastructure Layer (`infrastructure/`)
**CAN import:**
- domain/ (implements repository interfaces)
- application/
- shared/
- External libraries (databases, message queues, etc.)
- Python standard library

**CANNOT import:**
- interface/ (peers, not dependencies)

**Why:** Infrastructure provides technical implementations of domain abstractions.

### Interface Layer (`interface/`)
**CAN import:**
- domain/
- application/
- infrastructure/ (for DI container setup)
- shared/
- FastAPI
- External libraries
- Python standard library

**CANNOT import:**
- Nothing (top layer)

**Why:** Interface is the outermost layer, exposing the application to the outside world.

### Shared Layer (`shared/`)
**CAN import:**
- Python standard library
- Common utilities (resilience patterns, formatters)

**CANNOT import:**
- domain/
- application/
- infrastructure/
- interface/

**Why:** Shared is cross-cutting and must not depend on domain-specific logic.

## Validation

The project includes architectural tests using pytest-archon in `tests/python_sprint_zero/test_architecture.py` that enforce these rules automatically.

Run architectural tests:
```bash
pytest tests/python_sprint_zero/test_architecture.py -v
```

## Common Violations to Avoid

1. **Domain importing FastAPI**: Domain models should use Pydantic, not FastAPI types
2. **Domain importing infrastructure**: Repository implementations belong in infrastructure, not domain
3. **Application importing infrastructure**: Use cases depend on domain interfaces, not implementations
4. **DTOs in domain**: Data transfer objects belong in interface layer, domain has business entities
5. **Use cases in controllers**: Controllers should only coordinate, not contain business logic

## Dependency Injection Pattern

To respect layer boundaries while wiring up dependencies:

1. **Define interface in domain** (e.g., `CoconutRepository` ABC)
2. **Implement in infrastructure** (e.g., `InMemoryCoconutRepository`)
3. **Use cases depend on interface** (inject `CoconutRepository`, not implementation)
4. **DI container maps interface to implementation** (in interface layer)
5. **Controllers inject use cases** (not repositories directly)

This allows changing implementations without affecting domain or application layers.
