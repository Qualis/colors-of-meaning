# Self-Documenting Code: Real Examples from Codebase

This document provides concrete examples from the Python Sprint Zero codebase demonstrating the no-comments policy in practice. All examples are taken from actual production code.

## Table of Contents
1. [Pattern 1: Expressive Method Names](#pattern-1-expressive-method-names)
2. [Pattern 2: Descriptive Variable Names](#pattern-2-descriptive-variable-names)
3. [Pattern 3: Clear Error Messages](#pattern-3-clear-error-messages)
4. [Pattern 4: Small, Single-Purpose Functions](#pattern-4-small-single-purpose-functions)
5. [Pattern 5: Boolean Checks with Clear Intent](#pattern-5-boolean-checks-with-clear-intent)
6. [Before/After Refactoring Examples](#beforeafter-refactoring-examples)
7. [Project-Specific Patterns](#project-specific-patterns)

---

## Pattern 1: Expressive Method Names

### Example: Use Case Methods

**Location:** `src/python_sprint_zero/application/use_case/coconut_use_case.py:8-13`

```python
class GetCoconutUseCase:
    def __init__(self, query_repository: CoconutQueryRepository) -> None:
        self._query_repository = query_repository

    def execute(self, coconut_id: uuid.UUID) -> Coconut:
        return self._query_repository.read(coconut_id)
```

**Why This Works:**
- `GetCoconutUseCase` - Class name clearly states purpose
- `execute()` - Standard use case method name
- `query_repository` - Type hint + name explains dependency
- `read()` - Repository method that clearly describes action

**What This Replaces:**
No need for comments like "Get coconut from repository" or "Execute use case logic" - the names say it all.

---

### Example: Authentication Methods

**Location:** `src/python_sprint_zero/infrastructure/security/basic_authentication.py:19-25`

```python
def verify_credentials(self, username: str, password: str) -> bool:
    stored_password = self.user_credentials.get(username)

    if stored_password is None:
        return False

    return stored_password == password
```

**Why This Works:**
- `verify_credentials()` - Action-oriented name
- `stored_password` - Clear what this variable represents
- Return type `bool` - Makes success/failure obvious
- No magic strings or numbers

---

### Example: Controller Route Registration

**Location:** `src/python_sprint_zero/interface/api/controller/coconut_controller.py:29-47`

```python
def _register_routes(self) -> None:
    dependencies = [Depends(self.authentication_dependency)] if self.authentication_dependency else []

    self.router.add_api_route(
        "/{id}",
        self.get_coconut,
        methods=["GET"],
        response_model=CoconutApiResponseDataTransferObject,
        dependencies=dependencies,
    )
```

**Why This Works:**
- `_register_routes()` - Private method with clear purpose
- `dependencies` - Conditional logic with descriptive name
- Named parameters in `add_api_route()` - Self-documenting configuration

---

## Pattern 2: Descriptive Variable Names

### Example: Authentication Setup

**Location:** `src/python_sprint_zero/infrastructure/security/basic_authentication.py:55-64`

```python
def get_basic_authenticator() -> BasicAuthenticator:
    authenticator = BasicAuthenticator()

    setting_provider = get_application_setting_provider()
    admin_username = setting_provider.get("admin")
    admin_password = setting_provider.get("password")

    authenticator.register_user(admin_username, admin_password)

    return authenticator
```

**Why This Works:**
- `setting_provider` - Describes what the object provides
- `admin_username` / `admin_password` - Context included in name
- Each variable has a clear, specific purpose
- No abbreviations or cryptic names

---

### Example: Response Creation

**Location:** `src/python_sprint_zero/interface/api/controller/coconut_controller.py:66-69`

```python
created_id = self.create_coconut_use_case.execute(data_transfer_object.id)

response = Response(status_code=status.HTTP_201_CREATED)
response.headers["Location"] = f"/coconut/{created_id}"
```

**Why This Works:**
- `created_id` - Past tense indicates this is the result
- `response` - Clear what's being built
- Named constant `status.HTTP_201_CREATED` - Self-documenting status code

---

## Pattern 3: Clear Error Messages

### Example: Repository Validation

**Location:** `src/python_sprint_zero/infrastructure/persistence/in_memory/in_memory_coconut_query_repository.py:12-20`

```python
def read(self, id: uuid.UUID) -> Coconut:
    if not isinstance(id, uuid.UUID):
        raise ValueError("Invalid UUID")

    coconut = self._storage.get_coconut(id)

    if coconut is None:
        raise Exception("Coconut not found")

    return coconut
```

**Why This Works:**
- Error messages describe the problem clearly
- No comment needed to explain validation logic
- Explicit type checking with descriptive error
- Early returns keep logic simple

---

### Example: Authentication Errors

**Location:** `src/python_sprint_zero/infrastructure/security/basic_authentication.py:35-40`

```python
if not credentials:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Basic"},
    )
```

**Why This Works:**
- `detail="Authentication required"` - User-friendly message
- Named status code - `status.HTTP_401_UNAUTHORIZED`
- Standard HTTP headers - Self-documenting protocol

---

## Pattern 4: Small, Single-Purpose Functions

### Example: CQRS Separation

**Location:** `src/python_sprint_zero/application/use_case/coconut_use_case.py`

```python
class GetCoconutUseCase:
    def __init__(self, query_repository: CoconutQueryRepository) -> None:
        self._query_repository = query_repository

    def execute(self, coconut_id: uuid.UUID) -> Coconut:
        return self._query_repository.read(coconut_id)


class CreateCoconutUseCase:
    def __init__(self, command_repository: CoconutCommandRepository) -> None:
        self._command_repository = command_repository

    def execute(self, coconut_id: Optional[uuid.UUID] = None) -> uuid.UUID:
        if coconut_id is None:
            coconut_id = uuid.uuid4()

        coconut = Coconut(id=coconut_id)
        return self._command_repository.create(coconut)
```

**Why This Works:**
- Separate classes for Query vs Command (CQRS pattern)
- Each class has ONE responsibility
- Short methods that fit on screen
- Clear flow without comments

---

### Example: Factory Function

**Location:** `src/python_sprint_zero/interface/api/controller/coconut_controller.py:84-94`

```python
def create_coconut_controller(
    container: Container,
    authentication_dependency: Optional[Callable[[Optional[HTTPBasicCredentials]], None]] = None
) -> CoconutController:
    get_coconut_use_case = container[GetCoconutUseCase]
    create_coconut_use_case = container[CreateCoconutUseCase]

    return CoconutController(
        get_coconut_use_case=get_coconut_use_case,
        create_coconut_use_case=create_coconut_use_case,
        authentication_dependency=authentication_dependency,
    )
```

**Why This Works:**
- Factory function name describes what it creates
- Extracts from container with explicit types
- Named parameters make constructor call clear
- No comment explaining dependency injection pattern

---

## Pattern 5: Boolean Checks with Clear Intent

### Example: Credential Verification

**Location:** `src/python_sprint_zero/infrastructure/security/basic_authentication.py:32-47`

```python
def require_authentication(
    self, credentials: Optional[HTTPBasicCredentials] = Depends(basic_authentication)
) -> None:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )

    if not self.authenticator.verify_credentials(credentials.username, credentials.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
```

**Why This Works:**
- `if not credentials:` - Clear existence check
- `verify_credentials()` - Boolean method with clear name
- Guard clauses make validation flow obvious
- Different error messages for different failures

---

### Example: Null Checking

**Location:** `src/python_sprint_zero/infrastructure/persistence/in_memory/in_memory_coconut_query_repository.py:23-27`

```python
def add_to_storage(self, coconut: Coconut) -> None:
    if coconut.id is None:
        raise ValueError("Coconut ID cannot be None")

    self._storage.add_coconut(coconut)
```

**Why This Works:**
- Explicit `is None` check
- Descriptive error message explains constraint
- Guard clause pattern makes validation obvious

---

## Before/After Refactoring Examples

### Scenario 1: Use Case with Comments → Self-Documenting

**BEFORE (With Comments - FORBIDDEN):**
```python
class GetCoconutUseCase:
    def __init__(self, query_repository):
        # Store the repository for later use
        self.repo = query_repository

    def execute(self, id):
        # Call the repository to get the coconut
        return self.repo.read(id)
```

**AFTER (Self-Documenting - ACTUAL CODE):**
```python
class GetCoconutUseCase:
    def __init__(self, query_repository: CoconutQueryRepository) -> None:
        self._query_repository = query_repository

    def execute(self, coconut_id: uuid.UUID) -> Coconut:
        return self._query_repository.read(coconut_id)
```

**Improvements:**
- Type hints document types (no comment needed)
- `_query_repository` (private prefix + full name) vs `repo` (cryptic abbreviation)
- `coconut_id` (specific) vs `id` (generic)
- Return type `Coconut` documents what's returned

---

### Scenario 2: Error Handling with Comments → Descriptive Messages

**BEFORE (With Comments - FORBIDDEN):**
```python
def get_coconut(self, id):
    try:
        coconut = self.use_case.execute(id)
        return coconut
    except Exception as e:
        # Check if it's a not found error
        if "not found" in str(e).lower():
            # Return 404 status code
            raise HTTPException(status_code=404, detail=str(e))
        # Otherwise return 500
        raise HTTPException(status_code=500, detail=str(e))
```

**AFTER (Self-Documenting - ACTUAL CODE):**
```python
async def get_coconut(self, id: UUID4) -> CoconutApiResponseDataTransferObject:
    try:
        coconut = self.get_coconut_use_case.execute(id)
        return CoconutApiResponseDataTransferObject.from_domain_model(coconut)
    except Exception as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Coconut with id {id} not found",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving coconut: {str(e)}",
        )
```

**Improvements:**
- Named constants: `status.HTTP_404_NOT_FOUND` vs magic number `404`
- Descriptive error messages: "Coconut with id {id} not found"
- Type hints document async return type
- Method name `get_coconut_use_case` vs generic `use_case`

---

### Scenario 3: Conditional Logic with Comments → Boolean Variables

**BEFORE (With Comments - FORBIDDEN):**
```python
def _register_routes(self):
    # Create dependencies list with auth if provided
    if self.authentication_dependency:
        deps = [Depends(self.authentication_dependency)]
    else:
        deps = []

    # Register GET route
    self.router.add_api_route("/{id}", self.get_coconut, methods=["GET"], dependencies=deps)
```

**AFTER (Self-Documenting - ACTUAL CODE):**
```python
def _register_routes(self) -> None:
    dependencies = [Depends(self.authentication_dependency)] if self.authentication_dependency else []

    self.router.add_api_route(
        "/{id}",
        self.get_coconut,
        methods=["GET"],
        response_model=CoconutApiResponseDataTransferObject,
        dependencies=dependencies,
    )
```

**Improvements:**
- Variable named `dependencies` (full word) vs `deps` (abbreviation)
- Ternary expression is clear enough without comment
- Named parameters make route configuration self-documenting
- Type annotation `-> None` documents no return value

---

### Scenario 4: Authentication Logic with Comments → Expressive Names

**BEFORE (With Comments - FORBIDDEN):**
```python
def check_auth(self, creds):
    # Get the stored password for this user
    pwd = self.user_credentials.get(creds.username)

    # Check if user exists
    if pwd is None:
        return False

    # Compare passwords
    return pwd == creds.password
```

**AFTER (Self-Documenting - ACTUAL CODE):**
```python
def verify_credentials(self, username: str, password: str) -> bool:
    stored_password = self.user_credentials.get(username)

    if stored_password is None:
        return False

    return stored_password == password
```

**Improvements:**
- `verify_credentials` vs `check_auth` (more explicit)
- `stored_password` vs `pwd` (complete word)
- Direct parameters (`username`, `password`) instead of credential object
- Return type `-> bool` documents success/failure

---

## Project-Specific Patterns

### Pattern: CQRS Separation (Command Query Responsibility Segregation)

The project separates read operations from write operations using different repository interfaces:

```python
class CoconutQueryRepository(ABC):
    @abstractmethod
    def read(self, id: uuid.UUID) -> Coconut:
        pass

class CoconutCommandRepository(ABC):
    @abstractmethod
    def create(self, coconut: Coconut) -> uuid.UUID:
        pass
```

**Why No Comments Needed:**
- Suffix `QueryRepository` / `CommandRepository` - Documents purpose
- Method names `read()` / `create()` - Clear operations
- ABC base class - Indicates interface pattern
- Type hints document contracts

---

### Pattern: Dependency Injection with Type Hints

**Location:** Throughout codebase

```python
def __init__(self, query_repository: CoconutQueryRepository) -> None:
    self._query_repository = query_repository
```

**Why No Comments Needed:**
- Type hint `CoconutQueryRepository` documents expected interface
- Parameter name `query_repository` describes purpose
- Private prefix `_query_repository` indicates internal use
- Constructor pattern is standard dependency injection

---

### Pattern: Data Transfer Object (DTO) Conversion

**Location:** `src/python_sprint_zero/interface/api/data_transfer_object/`

```python
@classmethod
def from_domain_model(cls, domain_model: Any) -> "CoconutApiResponseDataTransferObject":
    id_value = getattr(domain_model, "id", None)
    if id_value is None:
        raise ValueError("Domain model id cannot be None")
    return cls(id=id_value)
```

**Why No Comments Needed:**
- Method name `from_domain_model` - Clear conversion direction
- Class method pattern - Standard factory pattern
- Explicit validation with descriptive error
- Return type annotation documents output

---

### Pattern: Error String Detection

**Location:** Multiple controllers

```python
if "not found" in str(e).lower():
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, ...)
elif "already exists" in str(e).lower():
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, ...)
```

**Why This Works:**
- String matching makes intent explicit
- Named status codes document HTTP semantics
- Different messages for different error types
- Pattern is consistent across controllers

---

## Quick Reference: When You're Tempted to Add a Comment

| Temptation | Refactoring Solution | Example from Codebase |
|------------|---------------------|----------------------|
| "This creates..." | Use `create_X()` or `build_X()` in name | `create_coconut_controller()` |
| "This validates..." | Use `verify_X()` or `validate_X()` | `verify_credentials()` |
| "This checks if..." | Extract to `is_X()` or `has_Y()` method | `if not credentials:` |
| "Get X from Y" | Use `get_X_from_Y()` or property | `get_coconut()` |
| "Register/setup..." | Use `register_X()` or `_setup_Y()` | `_register_routes()` |
| "Store for later" | Use descriptive name with context | `self._query_repository` |
| "Loop through..." | Extract to `process_each_X()` | (Not needed with clear naming) |
| "Magic number X means..." | Named constant | `status.HTTP_404_NOT_FOUND` |

---

## Anti-Patterns to Avoid

### ❌ Cryptic Abbreviations
```python
repo = get_repo()  # What kind of repo?
pwd = get_pwd()    # Password or present working directory?
dto = build_dto()  # What data is being transferred?
```

### ✅ Use Full, Descriptive Names
```python
query_repository = get_coconut_query_repository()
stored_password = get_stored_password()
response_data_transfer_object = build_coconut_response_dto()
```

---

### ❌ Generic Names
```python
def process(data):
    result = do_stuff(data)
    return result
```

### ✅ Use Specific, Context-Rich Names
```python
def verify_credentials(self, username: str, password: str) -> bool:
    stored_password = self.user_credentials.get(username)
    return stored_password == password
```

---

### ❌ Magic Numbers/Strings
```python
if response.status_code == 404:
    return None

time.sleep(5)  # Why 5?
```

### ✅ Named Constants with Context
```python
if response.status_code == status.HTTP_404_NOT_FOUND:
    return None

AUTHENTICATION_RETRY_DELAY_SECONDS = 5
time.sleep(AUTHENTICATION_RETRY_DELAY_SECONDS)
```

---

## Conclusion

The codebase demonstrates that well-named functions, variables, and classes eliminate the need for comments. Every example shown here is production code that:

1. **Compiles and runs** without comments
2. **Passes 100% test coverage** requirements
3. **Follows architectural rules** (hexagonal architecture, CQRS, DI)
4. **Remains maintainable** over time

When you're tempted to add a comment, refer back to these patterns and refactor instead.

## References

All examples taken from:
- `src/python_sprint_zero/application/use_case/coconut_use_case.py`
- `src/python_sprint_zero/interface/api/controller/coconut_controller.py`
- `src/python_sprint_zero/infrastructure/persistence/in_memory/in_memory_coconut_query_repository.py`
- `src/python_sprint_zero/infrastructure/security/basic_authentication.py`

See `.claude/CLAUDE.md` for full project rules and architectural guidelines.
