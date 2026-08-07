# Existing Feature Example: Query by Color Palette

The `query by palette` feature is the reference vertical slice. It touches every layer, has both
an API and a CLI entry point, and is fully covered by tests — so it shows the real shape of a
feature in this codebase rather than an idealised one.

It answers: *given a set of Lab colors with weights, which documents in the corpus have the
closest color distribution?*

## File Structure

```
src/colors_of_meaning/
├── domain/
│   ├── model/
│   │   ├── lab_color.py                            # Value object, validates L/a/b ranges
│   │   ├── color_codebook.py                       # Frozen dataclass, quantize() to a bin
│   │   └── colored_document.py                     # Histogram + document_id
│   ├── repository/
│   │   └── color_codebook_repository.py            # ABC: save / load / exists / delete
│   └── service/
│       └── distance_calculator.py                  # Port for histogram distance
├── application/
│   └── use_case/
│       ├── query_by_palette_use_case.py            # Palette → histogram → nearest neighbours
│       └── compare_documents_use_case.py           # Reused for the neighbour search
├── infrastructure/
│   ├── ml/
│   │   └── wasserstein_distance_calculator.py      # DistanceCalculator via POT
│   └── persistence/
│       ├── file_color_codebook_repository.py       # ColorCodebookRepository on disk
│       └── in_memory/
│           └── in_memory_color_codebook_repository.py
└── interface/
    ├── api/
    │   ├── main.py                                 # Lagom container, composed once at import
    │   ├── controller/
    │   │   └── query_controller.py                 # Router factory, closes over its use case
    │   └── data_transfer_object/
    │       └── palette_query_dto.py                # Request/response Pydantic models
    └── cli/
        └── query.py                                # tyro dataclass CLI

tests/colors_of_meaning/
├── domain/model/
│   ├── test_lab_color.py
│   ├── test_color_codebook.py
│   └── test_colored_document.py
├── application/use_case/
│   └── test_query_by_palette_use_case.py
├── infrastructure/
│   ├── ml/test_wasserstein_distance_calculator.py
│   └── persistence/
│       ├── test_file_color_codebook_repository.py
│       └── in_memory/test_in_memory_color_codebook_repository.py
└── interface/
    ├── api/
    │   ├── controller/test_query_controller.py
    │   └── data_transfer_object/test_palette_query_dto.py
    └── cli/test_query.py
```

## Key Implementation Details

### Domain Layer

- `LabColor` validates its own ranges (L 0–100, a and b −128–127) and raises on violation.
- `ColorCodebook` is a frozen dataclass holding `colors: List[LabColor]` and `num_bins`, with a
  `cached_property` for the vectorised palette coordinates. It validates in `__post_init__` that
  the color count matches `num_bins`.
- `ColorCodebookRepository` and `DistanceCalculator` are ABCs whose methods `raise
  NotImplementedError`. Because they are ports, `numpy`/`POT`/`torch` stay out of the domain.

### Application Layer

```python
class QueryByPaletteUseCase:
    def __init__(self, compare_use_case: CompareDocumentsUseCase, codebook: ColorCodebook) -> None:
        self.compare_use_case = compare_use_case
        self.codebook = codebook

    def execute(
        self,
        palette: List[Tuple[LabColor, float]],
        corpus_docs: List[ColoredDocument],
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        query_doc = self._palette_to_document(palette)
        return self.compare_use_case.find_nearest_neighbors(query_doc, corpus_docs, k)
```

The use case turns the palette into a `ColoredDocument` and then **delegates** the neighbour
search to `CompareDocumentsUseCase` instead of reimplementing it. Its dependencies arrive through
the constructor; it never constructs them.

### Infrastructure Layer

- `WassersteinDistanceCalculator` implements `DistanceCalculator` over a perceptual Lab cost
  matrix via `ot.emd2`. Swapping in `SlicedWassersteinDistanceCalculator` or
  `JensenShannonDistanceCalculator` changes nothing above it.
- Two adapters back the same repository port: file-based for real runs, in-memory for tests.

### Interface Layer

- **DTOs carry the validation at the edge**: `PaletteColorDTO` constrains `l`, `a`, `b` and
  `weight` with Pydantic `Field(ge=..., le=...)`, and `PaletteQueryRequestDTO` requires at least
  one color. The controller maps DTO → `LabColor`; the domain entity never sees a DTO.
- **Controllers are factories, not module-level routers**:

```python
def create_query_controller(
    query_use_case: QueryByPaletteUseCase,
    corpus_docs: List[ColoredDocument],
) -> APIRouter:
    router = APIRouter(tags=["query"])

    async def query_by_palette(request: PaletteQueryRequestDTO) -> PaletteQueryResponseDTO:
        ...

    router.add_api_route("/query/palette", query_by_palette, methods=["POST"], ...)
    return router
```

  Dependencies are closed over by the factory rather than resolved per-request with `Depends()`.
  `api/main.py` builds the Lagom container once at import and passes the resolved use case in.

- **Degrade rather than fail to boot**: when the trained artifacts are absent,
  `create_unavailable_query_controller(detail)` registers the same route returning a
  `QueryUnavailableDTO`, so the service still starts and its health endpoints still answer.

## Pattern Observations

1. **Port in domain, adapter in infrastructure.** Every third-party library reaches the domain
   only through an ABC.
2. **Constructor injection everywhere.** Nothing instantiates its own collaborators; the Lagom
   container maps abstractions to implementations in `api/main.py`.
3. **Use cases compose.** A new use case delegates to existing ones rather than duplicating them.
4. **DTOs never cross inward.** Pydantic models live in `interface/`, and the controller
   translates at the boundary.
5. **Two adapters per port where it helps.** A real one and an in-memory one keeps unit tests
   free of I/O.
6. **Graceful degradation.** Missing artifacts produce a documented unavailable response, not an
   import-time crash.
7. **Both entry points, one use case.** The API controller and the tyro CLI call the same
   `QueryByPaletteUseCase`.

## Test Patterns

- Tests mirror the source tree exactly.
- One logical assertion per test; related assertions on the same ML result are acceptable.
- Names follow `test_should_<behaviour>_when_<condition>`.
- `assertpy`'s `assert_that` for entity and DTO tests; plain `assert` and `pytest.raises` are
  acceptable for ML and domain-specific tests.
- Dependencies are mocked with `unittest.mock`; no network calls, no dataset downloads.
- Arrange-Act-Assert structure.
- Architectural rules are themselves tested, with `pytest-archon`.

## Import Flow

```
Interface → Application → Domain
    ↓            ↓
Infrastructure ──┘
```

Infrastructure and interface may import inward. Domain imports nothing but `shared`, and
application imports only `domain` and `shared`. There are no upward imports.
