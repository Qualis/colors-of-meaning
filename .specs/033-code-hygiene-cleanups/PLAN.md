# Plan: Code hygiene cleanups

## Implementation Strategy

Four behaviour-preserving cleanups, each independently committable, holding 100% coverage
throughout: (1) one shared artifact loader with a single error type, replacing ~11
copy-pasted `_load_codebook` bodies and the `FileNotFoundError`-vs-`ValueError` divergence;
(2) training-loop `print()` → structured logger; (3) `hasattr`-only tests → behavioural
tests; (4) drop the `# type: ignore` masking a missing return annotation. This is the only
sequence member that changes `src/`, so every step is validated by `tox` at 100% coverage
and by the `pytest-archon` boundary suite. Prefer editing existing files; the single new
file is the shared loader (and its test).

## Layer Changes

### Shared Layer (`src/colors_of_meaning/shared/`)

- New artifact loader (recommended placement — a pure filesystem/deserialize helper): load a
  pickled codebook/corpus by path, raising a **single** documented error type when absent.
  One home, one error contract. (If, during implementation, the loader needs to return and
  validate a `ColorCodebook`, it moves to an application/infrastructure split that keeps
  `domain/` pure — decision recorded in the commit.)

### Interface Layer (`src/colors_of_meaning/interface/`)

- Replace the private `_load_codebook` (and duplicated corpus loaders) in `ablate`,
  `authorship`, `compass`, `compress`, `decode_image`, `encode`, `eval_suite`, `generate`,
  `grounding`, `query`, and any others found by `git grep 'def _load_codebook'`, with a call
  to the shared loader. Net effect: `compress.py`/`encode.py` stop raising `ValueError` for a
  missing artifact and raise the shared type like the rest.
- `visualize.py`: give `_create_classifier` a real return annotation; remove the
  `# type: ignore`.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `ml/pytorch_color_mapper.py`, `ml/structured_pytorch_color_mapper.py`,
  `ml/supervised_pytorch_color_mapper.py`: replace training `print()` with a module-level
  logger call (correlation-id, matching the existing convention). Keep the same information;
  only the sink changes.

### Tests

- Add a test for the shared loader: returns the object when the path exists; raises the
  single error type when absent (one assertion each).
- Replace the `hasattr` existence tests (in `test_figure_renderer.py` and the two other
  flagged files) with behavioural assertions, or remove those already covered by a concrete
  behavioural test — enumerated so coverage stays 100%.
- Any test that asserted the old `ValueError` from `compress`/`encode` is updated to the
  shared error type.

## Dependency Injection

No Lagom change. The shared loader is called at each CLI composition root where
`_load_codebook` was; the CLIs keep their existing hand-wired composition-root pattern.

## Task List

Inside-out order (shared → infrastructure → interface → tests):

1. [ ] shared: add the artifact loader (single error type) + its test (found / missing).
2. [ ] interface: replace every private `_load_codebook`/corpus-load duplicate with the
   shared loader; update the `compress`/`encode` missing-artifact error to the shared type;
   update any test asserting the old `ValueError`.
3. [ ] infrastructure: training `print()` → structured logger in the three ML mappers.
4. [ ] interface: real return annotation on `visualize.py:_create_classifier`; remove the
   `# type: ignore`.
5. [ ] tests: replace/remove the `hasattr` existence tests with behavioural ones; keep
   coverage at 100%.
6. [ ] verify: full `tox` green (8 gates, 100% coverage); `pytest-archon` passes; `git grep
   'def _load_codebook'` and `git grep 'print('` (in `infrastructure/ml`) are clean.

## Testing Strategy

- **Shared loader:** one test returns the deserialized object for an existing `tmp_path`
  artifact; one asserts the single error type for a missing path (`pytest.raises`).
- **CLI refactor:** existing CLI tests must still pass unchanged except where they asserted
  the old `ValueError` (updated to the shared type) — this is the behaviour-preservation
  proof.
- **Logger swap:** assert the logger is called (or a log record is emitted) rather than
  capturing stdout; keep the line covered.
- **Behavioural replacements:** each new interface test asserts an observable outcome
  (`test_should_..._when_...`, one assertion), replacing an existence check.
- **Coverage:** removing tests must not drop coverage — verify the concrete behaviour was
  already covered before deleting an existence test; add a behavioural test first if not.
- Verify with `tox`, never `pytest` directly; keep new helpers grade-A for xenon (scans
  tests).

## Observability Plan

Training progress emitted via the structured logger with correlation-id (replacing
`print()`), consistent with the repo convention. No new metrics or tracing.

## Risks and Mitigations

- **Refactor changes behaviour** (the whole point is it must not). → Behaviour-preserving:
  CLI tests stay green except the intentional `ValueError`→shared-type change; land task 2
  behind a green `tox`.
- **Coverage drop from removing `hasattr` tests.** → Confirm the concrete is already covered
  (or add a behavioural test) before deleting; task 5 verifies 100%.
- **Layer-boundary violation from the loader's placement.** → Keep it a pure `shared/` helper
  (or an application/infrastructure split); `domain/` stays free of I/O; `pytest-archon`
  guards it in task 6.
- **`print()`→logger under `-W error`/random-order.** → Use the module logger already used
  elsewhere; assert on log emission, not stdout; watch the stochastic-coverage RNG-leak gotcha
  when touching ML tests.
- **Untracked `_synesthesia` duplicate confuses local runs.** → Working-tree hygiene note
  only (Open Question); not a tracked change in this feature.
- **Single error type breaks a caller that caught the old type.** → `git grep` for
  `except ValueError` / `except FileNotFoundError` around these call sites before switching;
  update any catcher.

## Validation against the spec

Single shared loader + consistent error type → tasks 1–2; behaviour preserved → task 2 (green
CLI tests); `print()`→logger → task 3; `# type: ignore` removed → task 4; `hasattr` tests
replaced without coverage loss → task 5; green + boundaries + grep-clean → task 6.
