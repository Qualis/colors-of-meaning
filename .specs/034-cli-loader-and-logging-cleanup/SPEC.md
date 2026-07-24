# Feature: CLI codebook-loader consolidation, structured training logs, behavioural service tests

## Overview

The mechanical DRY / observability tail of feature 033, split out because it is a
test-coupled refactor better landed and verified on its own. Feature 033 fixed the
**correctness** defect (all codebook-load sites now raise `FileNotFoundError`, not a
`ValueError`/`FileNotFoundError` mix) and closed a type hole; this feature removes the
remaining duplication and polish:

1. **One shared codebook loader.** Nine CLI modules define a near-identical
   `_load_codebook`, and ~6 more inline the same load-or-raise. Consolidate them into a
   single `interface/cli/codebook_loading.py::load_codebook`. The churn is in the **tests**:
   they variously `patch("{MODULE}._load_codebook")` and `patch("{MODULE}.FileColorCodebookRepository")`,
   so each call site's tests must be re-pointed and the now-redundant per-module loader tests
   removed in favour of one shared loader test.
2. **Structured training logs.** The three PyTorch mappers write epoch progress with
   `print()`. Route them through the standard-library logger (matching the repo's
   "stdlib logger + correlation-id" convention), converting the three `capsys` assertions to
   `caplog`.
3. **Behavioural service tests.** Replace the eight `assert hasattr(FigureRenderer, ...)`
   existence tests with behavioural assertions, or remove those already covered by a
   concrete renderer's behavioural test — no loss of coverage.
4. **Working-tree hygiene.** The untracked `src/colors_of_meaning_synesthesia` +
   `tests/...` duplicate (0 tracked files) is physically present and can be picked up by
   local flake8/black/xenon runs; note/remove it locally (it is not a repository change).

All behaviour-preserving; `tox` stays green at 100% coverage throughout.

## User Stories

- As a maintainer, I want one codebook loader with one error contract, so a missing
  artifact behaves identically everywhere and there is a single place to change.
- As an operator, I want training progress in the structured log (not bare stdout), so runs
  are observable the same way as the rest of the system.
- As an evaluator reading the tests, I want them to assert behaviour rather than that an
  abstract method name exists.

## Acceptance Criteria

- [ ] Given any CLI that loads a codebook, when the artifact is missing, then it raises via
  the single `load_codebook` helper; no CLI retains a private `_load_codebook`, and
  `git grep 'def _load_codebook'` returns nothing.
- [ ] Given the shared loader, when the artifact exists, then every migrated CLI returns the
  same object it did before (behaviour-preserving; existing CLI flow tests stay green after
  their patch targets are re-pointed to `{MODULE}.load_codebook`).
- [ ] Given a training run, when it reports progress, then it uses the module logger and no
  `print()` remains in `infrastructure/ml/*`; the three epoch tests assert via `caplog`.
- [ ] Given the interface test suite, when inspected, then the `hasattr`-only existence tests
  are replaced by behavioural assertions (or removed where redundant), with coverage still
  100%.
- [ ] Given the whole change, when `tox` runs, then it is green at 100% coverage with all 8
  gates and `pytest-archon` passing.

## Hexagonal Layer Impact

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/codebook_loading.py` (`load_codebook(codebook_name) -> ColorCodebook`,
raising `FileNotFoundError`). The ~15 CLI modules drop their private `_load_codebook` /
inline load in favour of it.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

`ml/pytorch_color_mapper.py`, `ml/structured_pytorch_color_mapper.py`,
`ml/supervised_pytorch_color_mapper.py`: module logger replaces training-loop `print()`.

### Tests

One shared-loader test replaces the per-module `TestLoadCodebook` classes; CLI flow tests
re-point their patches to `{MODULE}.load_codebook`; the three ML epoch tests move
`capsys` → `caplog`; the `test_figure_renderer.py` existence tests become behavioural.

## API Contracts

None.

## CLI Impact

No user-facing change; internally every CLI resolves codebooks through one loader.

## Dependency Injection

The shared loader is called at each CLI composition root where `_load_codebook` was; no
Lagom change.

## Observability

Training progress moves from `print()` to structured logging (module logger), a net
improvement using the existing convention; no new metrics/tracing.

## Open Questions

- **Loader placement.** `interface/cli/codebook_loading.py` (a CLI helper — all callers are
  CLIs) vs a shared/application service. Default: the CLI helper, keeping the interface-only
  concern in the interface layer and `domain/` pure.
- **Inline sites.** The ~6 inline load-or-raise sites already raise `FileNotFoundError`
  (consistent); migrating them is pure DRY and can trail the 9 `def` sites if a smaller first
  landing is preferred.
- **`_synesthesia` duplicate.** Local working-tree cleanup only; if it should instead be
  added to tool excludes (flake8/black/xenon) that is a one-line config change, not a code
  change.
