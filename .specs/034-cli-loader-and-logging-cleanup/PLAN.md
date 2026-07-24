# Plan: CLI codebook-loader consolidation, structured training logs, behavioural service tests

## Implementation Strategy

Land as three independently-committable, behaviour-preserving chunks, each verified by
`tox` at 100% coverage before the next. The refactor's risk is entirely in the **test
coupling** (mock-patch targets), so each chunk is small and immediately re-verified.

1. **Shared loader** — create `interface/cli/codebook_loading.py` + one test; migrate the 9
   `_load_codebook` sites (then the ~6 inline sites); re-point CLI-flow patches to
   `{MODULE}.load_codebook`; delete the redundant per-module loader tests.
2. **Training logs** — module logger replaces `print()` in the 3 ML mappers; convert the 3
   epoch tests `capsys` → `caplog`.
3. **Behavioural tests** — replace/remove the 8 `hasattr` existence tests.

## Layer Changes

### Interface Layer

- New `interface/cli/codebook_loading.py`:
  ```
  def load_codebook(codebook_name: str) -> ColorCodebook:
      codebook = FileColorCodebookRepository().load(codebook_name)
      if codebook is None:
          raise FileNotFoundError(f"Codebook not found: {codebook_name}")
      return codebook
  ```
- Per migrated CLI: remove `def _load_codebook`, remove the now-unused
  `FileColorCodebookRepository` import, add `from ...codebook_loading import load_codebook`,
  rename call sites `_load_codebook(` → `load_codebook(`.
- Per CLI test: remove the direct `TestLoadCodebook`; re-point flow patches
  `patch("{MODULE}._load_codebook")` → `patch("{MODULE}.load_codebook")`; drop tests that
  patched `{MODULE}.FileColorCodebookRepository` purely to exercise the loader (covered once by
  the shared-loader test).

### Infrastructure Layer

- Each ML mapper: `import logging` + `logger = logging.getLogger(__name__)`; replace
  `print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")` with
  `logger.info("Epoch [%d/%d], Loss: %.4f", epoch + 1, epochs, avg_loss)`.
- Each epoch test: `capsys` → `caplog`, wrapping the train call in `with
  caplog.at_level(logging.INFO):` and asserting `"Epoch [10/10]" in caplog.text`; rename
  `test_should_print_loss_...` → `test_should_log_loss_...`.

### Tests

- `test_figure_renderer.py`: replace the 8 `assert hasattr(...)` with behavioural checks
  against a concrete renderer, or remove those already covered by
  `matplotlib_figure_renderer` behavioural tests — verifying coverage stays 100% before each
  deletion.

## Task List

1. [ ] interface: `codebook_loading.py` + shared test (found / missing → `FileNotFoundError`).
2. [ ] interface: migrate the 9 `_load_codebook` `def` sites + re-point their tests; `tox` green.
3. [ ] interface: migrate the ~6 inline load-or-raise sites; `tox` green.
4. [ ] infrastructure: `print()` → module logger in the 3 ML mappers; convert 3 epoch tests to `caplog`; `tox` green.
5. [ ] tests: behavioural `FigureRenderer` tests; `tox` green at 100%.
6. [ ] hygiene: note/remove the untracked `_synesthesia` working-tree duplicate (local; or add to tool excludes).
7. [ ] verify: full `tox`; `git grep 'def _load_codebook'` and `git grep 'print(' src/.../infrastructure/ml` clean; `pytest-archon` green.

## Testing Strategy

- **Shared loader:** one test returns the object for an existing artifact; one asserts
  `FileNotFoundError` for a missing one.
- **Migration:** existing CLI flow tests stay green after re-pointing patches — the
  behaviour-preservation proof; per-module loader tests are removed only once the shared test
  covers their branches.
- **Logger:** assert on `caplog` (log emission), not stdout; keep the line covered.
- **Coverage:** confirm the concrete is already covered before deleting any existence test;
  keep new helpers grade-A for xenon (scans tests).
- Verify with `tox`, never `pytest` directly.

## Risks and Mitigations

- **Mock-path drift breaks CLI flow tests.** → Migrate one module at a time, `tox` after each;
  re-point every `patch("{MODULE}._load_codebook")` and remove repo-patching loader tests.
- **Coverage drop from removed tests.** → The shared-loader test covers both branches; confirm
  100% after each chunk.
- **`print()`→logger under `-W error` / random-order.** → Module logger + `caplog`; watch the
  stochastic-coverage RNG-leak gotcha when touching ML tests.
- **Layer boundary.** → Loader is an interface-layer CLI helper; `domain/` stays pure;
  `pytest-archon` guards it.

## Validation against the spec

Single loader, no `def _load_codebook` remaining → tasks 1–3; behaviour preserved → tasks 2–3
(green flow tests); `print()`→logger + `caplog` → task 4; `hasattr` → behavioural → task 5;
`_synesthesia` hygiene → task 6; green + boundaries + grep-clean → task 7.
