# Plan: Consolidate linting & formatting into Ruff

## Implementation Strategy

A pure, behaviour-preserving toolchain swap: Ruff replaces `flake8`+plugins+`black`; every
other gate stays. Config moves to `pyproject.toml [tool.ruff]`. The only risk is Ruff
surfacing findings that differ from the flake8 baseline, so the migration runs Ruff, reconciles
each diff in code (no blanket ignores, no comments), and verifies with the full `tox`. Land it
before spec 036 so lint/format config already lives in `pyproject.toml` when the metadata
migration happens. Config/tooling only — no `src` logic changes.

## Layer Changes

No domain/application/infrastructure/interface/shared source changes. Tooling config only:

- **`pyproject.toml`.** Add:
  ```
  [tool.ruff]
  line-length = 120
  extend-exclude = [".tox", "build", "dist", ".eggs", ".claude", ".vagrant", "*.md", "*.MD"]
  force-exclude = true
  [tool.ruff.lint]
  select = ["E", "W", "F", "B", "N", "PLE0100", "PLE0101", "PLW0602"]
  ignore = ["E203"]
  [tool.ruff.lint.pycodestyle]
  max-line-length = 140
  [tool.ruff.lint.flake8-bugbear]
  extend-immutable-calls = ["fastapi.Depends", "fastapi.Query"]
  ```
  Remove `[tool.black]`. Ruff has a single `line-length` (the formatter's, 120 for Black
  parity); the lint side keeps flake8's 140 through `lint.pycodestyle.max-line-length` —
  there is no `[tool.ruff.format] line-length`. Exclusions previously in `[flake8]`/black
  become `extend-exclude`; markdown is excluded because Ruff 0.16's formatter rewrites python
  code fences inside `.md`, which Black never did.
- **`setup.cfg`.** Remove the `[flake8]` section. In `[options.extras_require] testing`, drop
  `flake8`, `flake8-bugbear`, `flake8-typing-imports`, `pep8-naming`, `black`; add `ruff`.
- **`tox.ini` `[testenv] commands`.** Replace `flake8` with `ruff check .`, and
  `black . --check` with `ruff format --check .`. The `[testenv:format]` env runs
  `ruff format .` (and `ruff check --fix .`) instead of `black .`.
- **`requirements.lock`.** Regenerate so the retired tools drop out and a pinned `ruff`
  appears (freeze from a green env, as in spec 030; spec 036 will change how the lock is
  produced but not this feature).
- **Docs.** `CONTRIBUTING.rst` tool list and `CLAUDE.md` static-analysis table: `flake8`+`black`
  → `ruff` (lint + format). Confirm `README.MD`'s gate wording still holds.

## Dependency Injection

None.

## Task List

1. [ ] config: add `[tool.ruff]` to `pyproject.toml`; remove `[tool.black]` and `setup.cfg`
   `[flake8]`; swap the `testing` extra (`ruff` in, flake8/black stack out).
2. [ ] migrate: run `ruff check .` and `ruff format --check .`; reconcile every diff vs the
   flake8/black baseline in code (extract-to-variable, rename — no comments, no blanket
   ignores); commit any one-time `ruff format` normalisation separately for reviewability.
3. [ ] gate: swap the `tox.ini` `[testenv]` and `[testenv:format]` commands to Ruff.
4. [ ] lock: regenerate `requirements.lock` (pinned `ruff`, retired tools gone).
5. [ ] docs: update `CONTRIBUTING.rst` + `CLAUDE.md` (and verify `README.MD`) tool references.
6. [ ] verify: full `tox` green at 100% coverage; `ruff check`/`ruff format --check` clean;
   confirm no gate step other than lint/format changed.

## Testing Strategy

No new `src` code, so no new unit tests and no coverage delta (a one-time format normalisation
changes whitespace only). Validation is the gate itself:

- **Parity:** `ruff check .` clean after reconciliation; `ruff format --check .` reports no
  changes on the committed tree.
- **Gate integrity:** `tox` green; bandit/semgrep/pip-audit/radon/xenon/mypy/pytest outputs
  unchanged from before the swap.
- **Docs guard:** the spec-032 `test_docs_claims_consistency.py` still passes; update it only
  if a tool name it asserts changed.
- Verify with `tox`, never the tools directly, for the final check.

## Observability Plan

None.

## Risks and Mitigations

- **Ruff surfaces new/lost findings vs flake8.** → Task 2 reconciles each in code; keep the
  rule set minimal (`E,W,F,B,N`) so the diff is small; no blanket ignores.
- **Formatter reflows the tree** (Ruff format ≈ Black but not byte-identical). → Run
  `ruff format` once, commit the normalisation as its own reviewable diff; thereafter
  `--check` is stable.
- **Dropping `flake8-typing-imports` loses a check.** → Documented as obviated by
  `python_requires>=3.11` + Ruff `UP`; acceptable and recorded in the SPEC.
- **`B008` behaviour change.** → Replace the blanket ignore with `extend-immutable-calls`
  for `fastapi.Depends`/`fastapi.Query`, preserving intent more precisely than the old ignore.
- **Docs/lock drift.** → Tasks 4–5 update the lock and every tool reference; the docs guard
  and `tox` catch omissions.

## Validation against the spec

Ruff replaces flake8/black in the gate, others unchanged → tasks 1,3,6; rule/ignore parity +
`extend-immutable-calls` → task 1; formatter line-length 120 → tasks 1–2; retired config/deps
removed + `ruff` pinned → tasks 1,4; docs updated → task 5; findings reconciled not ignored →
task 2; green at 100% → task 6.
