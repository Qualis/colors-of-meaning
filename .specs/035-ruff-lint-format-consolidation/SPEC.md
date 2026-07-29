# Feature: Consolidate linting & formatting into Ruff

## Overview

Replace the five-tool lint/format stack — `flake8` + `flake8-bugbear` + `pep8-naming` +
`flake8-typing-imports` + `black` (with its implicit `isort`-style concerns) — with a single
tool, **Ruff**, running as both linter (`ruff check`) and formatter (`ruff format`). Ruff is
a Rust-based drop-in that natively reimplements the plugins this repo uses (bugbear `B`,
pep8-naming `N`, pycodestyle `E`/`W`, pyflakes `F`) plus a Black-compatible formatter, at
10–100× the speed. Fewer tools means fewer independent version-drift points
(the class of failure recorded in the project's tox-env-drift history) and a cleaner, modern
toolchain signal for a repository whose purpose is to demonstrate production capability.

**Scope boundary — Ruff replaces only the lint/format block.** The security and quality
scanners stay exactly as they are: `bandit -r src`, `semgrep`, `pip-audit`, `radon`,
`xenon`, and `mypy` are unchanged. Ruff's own security (`S`) rules are a *subset* of
`bandit`, so keeping full `bandit` avoids any coverage regression; this feature does not
touch it. This keeps the change small, reversible, and free of security-parity risk.

**Honest parity gaps.** Ruff's letter selectors are not a strict superset of the retired
stack. Measured by running flake8 7.3.0 + bugbear 25.11.29 + pep8-naming 0.15.1 +
flake8-typing-imports 1.17.0 against Ruff 0.16.0 over per-rule probe files, these checks have
no equivalent under the shipped configuration:

- `flake8-typing-imports` `TYP001` (version-gated `typing` imports against `python_requires`).
  Obviated in practice: the project requires Python ≥3.11, so every `typing` symbol in use is
  available. Dropped deliberately rather than pretending Ruff covers it.
- bugbear `B036` (`except BaseException` without re-raise), `B040` (exception whose `add_note`
  is never raised), `B042` (exception subclass not forwarding to `super().__init__()`). Ruff's
  `BLE001` is **not** a `B036` substitute — it also flags `except Exception`, which would force
  the `src/` change this feature excludes.
- pep8-naming `N808` (TypeVar CapWords). No `TypeVar` exists in the codebase today.
- pycodestyle comment-shape rules `E262`, `E266`, and the `#!` form of `E265`. Every other
  E1xx/E2xx/E3xx rule Ruff leaves to preview is normalised by `ruff format`, so
  `ruff format --check` still fails on it; comment shape is the residual.

Two lost checks are recovered by rule codes outside the letter selectors, both selected:
bugbear `B037` (`return`/`yield` in `__init__`) via `PLE0100`/`PLE0101`, and pyflakes `F824`
(unassigned `global`) via `PLW0602`. Bugbear `B001` and `B041` are already covered by `E722`
and `F601`. No finding present in the tree today is lost; the gaps are prospective.

## User Stories

- As a maintainer, I want one fast tool for linting and formatting instead of five, so there
  are fewer moving parts to pin, upgrade, and keep from drifting.
- As a contributor, I want `ruff check` / `ruff format` to give the same style guarantees the
  flake8+black stack did, so the migration is invisible to how I write code.
- As an evaluator, I want the toolchain to reflect current best practice (Ruff), reinforcing
  that the project is maintained to a modern standard.

## Acceptance Criteria

- [ ] Given the gate, when `tox` runs, then `ruff check .` and `ruff format --check .` replace
  the `flake8` and `black . --check` commands, and all other gate steps (bandit, semgrep,
  pip-audit, radon, xenon, mypy, pytest) are unchanged; `tox` is green at 100% coverage.
- [ ] Given the Ruff configuration in `pyproject.toml [tool.ruff]`, when linting, then it
  selects the equivalents of the retired stack (`E`, `W`, `F`, `B`, `N`), preserves the
  intentional ignores (`E203`), and replaces the blanket `B008` ignore with
  `lint.flake8-bugbear.extend-immutable-calls` for `fastapi.Depends`/`fastapi.Query`.
- [ ] Given the formatter, when `ruff format` runs, then it uses `line-length = 120` (Black
  parity) and leaves the codebase unchanged (the current Black-formatted tree is already
  Ruff-format stable, or the one-time reformat is committed and reviewed).
- [ ] Given the retired tools, when the config is inspected, then `[flake8]` is removed from
  `setup.cfg`, `[tool.black]` from `pyproject.toml`, and `flake8`, `flake8-bugbear`,
  `flake8-typing-imports`, `pep8-naming`, `black` are removed from the `testing` extra and
  replaced by a pinned `ruff` (captured in the lockfile).
- [ ] Given the docs, when read, then the `CONTRIBUTING.rst` tool list and the `CLAUDE.md`
  static-analysis table say `ruff` (lint + format) instead of `flake8`/`black`, and the
  `README.MD` "8 quality gates" wording still matches reality.
- [ ] Given the migration, when `ruff check` first runs, then any finding that differs from
  the flake8 baseline is reconciled in-code (extract-to-variable etc. — no comments, per the
  no-comments rule), not blanket-ignored.

## Hexagonal Layer Impact

Tooling/config only; **no `src/` layer changes** and no behaviour change, so the
`pytest-archon` suite and coverage are unaffected (beyond any one-time formatting normalisation,
which touches whitespace only). Files: `pyproject.toml` (add `[tool.ruff]`, remove
`[tool.black]`), `setup.cfg` (remove `[flake8]`; swap `testing` extra deps), `tox.ini`
(swap the two commands), `requirements.lock` (drop the retired tools, add pinned `ruff`),
`CONTRIBUTING.rst`, `.claude/CLAUDE.md`, and possibly a one-time `ruff format` normalisation
diff across `src`/`tests`.

## API Contracts

None.

## CLI Impact

No project CLI changes. Developer commands change: `flake8`/`black` → `ruff check` /
`ruff format`.

## Dependency Injection

None.

## Observability

None.

## Open Questions

- **Lint line length.** flake8 currently allows 140 (headroom for the few unsplittable string
  literals Black leaves >120 — see spec 032). Ruff option: keep lint `line-length = 140` while
  the formatter targets 120, or set both to 120 with `per-file-ignores`/`noqa` on the handful
  of long-string lines. Default: mirror today's behaviour (lint 140 / format 120) so the
  migration is behaviour-preserving; tightening is a follow-up.
- **Extra rule families.** Ruff cheaply offers `UP` (pyupgrade), `SIM` (simplify), `I`
  (isort), `C4`, etc. Default: match the current rule set only (`E,W,F,B,N`) to keep the
  migration a pure swap; adopting `I`/`UP`/`SIM` is a deliberate follow-up so any churn is
  reviewed on its own.
- **`W503`/`E203`.** Both are Black/Ruff-formatter concerns; the Ruff formatter obviates
  `W503`, and `E203` stays ignored. Confirm no residual pycodestyle conflict during migration.
- **Sequencing.** Land 035 before 036 (uv): consolidating lint/format config into
  `pyproject.toml` first makes the 036 `setup.cfg → [project]` migration cleaner.
