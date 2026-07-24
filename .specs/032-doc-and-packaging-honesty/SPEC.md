# Feature: Doc & packaging honesty

## Overview

Make every claim a reader encounters match what is actually implemented, ship the package's
real description, and defend that honesty with a test. `README.MD` is already exemplary (it
carries a ✅/◑/➕/⬜ scorecard and candid hedges), but three surfaces undercut it: `setup.cfg`
ships the generic PyScaffold stub `README.rst` as the PyPI `long_description` (not the real
doc); `CLAUDE.md` §§ "Observability / Security / Enhancing System Quality" and `README.rst`
advertise Redis, Vault, Pub/Sub, distributed tracing, metrics collection, circuit breakers,
and Terraform — **none implemented**; and the `docs/` Sphinx build is vestigial (no
`conf.py`, empty `requirements.txt`) so `tox -e docs` would fail. For a repository whose
purpose is to demonstrate production capability to a skeptical evaluator, a claim that fails
a `grep` is the largest avoidable dent in credibility. This feature reconciles the docs to
reality (relocating genuine roadmap items under an explicit "not yet implemented" heading,
not silently deleting), ships `README.MD` as the description, removes the vestigial docs
scaffolding and PyScaffold leftovers, extends the existing grep-guard test into an enforced
honesty invariant, and aligns two inconsistent tool configs. It builds nothing that is
currently over-claimed — that would be gold-plating for a demo; the honest, higher-signal
move is to make the real strengths legible.

## User Stories

- As a prospective client, I want every capability claim in the README and contributor docs
  to survive a `grep`, so the project reads as trustworthy rather than aspirational.
- As a contributor, I want `pip install` / PyPI to show the real project description, and
  `tox -e docs` to either work or not exist, so first contact is clean.
- As a maintainer, I want an automated guard that fails if a doc re-introduces an
  un-implemented-infrastructure claim as a present capability, so honesty is enforced, not
  promised.
- As a reader of `CLAUDE.md`, I want its "system qualities" to reflect what the code does,
  with genuine future work clearly labelled as such.

## Acceptance Criteria

- [ ] Given the built package, when inspected, then `setup.cfg` sets `long_description =
  file: README.MD` with `long_description_content_type = text/markdown`, a real one-line
  `description`, and the real project `url`; `python -m twine check dist/*` passes.
- [ ] Given `README.rst`, when read, then it is either removed or reduced to a short pointer
  to `README.MD`, and `git grep README.rst` shows no remaining reference.
- [ ] Given `CLAUDE.md` and `README.rst`, when searched, then Redis / Vault / Pub-Sub /
  circuit breakers / distributed tracing / metrics collection / Terraform no longer appear as
  **present** capabilities; each is removed or relocated under an explicit "Target
  architecture (not yet implemented)" heading. The genuinely-present capabilities (Argon2id
  auth, correlation-id structured logging, Packer/Ansible image builds, real health checks,
  the 8-gate/100%-coverage discipline) are stated plainly and retained.
- [ ] Given `tests/.../shared/test_docs_claims_consistency.py`, when the suite runs, then a
  test fails if any un-implemented-infra keyword appears as a claimed present capability
  outside a roadmap-marked section; the existing README/design guards still pass.
- [ ] Given the vestigial docs build, when resolved, then either (recommended) `.readthedocs.yml`,
  the `[testenv:{docs,doctests,linkcheck}]` envs, the empty `docs/requirements.txt`, and the
  `docs/conf.py` flake8 exclude are removed, or a minimal working `docs/conf.py` +
  `requirements.txt` make `tox -e docs` build — no half-configured build remains.
- [ ] Given `src/colors_of_meaning/__init__.py`, when read, then the `# TODO` comment and the
  dead `sys.version_info >= (3, 8)` branch are gone (Python ≥3.11 is required; the comment
  violated the no-comments rule) and `import colors_of_meaning; __version__` still resolves;
  `setup.py`'s bare `except:` is narrowed or removed.
- [ ] Given the lint/format config, when compared, then flake8 `max_line_length` matches
  black's `line-length` (120), and `mypy` type-checks `tests/` as well as `src` (or the
  scope is a deliberate, documented choice) — no silent config divergence.

## Hexagonal Layer Impact

Mostly documentation, packaging, and tooling config. The only `src/` touch is the
coverage-exempt `__init__.py` cleanup. No domain/application/infrastructure/interface
behaviour changes; the `pytest-archon` suite is unaffected.

### Interface / CLI / API

No controller, DTO, or CLI changes.

### Shared Layer

No `shared/` source change.

### Tests (`tests/colors_of_meaning/shared/`)

`test_docs_claims_consistency.py` is extended with the honesty-invariant guards, reusing its
existing `_read_readme` / `_read_design_doc` / `assert_that` pattern; a `_read_claude_md`
helper is added.

### Repository / packaging files

`setup.cfg`, `README.rst`, `README.MD`, `.claude/CLAUDE.md`, `docs/design.md` (only if a claim
drifts), `.readthedocs.yml`, `tox.ini` (docs envs), `docs/requirements.txt`,
`src/colors_of_meaning/__init__.py`, `setup.py`, `pyproject.toml`/`setup.cfg` (flake8 + mypy
config alignment).

## API Contracts

None.

## CLI Impact

If the vestigial docs envs are removed (recommended), `tox -e docs` / `doctests` / `linkcheck`
cease to exist; no user-facing CLI command in `interface/cli/` changes.

## Dependency Injection

None.

## Observability

No observability code changes. The reconciliation makes the docs describe the observability
that **already exists** (correlation-id structured logging) and moves the absent
metrics/tracing claims under the roadmap heading.

## Open Questions

- **CLAUDE.md needs owner sign-off.** It is the owner's authored project-instructions file;
  the reconciliation is proposed here but the specific edits are applied only after explicit
  approval — isolated as its own plan task so the rest of the feature can proceed meanwhile.
- **Remove vs. wire the docs build.** Recommend removing the vestigial Sphinx/RTD scaffolding
  (there is no `conf.py`); a minimal real Sphinx setup is the alternative if published docs
  are wanted.
- **`Development Status` classifier.** `4 - Beta` today; advancing to `5 - Production/Stable`
  is a judgement call for the owner, folded in only if desired.
- **mypy on tests.** Enabling `mypy tests` may surface untyped test helpers; if the cleanup is
  larger than this feature warrants, documenting the `src`-only scope as deliberate is an
  acceptable resolution.
