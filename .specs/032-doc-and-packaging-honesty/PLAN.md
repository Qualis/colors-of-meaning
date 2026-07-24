# Plan: Doc & packaging honesty

## Implementation Strategy

Reconcile the three surfaces that undercut the already-honest `README.MD`: the shipped
description, the over-claiming prose in `CLAUDE.md`/`README.rst`, and the vestigial docs
build — then lock honesty in with an extended grep-guard test, and align two divergent tool
configs. No capability is *built*; over-claimed items are removed or relocated under an
explicit roadmap heading. The one owner-authored file (`CLAUDE.md`) is edited only after
sign-off, isolated as its own task. Independent of features 030/031; can land any time.

Prefer editing existing files. Only new file: none required (the guard test already exists;
docs scaffolding is removed, not added, under the recommended option).

## Layer Changes

### Tests (`tests/colors_of_meaning/shared/`)

- Extend `test_docs_claims_consistency.py`: add a `_read_claude_md` helper and
  one-assertion-per-test guards — e.g.
  `test_should_not_claim_redis_as_a_present_capability_when_reading_claude_md`,
  `..._vault...`, `..._distributed_tracing...`, `..._terraform...` — each asserting the
  keyword does not appear as a present capability (or appears only under the roadmap
  heading). Keep the existing README-count and design-doc Wasserstein guards. Keep each test
  grade-A for xenon (the whole-repo complexity gate scans tests — repo gotcha).

### Repository / packaging files

- **`setup.cfg`.** `description` → real one-liner; `url` → real repo; `long_description =
  file: README.MD`; `long_description_content_type = text/markdown`.
- **`README.rst`.** Reduce to a one-paragraph pointer to `README.MD`, or delete; confirm
  `git grep README.rst` is clean afterward (only `setup.cfg` referenced it).
- **`CLAUDE.md` (task-gated on sign-off).** Scope §§ "Observability Requirements", "Security
  Requirements", "Enhancing System Quality", and the README-mirrored "System Qualities": move
  Redis / Vault / Pub-Sub / circuit breakers / distributed tracing / metrics / Terraform under
  an explicit "Target architecture (not yet implemented)" heading; keep and state the real
  capabilities plainly.
- **Vestigial docs (recommended: remove).** Delete `.readthedocs.yml`, the
  `[testenv:{docs,doctests,linkcheck}]` envs in `tox.ini`, the empty `docs/requirements.txt`,
  and the `docs/conf.py` entry from the flake8 `exclude` in `setup.cfg`. (`docs/design.md`
  stays — it is real and guarded.)
- **`src/colors_of_meaning/__init__.py`.** Remove the `# TODO` comment; collapse the
  `sys.version_info >= (3, 8)` guard to the direct `from importlib.metadata import
  PackageNotFoundError, version` (≥3.11 guaranteed). Behaviour unchanged; file is
  coverage-exempt (`.coveragerc` omits `*/__init__.py`).
- **`setup.py`.** Narrow the bare `except:`+`# noqa` to a specific exception (or drop the
  try/except); setup.py is outside the coverage source scope.
- **Config alignment.** Set flake8 `max_line_length = 120` (match black) in `setup.cfg`; add
  `mypy tests` to the gate (or record the `src`-only scope as deliberate) — whichever keeps
  the gate green with least churn, documented in the PR.

## Dependency Injection

None.

## Task List

1. [ ] packaging: `setup.cfg` description/url/`long_description`→`README.MD` (markdown);
   slim or remove `README.rst`; build and `twine check dist/*`.
2. [ ] docs build: remove the vestigial Sphinx/RTD/docs-tox scaffolding (or wire a minimal
   `conf.py`); confirm no dangling reference (flake8 exclude, `.readthedocs.yml`).
3. [ ] cleanup: `__init__.py` comment + dead branch; `setup.py` bare except; confirm
   `import colors_of_meaning; colors_of_meaning.__version__` resolves.
4. [ ] config: align flake8 `max_line_length` to 120; add `mypy tests` (or document
   `src`-only); `tox` green.
5. [ ] tests: extend `test_docs_claims_consistency.py` with the honesty-invariant guards
   (one assertion each; `assert_that`); existing guards stay green.
6. [ ] CLAUDE.md (after owner sign-off): reconcile the aspirational sections under a roadmap
   heading; re-run the guard so `CLAUDE.md` passes it.
7. [ ] verify: full `tox` green at 100% coverage; `git grep` shows no un-relocated
   over-claim and no `README.rst` reference.

## Testing Strategy

- **Honesty guards:** one `assert_that(...).does_not_contain(...)` (or roadmap-scoped) per
  invariant, `test_should_..._when_...` named, base-entity style (`assert_that`) matching the
  file. Self-covering (they exercise their own grep logic); no `src/` coverage delta.
- **Packaging:** `twine check` in the existing publish env validates markdown
  `long_description` rendering; no bespoke unit test.
- **`__init__.py`:** coverage-exempt; the existing version-smoke import still passes.
- **Config alignment:** validated by `tox` staying green after the flake8/mypy changes.
- Verify with `tox`, never `pytest` directly.

## Observability Plan

None. Docs are edited to describe the existing correlation-id logging accurately; no code
added.

## Risks and Mitigations

- **Editing the owner's `CLAUDE.md`.** → Isolated as task 6, gated on explicit sign-off; the
  rest of the feature proceeds without it.
- **`mypy tests` surfaces many untyped-helper errors.** → If larger than warranted, keep
  `src`-only and document the deliberate scope (Open Question) rather than block the feature.
- **Markdown `long_description` fails `twine check`.** → Run `twine check` in task 1; fix
  content-type / rendering before relying on it.
- **Removing docs envs breaks a referenced command.** → `grep` for `tox -e docs` references
  first; `.readthedocs.yml` and the docs envs are self-contained.
- **Honesty guard is brittle / false-positives on legitimate prose** (e.g. "metrics" in
  "retrieval metrics"). → Scope the guard to the specific over-claim phrases and the roadmap
  heading, not bare substrings; the existing file already distinguishes README vs design doc.
- **xenon flags a new test helper.** → Keep guards trivial one-liners; extract any filter to
  a helper (repo gotcha) to stay grade A.

## Validation against the spec

`long_description`→README.MD + twine → task 1; README.rst slimmed → task 1; docs build
resolved → task 2; PyScaffold cleanup → task 3; config alignment → task 4; honesty guard →
task 5; CLAUDE.md reconciliation → task 6; grep-clean + green → task 7.
