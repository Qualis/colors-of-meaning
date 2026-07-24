# Feature: Code hygiene cleanups

> **Delivery status (2026-07-21).** Delivered here: the **correctness** fix — all
> codebook-load sites now raise `FileNotFoundError` (was a `ValueError` mix in
> `compress.py`/`encode.py`) — and the `visualize.py` `# type: ignore` replaced by a real
> `-> Classifier` return annotation; `tox` green at 100%. The mechanical, test-coupled tail —
> consolidating the ~15 load sites behind one `load_codebook`, routing training `print()`
> through the logger, and replacing the `hasattr` existence tests — was split into feature
> **034** so it can be landed and verified on its own (the CLI tests patch both
> `{MODULE}._load_codebook` and `{MODULE}.FileColorCodebookRepository`, so consolidation is a
> ~20-file test refactor with no correctness stake now that the error type is uniform).

## Overview

Remove the small set of code-level smells an evaluator would notice on a close read, without
changing behaviour. The audit found: the codebook/corpus loader is copy-pasted across ~11
CLI modules, and the duplicates **disagree on the error type** for the identical "artifact
not found" condition (most raise `FileNotFoundError`; `compress.py` and `encode.py` raise
`ValueError`); ML-training loops write progress with `print()` instead of the structured
logger used everywhere else; a cluster of interface tests assert only `hasattr(...)` (ABC
method names) rather than behaviour; and one `# type: ignore` at `visualize.py` hides a
missing return annotation. This feature consolidates the loader behind one shared helper with
a single, consistent error type, routes training progress through the logger, replaces the
low-value existence tests with behavioural ones, and fixes the type-ignore — all while
holding 100% coverage and every gate. Independent of features 030–032; touches `src/`, so it
is the coverage-critical member of the sequence.

## User Stories

- As a maintainer, I want one loader for codebook/corpus artifacts with one error type, so a
  missing artifact behaves identically across every CLI and there is one place to change.
- As an operator, I want training progress in the structured log (with correlation-id), so
  runs are observable the same way as the rest of the system rather than via bare stdout.
- As an evaluator reading the tests, I want them to assert behaviour, not merely that an
  abstract method name exists, so the suite demonstrably protects the science.
- As a maintainer, I want no `# type: ignore` masking a missing annotation, so mypy strict
  coverage is real.

## Acceptance Criteria

- [ ] Given any CLI that loads a codebook (or corpus) artifact, when the artifact is missing,
  then it raises **one** consistent error type via a single shared loader; the previous
  `FileNotFoundError`-vs-`ValueError` divergence (`compress.py`, `encode.py`) is gone, and no
  CLI carries its own copy of the load body.
- [ ] Given the shared loader, when the artifact exists, then every CLI that used a private
  `_load_codebook` returns the same object it did before (behaviour-preserving refactor).
- [ ] Given a training run, when it reports progress, then it uses the structured logger
  (with correlation-id) and no `print()` remains in `infrastructure/ml/*` training loops.
- [ ] Given the interface test suite, when inspected, then the `hasattr`-only existence tests
  are replaced by tests asserting observable behaviour (or removed where a behavioural test
  already covers the concrete), with no loss of coverage.
- [ ] Given `interface/cli/visualize.py`, when type-checked, then the `# type: ignore` on the
  classifier-factory helper is removed and replaced by a real return annotation; `mypy` is
  clean.
- [ ] Given the whole change, when `tox` runs, then it is green at 100% coverage with all 8
  gates, and no domain/application/infrastructure/interface boundary is violated
  (`pytest-archon` still passes).

## Hexagonal Layer Impact

### Application Layer (`src/colors_of_meaning/application/`)

No use-case behaviour change. If the shared artifact-loader is expressed as application-level
orchestration it lives here; otherwise it is a shared helper (see below). Decision recorded
in the plan.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

`ml/pytorch_color_mapper.py`, `ml/structured_pytorch_color_mapper.py`,
`ml/supervised_pytorch_color_mapper.py`: replace training-loop `print()` with the module
logger (correlation-id, matching the existing logging convention).

### Interface Layer (`src/colors_of_meaning/interface/`)

The ~11 CLI modules (`ablate`, `authorship`, `compass`, `compress`, `decode_image`, `encode`,
`eval_suite`, `generate`, `grounding`, `query`, …) drop their private `_load_codebook`
(and corpus-load duplicates) in favour of the shared loader; `visualize.py` gains a real
return annotation in place of the `# type: ignore`.

### Shared Layer (`src/colors_of_meaning/shared/`)

Likely home of the consolidated artifact loader (a pure filesystem/deserialize helper with
one documented error type), unless it is better placed as an application service — the plan
picks one and justifies it against the layer rules.

### Tests

Interface `hasattr` existence tests (concentrated in
`tests/.../domain/service/test_figure_renderer.py` and two others) are replaced by
behavioural assertions or removed where redundant; a test for the shared loader (found and
missing paths, the single error type) is added.

## API Contracts

None.

## CLI Impact

No user-facing CLI argument or output change. Internally, every CLI resolves artifacts
through the shared loader; a missing artifact now fails identically everywhere.

## Dependency Injection

The shared loader is injected/called at each CLI composition root exactly where the private
`_load_codebook` was called; no Lagom container change. (CLIs already hand-wire their
composition roots — this feature does not change that pattern, only de-duplicates the load
step.)

## Observability

Training progress moves from `print()` to structured logging with correlation-id — a net
observability improvement using the existing convention (no new metrics/tracing infra).

## Open Questions

- **Loader placement: `shared/` vs an application service.** A pure "load pickle or raise"
  helper fits `shared/`; if it should carry domain meaning (e.g. return a `ColorCodebook`
  and validate it) an application/infrastructure split may be cleaner. The plan chooses one
  and justifies it against the layer-boundary rules; both keep `domain/` pure.
- **Single error type choice.** `FileNotFoundError` (stdlib, precise for a missing file) vs a
  domain-specific `ArtifactNotFoundError`. Default: `FileNotFoundError` for a missing path
  (it is literally that), reserving a domain error only if richer context is needed.
- **Untracked `_synesthesia` working-tree duplicate.** The `src/colors_of_meaning_synesthesia`
  + `tests/...` copy is untracked (0 tracked files) but physically present, so local
  flake8/black/xenon/pytest can pick it up. This is a working-tree hygiene note, not a repo
  change; resolving it (delete locally, or add to tool excludes) is optional and called out
  so it is not mistaken for tracked drift.
- **Scope of the `hasattr` test replacement.** Some existence tests document the ABC surface;
  where a behavioural test already covers the concrete implementation they are removed, else
  upgraded — the plan lists which, to keep coverage at 100%.
