# Feature: Reproducible authored-corpus authorship report & figures (`tox -e`)

## Overview

Every other committed report artifact regenerates from a committed command —
`eval_suite → reports/eval_results.md`, `rate_distortion → reports/rate_distortion.md`,
`interpretability → reports/interpretability.md`, `compass → reports/story_compass.md`.
The authored-corpus authorship artifacts are the exception: `reports/documents_authorship.md`
and its **136 committed figures** (`reports/figures/documents_color_tsne.png`,
`documents_color_signatures.png`, `documents_a4_gallery.png`, and 133
`reports/figures/a4/<author>__<work>.png` sheets) have **no committed generator**. A repo-wide
search finds nothing in `src/`, `bin/`, or a Makefile that writes them; the report is
hand-assembled and the figures were produced by an uncommitted one-off. This contradicts the
project's core "every committed artifact is regenerable from a committed command" ethos
(established by `006`, `019`, `022`, `026`).

This feature closes that gap by adding committed `tox -e` entrypoints that regenerate the
authorship figures and the machine-computed parts of the report. Most of the machinery already
exists and is reused:

- t-SNE and per-author signatures already have use-case methods
  (`application/use_case/visualize_documents_use_case.py`: `execute_projection`,
  `execute_corpus_signatures`) and renderer methods
  (`domain/service/figure_renderer.py`: `render_tsne_projection`, `render_corpus_signatures`).
- The per-book A4 signature sheet already has a primitive (`interface/cli/encode_image.py`,
  `--layout signature`).
- The document corpus is already a first-class `DatasetRepository`
  (`infrastructure/dataset/document_corpus_dataset_adapter.py`, `022`).

The **net-new** pieces are a small contact-sheet renderer for the A4 gallery montage, an
`AuthorshipReportUseCase` + CLI that writes the report's computed tables (mirroring how
`eval_suite` writes `eval_results.md`), a cached manifest for the expensive data-scaling sweep,
and a fix for a residual non-determinism in the documents path.

**Book generation is explicitly out of scope.** `reports/book/*` already has a committed
generator (`[testenv:generate]`, `interface/cli/generate.py`); it needs a live API key (never
CI) and the LLM is non-deterministic (no seed), so byte-for-byte reproduction is impossible by
construction. No new environment is warranted for it.

## Core Domain Concepts

- **Authorship report artifact**: `reports/documents_authorship.md` — narrative prose plus
  four computed tables (corpus authors/works, split sizes, held-out colour vs TF-IDF results,
  and the data-scaling sweep). Only the computed tables are machine-written; prose is a template.
- **A4 gallery contact sheet**: `documents_a4_gallery.png` — the 133 per-book A4 signature
  sheets tiled into one montage, ordered by author. Net-new renderer.
- **Data-scaling manifest**: a committed `reports/data/authorship_scaling.json` holding the
  8-seed × 3-cap sweep results, so the report regenerates without re-running 24 trainings; the
  sweep is refreshed only by an explicit opt-in.
- **Deterministic documents path**: the observed drift where `documents_rate_distortion`
  color-VQ ΔE changes run-to-run (while AG-News reproduces exactly) is a residual
  non-determinism in the documents source; reproducibility of the authorship artifacts requires
  removing it.

## User Stories

- As a contributor, I want `tox -e` commands that regenerate `documents_authorship.md`'s
  figures and computed tables, so the committed authorship artifacts obey the same
  regenerate-from-a-committed-command rule as every other report.
- As a researcher, I want the t-SNE, per-author signatures, 133 A4 sheets, and the A4 gallery
  to be produced deterministically to their exact committed filenames from the trained documents
  projector, so the gallery is an auditable artifact rather than a one-off.
- As a maintainer, I want the expensive 8-seed × 3-cap scaling sweep cached in a committed
  manifest and refreshed only on an explicit opt-in, so the default report regeneration stays
  fast and CI-adjacent while the sweep stays reproducible on demand.
- As a reviewer, I want the documents-sourced artifacts to reproduce bit-identically across
  runs, so a regenerated authorship figure is comparable to the committed one.
- As a maintainer, I want book generation left to the existing `generate` env, so no
  non-reproducible, key-dependent path is dressed up as a reproducible artifact.

## Acceptance Criteria

- [ ] Given a `documents/<author>/<work>.txt` corpus and a trained documents projector +
  codebook, when the figures command runs, then it writes `reports/figures/documents_color_tsne.png`
  and `reports/figures/documents_color_signatures.png` via the existing
  `VisualizeDocumentsUseCase` methods, at those exact paths.
- [ ] Given the same inputs, when the figures command runs, then it writes one
  `reports/figures/a4/<author>__<work>.png` signature sheet per work (reusing the
  `encode_image --layout signature` primitive) and tiles them into
  `reports/figures/documents_a4_gallery.png`, ordered by author.
- [ ] Given a fixed seed and inputs, when the figures command is run twice, then every output
  PNG is byte-identical between runs (the documents-path non-determinism observed in
  `documents_rate_distortion` is removed).
- [ ] Given the trained documents projector, when the authorship-report command runs, then it
  writes `reports/documents_authorship.md`'s computed tables — corpus authors/works, per-split
  paragraph counts, and the held-out colour vs TF-IDF results — from a real train/eval, mirroring
  how `eval_suite` writes `reports/eval_results.md`; narrative prose is preserved from a template.
- [ ] Given the committed `reports/data/authorship_scaling.json`, when the report command runs
  without the sweep opt-in, then the data-scaling table is rendered from the manifest and no
  training sweep is executed.
- [ ] Given the sweep opt-in flag, when the report command runs, then the 8-seed × 3-cap sweep
  is executed (via seed-varied configs, since `train` has no `--seed` flag today) and the
  manifest is rewritten; this mode is documented as non-CI and long-running.
- [ ] Given `./documents/` is git-ignored, when the test suite runs, then every test builds a
  synthetic `tmp_path` document tree (no real-file, no network, no live API), the gallery montage
  is exercised by at least one real (non-mocked) render, and `tox` stays green at 100% coverage.
- [ ] Given book generation, when this feature is complete, then no new environment is added for
  it and the README/roadmap state that `reports/book/*` is regenerated by `tox -e generate` and
  is inherently non-reproducible.
- [ ] Given the new `tox -e` environments, when they are added to `tox.ini`, then they are
  documented in the README CLI reference alongside the other report generators.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

`domain/service/figure_renderer.py` gains one abstract method, `render_a4_gallery(sheet_paths:
List[str], output_path: str, columns: int)`, for the contact-sheet montage — the only port
addition. No new business rule; the author→label mapping and paragraph rules already live in
`022`'s adapter and `shared/document_corpus.py`. An optional `AuthorshipReport` value object may
carry the computed tables if it keeps the use case pure.

### Application Layer (`src/colors_of_meaning/application/`)

New `application/use_case/authorship_report_use_case.py`: orchestrates train → held-out eval →
compute corpus/split/results tables → read the scaling manifest (or run the sweep on opt-in) →
emit the report; it depends only on the `DatasetRepository`, `Classifier`/evaluation, and a
report-writer port, never on I/O directly. `VisualizeDocumentsUseCase` gains an
`execute_a4_gallery(...)` method delegating to the new renderer method. No use case reaches the
filesystem or the network directly.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/visualization/matplotlib_figure_renderer.py` implements `render_a4_gallery`
  (tile the per-book PNGs into an author-ordered grid).
- The residual non-determinism in the documents path is identified and seeded — most likely the
  paragraph sampling / embedding order in `infrastructure/dataset/document_corpus_dataset_adapter.py`
  or its interaction with the `max_samples` budget — so repeated documents runs are bit-identical
  (AG-News already is). Seeding reuses `shared/determinism.py` and threaded `training.seed`.
- A manifest reader/writer for `reports/data/authorship_scaling.json` (a thin JSON adapter),
  and the report-writer that renders the `.md` from a template + computed tables.

### Interface Layer (`src/colors_of_meaning/interface/`)

- New `interface/cli/authorship.py` and `[testenv:authorship]`: writes
  `reports/documents_authorship.md` (default) plus a `--refresh-scaling` opt-in that runs the
  24-training sweep and rewrites the manifest. Takes `--config`, `--model-path`,
  `--codebook-name`, `--paragraphs-per-work`, and the split flags, matching sibling CLIs.
- Documents figures: extend `interface/cli/visualize.py` (or `visualize_corpus.py`) with
  `--source documents`, writing the `documents_*` figures + A4 sheets + gallery, and/or a
  dedicated `[testenv:visualize_documents]`. Reuses `VisualizeDocumentsUseCase` and the
  `DocumentCorpusDatasetAdapter`; no new figure logic beyond the gallery method.
- README CLI reference documents the new environment(s); the roadmap notes book stays on
  `generate`.

### Shared Layer (`src/colors_of_meaning/shared/`)

Reuses `shared/document_corpus.py` (paragraph/parse) and `shared/determinism.py` (seeding). A
small pure helper for author-ordered gallery tiling order may live here if it stays framework-free.

## API Contracts

None. No controller, route, or DTO is added or changed. The authorship artifacts are offline
research outputs.

## CLI Impact

New `tox -e authorship` (writes `documents_authorship.md`; `--refresh-scaling` opt-in for the
sweep) and a documents figures path (`tox -e visualize -- --source documents …` or a new
`tox -e visualize_documents`). No existing CLI flag changes. No new third-party dependency.

## Dependency Injection

The new use case receives its `DatasetRepository`, evaluation collaborators, `FigureRenderer`,
and manifest/report-writer ports by construction/Lagom, exactly as `eval`, `rate_distortion`, and
`visualize_corpus` wire their dependencies today.

## Observability

`correlation-id` structured logging at report/figure generation: authors and works discovered,
per-split paragraph counts, held-out accuracies, each figure path written, and whether the
scaling table came from the manifest or a fresh sweep. No new metrics/tracing infrastructure
(consistent with the repo's stdlib-logger convention).

## Open Questions

- **One env or two.** A single `tox -e authorship` that drives both report and figures, versus
  a `visualize_documents` (figures) + `authorship` (report) split. Default: split, so the fast
  deterministic figures are not coupled to the report/train step.
- **Scaling manifest scope.** Commit `reports/data/authorship_scaling.json` and read it by
  default (recommended), or always re-run the sweep? Default: commit the manifest; opt-in refresh.
  Should the manifest be validated against a schema/committed test?
- **Split the spec.** Figures generator is a clean MVP; the report-writer, the sweep manifest,
  and the seeding fix could each be a follow-up. Ship as phases within `028`, or separate specs?
- **Root cause of the documents non-determinism.** Confirm whether the drift is in paragraph
  sampling, embedding batch order, or the color-VQ codebook fit on the documents distribution,
  and seed exactly that step; add a regression test asserting a twice-run documents artifact is
  identical.
- **A4 gallery layout.** Fixed column count vs aspect-driven; author separation (grouping/labels)
  in the montage. Default: fixed columns, author-sorted, matching the committed contact sheet.
- **Book.** Confirmed out of scope; only a README/roadmap note that `generate` owns it and it is
  non-reproducible by construction.
