# Plan: Reproducible authored-corpus authorship report & figures (`tox -e`)

## Implementation Strategy

Close the one remaining "committed artifact with no committed generator" gap by adding `tox -e`
entrypoints that regenerate the authorship figures and the computed parts of
`reports/documents_authorship.md`. Reuse the machinery that already exists
(`VisualizeDocumentsUseCase.execute_projection` / `execute_corpus_signatures`,
`encode_image --layout signature`, the `DocumentCorpusDatasetAdapter` from `022`, and the
`eval_suite`→`eval_results.md` writer pattern); build only the genuinely missing pieces (a
contact-sheet renderer, an `AuthorshipReportUseCase` + writer, a cached scaling manifest, and a
seeding fix for the documents path).

Ship in phases, each independently green under `tox` so review stays small. The figures
generator is the MVP (highest value: 136 committed figures with zero generator); the
report-writer, sweep manifest, and seeding fix follow. Book generation is out of scope —
`[testenv:generate]` already owns `reports/book/*` and cannot be reproducible (live key,
non-deterministic LLM).

## Layer Changes

### Domain (`domain/`)
- `service/figure_renderer.py` — add `render_a4_gallery(sheet_paths, output_path, columns)` to
  the `FigureRenderer` port (the only port addition).
- Optional `model/authorship_report.py` value object holding the computed tables, if it keeps
  the use case pure.

### Application (`application/`)
- `use_case/visualize_documents_use_case.py` — add `execute_a4_gallery(...)` delegating to the
  new renderer method.
- `use_case/authorship_report_use_case.py` (new) — orchestrate train → held-out eval → compute
  corpus/split/results tables → read manifest (or run sweep on opt-in) → emit report; depends
  only on ports (`DatasetRepository`, evaluation collaborators, `FigureRenderer`, a
  report-writer/manifest port).

### Infrastructure (`infrastructure/`)
- `visualization/matplotlib_figure_renderer.py` — implement `render_a4_gallery` (author-sorted
  grid tiling of the per-book PNGs; deterministic, no timestamped metadata).
- `dataset/document_corpus_dataset_adapter.py` — identify and seed the residual non-determinism
  (paragraph sampling / embedding order / budget interaction) so documents runs are bit-identical;
  reuse `shared/determinism.py`.
- New thin JSON manifest reader/writer for `reports/data/authorship_scaling.json` and a
  template-based report-writer for `documents_authorship.md` (mirroring the `eval_suite` writer).

### Interface (`interface/`)
- `cli/authorship.py` (new) + `[testenv:authorship]` in `tox.ini` — default writes
  `reports/documents_authorship.md`; `--refresh-scaling` runs the 8-seed × 3-cap sweep (via
  seed-varied configs, since `train` exposes no `--seed`) and rewrites the manifest.
- `cli/visualize.py` (or `visualize_corpus.py`) — add `--source documents` writing the
  `documents_*` figures + 133 `a4/*.png` + gallery, and/or `[testenv:visualize_documents]`.
- README CLI reference: document the new env(s); roadmap note that `generate` owns book and it
  is non-reproducible.

### Shared (`shared/`)
- Reuse `document_corpus.py` and `determinism.py`. A pure author-ordering helper for the gallery
  may live here if framework-free.

## Dependency Injection

New use case and CLI construct/inject the `DatasetRepository` (HF or documents), `FigureRenderer`,
evaluation collaborators, and manifest/report-writer ports the same way `eval`, `rate_distortion`,
and `visualize_corpus` do today. No Lagom/API container changes.

## Task List

1. [ ] Add `render_a4_gallery` to the `FigureRenderer` port + matplotlib impl (deterministic
   author-sorted tiling); unit-test the tiling with a real render to a `tmp_path`.
2. [ ] Add `VisualizeDocumentsUseCase.execute_a4_gallery` + wire a documents figures path
   (`--source documents` on `visualize`/`visualize_corpus` or a `visualize_documents` CLI) that
   writes `documents_color_tsne.png`, `documents_color_signatures.png`, 133 `a4/*.png`, and the
   gallery, to their exact committed paths.
3. [ ] Seed the documents path so repeated documents figure/rate-distortion runs are
   byte-identical; add a regression test that runs a synthetic documents artifact twice and
   asserts equality.
4. [ ] Add `AuthorshipReportUseCase` + report-writer that emits the corpus/split/held-out tables
   into `documents_authorship.md` from a real train/eval, preserving narrative prose from a
   template; `[testenv:authorship]`.
5. [ ] Add the committed `reports/data/authorship_scaling.json` manifest + reader; render the
   data-scaling table from it by default; `--refresh-scaling` opt-in re-runs the 24-training
   sweep (documented non-CI/long-running) and rewrites the manifest.
6. [ ] Document the new env(s) in the README CLI reference; note book stays on `tox -e generate`
   and is non-reproducible; add the delivery note to `.specs/ROADMAP.md`.
7. [ ] Run `tox`; confirm all 8 gates + 100% coverage; regenerate the figures on the real
   `./documents/` corpus locally and confirm a second run is byte-identical.

## Testing Strategy

- **Synthetic corpus only.** Like `022`, every test builds a `tmp_path`
  `documents/<author>/<work>.txt` tree — no real `./documents/`, no network, no live API key.
- **Real render, not only mocks.** Per the `visualize_corpus` lesson (mocked-renderer tests hid
  real crashes an adversarial review caught), the gallery/figure paths get at least one test that
  invokes the *real* `MatplotlibFigureRenderer` to a `tmp_path` and asserts a non-empty PNG,
  alongside mocked-collaborator CLI tests.
- **Determinism assertions.** A test runs a documents artifact twice and asserts byte-equality
  (the reproducibility AC). Guard against the known RNG-leak pitfall: no bare `torch.manual_seed`
  in a test body that leaks global state under `--random-order`; seed through the documented path.
- **Colour-from-document-embedding.** Signatures/A4/t-SNE must read each document's colour from
  its document embedding, not the mean of sentence colours (mean-of-Lab cancels chroma) — assert
  the per-document colour path, consistent with the corpus/compass convention.
- **House rules.** One logical assertion per test, `test_should_..._when_...`, `assertpy` for
  entity tests / plain asserts for ML; keep every test function xenon grade A (extract
  comprehensions/filters into helpers — the grade-A gate scans tests too).
- Manifest read path is fully covered with a committed-shape fixture; the `--refresh-scaling`
  sweep is covered with a stubbed trainer (no real 24-run training in CI).

## Observability Plan

`correlation-id` structured logging (stdlib logger + `uuid`, per the repo convention — no new
tracing/metrics infra): authors/works discovered, per-split paragraph counts, held-out
accuracies, each figure path written, and whether the scaling table was read from the manifest or
freshly swept.

## Risks and Mitigations

- **Matplotlib PNG non-determinism.** Timestamped metadata can defeat byte-equality; set
  deterministic metadata / fixed DPI as the renderers already do for committed figures, and
  assert byte-equality only after that is pinned.
- **Mocked-renderer blind spots.** Mitigated by the mandatory real-render smoke test above.
- **Sweep expense / no `--seed` flag.** The 24-training sweep is long and needs seed-varied
  configs; keep it opt-in behind `--refresh-scaling`, cache results in the committed manifest,
  and stub the trainer in tests so CI never runs it.
- **Documents git-ignored.** The real authorship artifacts remain local/opt-in and are not
  CI-regenerated (like `022`/`documents_rate_distortion`); only synthetic-tree tests run in CI.
- **Scope creep.** If `028` grows too large, split after Phase 2 (figures shipped) into a
  follow-up for the report-writer + sweep manifest, keeping each PR reviewable.
- **Over-reach on book.** Explicitly no new env for `generate`; only a documentation note, so no
  non-reproducible path is presented as a reproducible artifact.
