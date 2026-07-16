# Feature: Matched-budget baselines at scale — the three-way head-to-head that was never committed

## Overview

The single most decisive comparison for the thesis — the compressed **color** method
against the uncompressed **HNSW embedding** k-NN and the **TF-IDF** bag-of-words, at
the *same* sample budget and split — has never been committed at scale. The color
method *is* compressed embeddings, so the honest question is "how much accuracy does
the 384→3 compression cost versus using the embeddings directly (HNSW)?", and the
committed evidence does not answer it:

1. **The scaled suite is color-only by construction.**
   `interface/cli/eval_suite.py:_build_cells` (`:155-166`) emits **only**
   `method="color"` cells, and `_build_evaluate_use_case_factory` (`:130-144`) always
   builds a `ColorHistogramClassifier`. So the committed `reports/eval_results.md`
   (from `019-scaled-multidataset-color-eval`) reports color at ~4,000/full samples
   with **no TF-IDF or HNSW rows at that budget**. The `TFIDFClassifier`
   (`infrastructure/evaluation/tfidf_classifier.py:9`) and `HNSWClassifier`
   (`hnsw_classifier.py:12`) already exist and are simply not driven by the suite.

2. **The README table juxtaposes different budgets.** `README.MD:141-147` places the
   Color Method at a **400-sample** budget (`:147`, `:149`) next to TF-IDF 90.63% and
   HNSW 91.99% quoted on the **full 7,600-doc test set**. A matched 400-sample
   three-way *does* exist further down (`README.MD:199-201`: TF-IDF 82.00 / HNSW
   85.25 / Color 81.25), which shows color **last**, ~4 points behind HNSW — but the
   marquee table a reader sees first conflates "smaller sample" with "worse method",
   and the matched comparison is only at 400 samples. **No matched three-way exists
   at the scaled (≥4,000) budget**, and HNSW gains ~6.7 points from 400→full while
   color is pinned near 81%, so the gap most plausibly *widens* at scale — exactly
   the number no one has run.

This feature adds TF-IDF and HNSW as first-class cells in the existing scaled suite
so that, for every dataset and budget, `reports/eval_results.md` carries a
**matched** color / tfidf / hnsw comparison — same seed, same stratified split, same
`max_samples` — reporting classification (accuracy, macro-F1) and, for the
retrieval-capable methods, the real MRR/recall from `025-real-retrieval-metrics`. It
reuses `019`'s suite, fidelity gate, and report writer; the only structural change is
that cells are parameterised by method.

**Honest framing.** The comparison must surface the **rate asymmetry**, not just
accuracy: color spends 12 bits/token (`eval_suite.py:64`,
`COLOR_BITS_PER_TOKEN = 12.0`) while HNSW ranks over the full 384-dim float32
embedding (12,288 bits) and TF-IDF over a sparse lexical vector. The table therefore
tells the whole rate-distortion story in one row: *at a matched sample budget, this
is color's accuracy, this is the baseline's accuracy, and this is the compression
color buys for that accuracy gap.* The point is not to make color win; it is to stop
hiding the comparison.

No new dependency: all classifiers, the retriever port (`025`), the fidelity gate,
and the report writer already exist.

**Depends on:** `025-real-retrieval-metrics` (so the MRR/recall columns are real for
color and HNSW) and `019-scaled-multidataset-color-eval` (the suite this extends).

## Core Domain Concepts

- **Matched budget**: identical `max_samples`, split, and seed across every method in
  a comparison, so an accuracy difference reflects the *method*, never the data slice.
  The suite already threads `seed` and `budget` per cell
  (`evaluation_suite_use_case.py:74`); this feature makes the *method* vary too.
- **Method cell**: an `EvaluationCell` (`evaluation_suite_use_case.py:14-21`)
  extended so `method ∈ {color, tfidf, hnsw}` drives which classifier/retriever the
  factory builds.
- **Rate-annotated row**: each result row carries the method's footprint (color = 12
  bits/token; HNSW = 384×32 bits/token; TF-IDF = lexical, reported as n/a or vector
  size) so accuracy and compression are read together.
- **Retrieval-capable vs classification-only**: color and HNSW rank a corpus and get
  MRR/recall; TF-IDF logistic regression does not and reports classification only
  (its retrieval cell is skipped with a reason), per `025`.

## User Stories

- As a researcher, I want color, TF-IDF, and HNSW evaluated at the **same budget and
  split** on each dataset, so the accuracy gap is attributable to the method.
- As a skeptic, I want the compressed color method compared directly against **HNSW
  over the raw embeddings**, so I can see exactly what the 384→3 compression costs.
- As a reader, I want the README marquee table to stop pairing color@400 with
  baselines@full-test, and instead show one matched, budget-labelled comparison.
- As a maintainer, I want the baselines to flow through the **existing** suite and
  report writer, so there is one command and one committed artifact.
- As a researcher, I want MRR/recall reported for the retrieval-capable methods and
  TF-IDF's retrieval row **explicitly skipped**, not faked as `0.0`.

## Acceptance Criteria

- [ ] Given a dataset and a budget, when the suite runs, then `reports/eval_results.md`
  contains a `color`, a `tfidf`, and an `hnsw` row at that **identical** budget and
  seed (no row mixes budgets), each with accuracy and macro-F1.
- [ ] Given the retrieval-capable methods (color, hnsw), when the suite runs with
  retrieval enabled, then their rows carry a **real** MRR and recall@k (from `025`),
  not `0.0000`.
- [ ] Given TF-IDF (classification-only), when a retrieval metric is requested, then
  its retrieval cell is **skipped with a stated reason**, and it still reports
  classification accuracy/macro-F1.
- [ ] Given each result row, when it is written, then it carries the method's
  bits/token (color = 12.0; HNSW = 12288.0; TF-IDF = n/a or the vector footprint) so
  the rate asymmetry is visible next to accuracy.
- [ ] Given the same datasets, budgets, seed, and distance, when the suite runs
  twice, then the committed metrics are identical (determinism preserved).
- [ ] Given the fidelity gate, when a scaled `color`/`sliced` cell is present, then
  the gate still guards it exactly as in `019`; `tfidf`/`hnsw` cells set
  `requires_fidelity=False`.
- [ ] Given `reports/eval_results.md` is refreshed, when the README "Current
  Performance" table is regenerated, then it is a **view over the matched rows** and
  no longer presents color@400 beside baselines@full-test (the doc edit itself is
  owned by `027-reconcile-retrieval-and-claims`; this feature produces the numbers).
- [ ] Given `tox` is run, then all eight quality gates pass, coverage stays 100%,
  each new test has one logical assertion named `test_should_..._when_...`, and no
  comments exist.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No new business rules. Reuses `EvaluationResult`, the `Classifier`/`Retriever`
ports, and `EvaluationCell`. No `sklearn`/`hnswlib`/`ot` imports enter `domain/`.

### Application Layer (`src/colors_of_meaning/application/`)

- `EvaluationSuiteUseCase` (`application/use_case/evaluation_suite_use_case.py`) is
  reused essentially unchanged — it already runs one `EvaluateUseCase` per cell and
  guards fidelity (`:63-77`). The behavioural change is entirely in *what the factory
  builds per method*, so the suite orchestration is untouched (or gains only a
  retrieval branch if MRR is produced inside the suite rather than a sibling pass).
- If retrieval metrics are produced in the suite, a small addition dispatches
  retrieval-capable cells to `RetrievalEvaluateUseCase` (`025`) and skips TF-IDF with
  a reason; otherwise retrieval is a second suite invocation (Open Question).

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No new adapters — `TFIDFClassifier`, `HNSWClassifier`, `ColorHistogramClassifier`,
and the `025` retrievers already exist. Reuses them.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval_suite.py` — the core change:
  - `_build_cells` (`:155-166`) emits, per dataset, one cell per method in a new
    `--methods {color,tfidf,hnsw}` list (default all three) at the shared budget.
  - `_build_evaluate_use_case_factory` (`:130-144`) dispatches on `cell.method` to
    build `ColorHistogramClassifier` / `TFIDFClassifier` / `HNSWClassifier` (and the
    matching retriever for MRR).
  - `_result_rows`/`_result_row` (`:207-225`) gain a `bits/token` value per method
    and keep the `mrr` column (now real for color/hnsw, blank/`n/a` for tfidf).
  - `COLOR_BITS_PER_TOKEN` (`:64`) generalises to a per-method footprint map.
- `tox.ini`: the existing `[testenv:eval_suite]` is reused; `--methods` is a new flag.
- `configs/*`: reuse `agnews_full.yaml`, `imdb_run.yaml`, `newsgroups_run.yaml`.

### Shared Layer

No changes.

## API Contracts

None. Evaluation is offline/CLI; no HTTP surface changes.

## CLI Impact

- `eval_suite`: add `--methods` (default `color tfidf hnsw`) and, if retrieval
  metrics are produced here, `--task`/`--k-values` mirroring `025`. Default single
  command produces the matched multi-method report. Existing flags and the fidelity
  gate are unchanged.

## Dependency Injection

The factory constructs the per-method classifier/retriever and injects it into
`EvaluateUseCase`/`RetrievalEvaluateUseCase`, exactly as the current factory
constructs the color classifier (`eval_suite.py:137-144`). No Lagom/API change.

## Observability

`correlation-id` logging per cell already exists
(`evaluation_suite_use_case.py:79-92`); it gains `method` (already logged) and
`bits_per_token`. Retrieval cells log `mrr`/`recall_at_k`. No new metrics/tracing.

## Open Questions

- **Retrieval inside the suite or a second pass?** Default: dispatch
  retrieval-capable cells to `RetrievalEvaluateUseCase` within the suite so one run
  yields both classification and retrieval; fall back to a second `--task retrieval`
  invocation if it keeps the suite simpler.
- **TF-IDF footprint value.** Report `n/a`, or the mean non-zero TF-IDF vector size
  in bits? Default: `n/a` with a footnote, since TF-IDF rate is corpus-dependent and
  not the comparison's point.
- **Budget for the committed scaled run.** Reuse `019`'s ≥4,000 (or full where CI
  time allows) and record it per row; the full AG News test set for HNSW/TF-IDF is
  cheap (no EMD re-rank), so consider running baselines at full test set *and* at the
  matched budget, clearly labelled, so both the fair (matched) and the ceiling (full)
  numbers are visible without conflation.
- **Does color ever win?** Expected: no on accuracy; the deliverable is the honest
  gap plus the compression it buys, not a color victory. State this in `027`.
