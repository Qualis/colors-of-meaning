# Plan: Matched-budget baselines at scale — the three-way head-to-head that was never committed

## Implementation Strategy

Make the decisive comparison committable by parameterising the existing scaled suite
over *method*, not just dataset/budget/distance. `019` already built everything hard:
the fidelity-gated sliced-Wasserstein proxy, the per-cell `EvaluationSuiteUseCase`
(`evaluation_suite_use_case.py:52-77`), the deterministic seeded splits, and the
committed-report writer (`eval_suite.py:249-283`). The only reason `reports/eval_results.md`
is color-only is that `_build_cells` (`eval_suite.py:155-166`) hardcodes
`method="color"` and the factory (`:130-144`) always builds a color classifier. This
plan varies the method so every dataset/budget yields a matched color / tfidf / hnsw
comparison, with real MRR/recall (from `025`) for the retrieval-capable methods.

The baselines are already implemented and cheap relative to color: `TFIDFClassifier`
(`tfidf_classifier.py`) is TF-IDF + logistic regression; `HNSWClassifier`
(`hnsw_classifier.py`) is k-NN over the raw 384-dim embeddings via `hnswlib`. Neither
uses the exact-EMD re-rank, so a full-test-set baseline run is inexpensive — meaning
we can commit *both* the matched-budget number (fair) and the full-test number
(ceiling), each explicitly labelled, ending the budget conflation for good.

Two honesty rules drive the design:

1. **Matched by construction.** Every method in a comparison shares one
   `(dataset, budget, seed)`; the suite already passes `seed`/`budget` per cell
   (`evaluation_suite_use_case.py:74`), so making `method` vary keeps the split
   identical across methods automatically.
2. **Rate travels with accuracy.** Each row carries the method's bits/token, so the
   table reads as rate-distortion, not a leaderboard: color's accuracy deficit is
   shown next to the compression it buys.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- No changes. `EvaluationResult`, `EvaluationCell`, and the `Classifier`/`Retriever`
  ports are reused.

### Application Layer (`src/colors_of_meaning/application/`)

- `EvaluationSuiteUseCase` — reused as-is for classification. If retrieval metrics
  are produced in the same run, add a thin branch: for retrieval-capable cells build
  a `RetrievalEvaluateUseCase` (`025`) via the factory and merge its MRR/recall into
  the cell result; skip TF-IDF retrieval with a recorded reason. Keep the fidelity
  rejection (`_reject_unfaithful_scaled_cells` `:67-69`) unchanged.
- Tests: the suite runs one cell per method (mocked use cases — no datasets); a
  TF-IDF retrieval request is skipped with a reason; an unfaithful scaled color cell
  still raises.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- No new adapters. `TFIDFClassifier`, `HNSWClassifier`, `ColorHistogramClassifier`,
  and the `025` retrievers are reused.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval_suite.py`:
  - Add `--methods: List[str]` to `EvalSuiteArgs` (default `["color","tfidf","hnsw"]`).
  - `_build_cells` (`:155-166`): nested over `(dataset, budget) × method`, setting
    `requires_fidelity = (method == "color" and distance == "sliced")` and a
    per-method `bits_per_token`.
  - `_build_evaluate_use_case_factory` (`:130-144`): dispatch on `cell.method` —
    `color` → `_build_color_classifier` (existing `:116-127`), `tfidf` →
    `TFIDFClassifier()`, `hnsw` → `HNSWClassifier(embedding_adapter, k=...)`.
  - `_result_rows`/`_result_row` (`:207-225`): add a `bits/token` cell per method and
    keep `mrr` (real for color/hnsw, `n/a` for tfidf); header updated.
  - Replace the module constant `COLOR_BITS_PER_TOKEN` (`:64`) with a
    `_BITS_PER_TOKEN = {"color": 12.0, "hnsw": 12288.0, "tfidf": None}` map.
  - `_reproduce_command` (`:236-246`) includes `--methods`.
- `tox.ini`: reuse `[testenv:eval_suite]`.
- Architecture test: add `--methods` handling to the CLI→use-case rule; assert the
  suite CLI may import the three classifiers while `domain/` may not.
- `README.MD`: numbers refreshed here, but the marquee-table rewrite is `027`'s
  Definition of Done, not this feature's.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes.

## Dependency Injection

The factory builds the per-method classifier/retriever and injects it into the
evaluate/retrieval use case — the pattern the factory already uses for color
(`eval_suite.py:137-144`). No Lagom/API container change.

## Task List

1. [ ] interface: `--methods` flag; `_build_cells` fans out over methods at the
   shared budget/seed; `requires_fidelity` only for `color`/`sliced`.
2. [ ] interface: factory dispatch on `cell.method` to the three classifiers (+
   retrievers for MRR); per-method `bits/token` map.
3. [ ] interface: report writer adds the `bits/token` column and keeps a real `mrr`
   column; `tfidf` retrieval shown as `n/a` with a footnote.
4. [ ] application: retrieval branch in the suite (or a documented second pass) that
   skips TF-IDF retrieval with a reason; tests with mocked use cases.
5. [ ] interface: tests — `_build_cells` yields one cell per method per dataset at
   identical budget/seed; factory returns the right classifier per method; report row
   renders the footprint and real MRR.
6. [ ] integration (marked, not unit): run the matched suite on AG News (+ IMDB,
   20NG) at the `019` budget and at full-test for the cheap baselines; commit
   `reports/eval_results.md` with matched rows; record library/seed provenance
   (`_provenance_line` `:228-233` reused).
7. [ ] run `tox`; confirm 8 gates + 100% coverage; reproduce one matched row from its
   config end-to-end.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...`, no network
in unit tests (mock the embedding adapter, dataset repos, and use cases; tiny
synthetic codebook). Key tests:

- **Cell fan-out:** `_build_cells` with `--methods color tfidf hnsw` produces three
  cells per dataset at one budget and seed; `requires_fidelity` true only for the
  `color`/`sliced` cell.
- **Factory dispatch:** `cell.method == "tfidf"` builds a `TFIDFClassifier`;
  `"hnsw"` builds an `HNSWClassifier`; `"color"` builds the color classifier.
- **Row rendering:** a color row shows `12.00` bits/token and a real MRR; a tfidf row
  shows `n/a` bits/token and no MRR; budgets never mix within a comparison.
- **Retrieval skip:** requesting retrieval for TF-IDF records a reason and still
  yields a classification result (mocked).
- **Determinism:** two suite builds with the same args yield identical cells.
- **Scaled numbers:** produced by the integration run and committed to
  `reports/eval_results.md`; not asserted in unit tests (no model/dataset downloads
  in CI unit runs).

## Observability Plan

`correlation-id` per-cell logging already exists
(`evaluation_suite_use_case.py:79-92`); extend the payload with `bits_per_token` and,
for retrieval cells, `mrr`/`recall_at_k`. No new metrics/tracing.

## Risks and Mitigations

- **Budget conflation persists in prose.** Mitigation: the report writes matched
  rows with an explicit `budget` column; `027` rewrites the README table to the
  matched view and removes the color@400-vs-baseline@full pairing.
- **Baselines look artificially strong at full test set.** Mitigation: commit both
  the *matched-budget* number (fair) and the *full-test* number (ceiling), each
  labelled; never compare across the two.
- **Determinism drift across methods.** Mitigation: one seeded stratified split per
  `(dataset, budget)` shared by all methods; `hnswlib` `set_num_threads(1)` +
  fixed `random_seed` already in the classifiers; a repeat-build test.
- **Suite complexity creep.** Mitigation: keep orchestration in
  `EvaluationSuiteUseCase` unchanged; all method variance lives in the CLI factory.
- **Color loses and it is tempting to bury it.** Mitigation: the acceptance criteria
  require the baseline rows to be committed; `027` states the gap and the compression
  trade-off plainly rather than hiding it.
