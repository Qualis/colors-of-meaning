# Feature: Real retrieval metrics — implement the `Retriever` port and stop reporting MRR = 0

## Overview

The project is framed as "semantic compression **and retrieval**" (`README.MD:7`,
`:13`, `:38`), but **no retrieval-quality number is ever computed**. Two concrete
defects make the "retrieve" pillar unmeasured:

1. **MRR and recall@k are hardcoded to zero on the path that actually runs.**
   `EvaluateUseCase.execute` only ever calls
   `MetricsCalculator.calculate_classification_metrics`
   (`application/use_case/evaluate_use_case.py:34`), which returns `recall_at_k={}`
   and `mrr=0.0` unconditionally
   (`infrastructure/evaluation/sklearn_metrics_calculator.py:22-23`). Every `mrr`
   cell in the committed `reports/eval_results.md` (written by
   `interface/cli/eval_suite.py:207-225`, which prints a `mrr` column) is therefore
   a constant `0.0000` presented as a measurement.

2. **The real retrieval metric exists but is dead code fed by a dead port.**
   `SklearnMetricsCalculator.calculate_retrieval_metrics`
   (`sklearn_metrics_calculator.py:27-49`) already computes recall@k and MRR
   correctly from ranked results, and the `Retriever` port that would feed it is
   defined (`domain/service/retriever.py:7-14`: `fit(samples)`,
   `search(query, k) -> List[Tuple[EvaluationSample, float]]`) — but it has **zero
   implementations** and is invoked nowhere in `src/`. The ranking it needs is
   already produced and discarded: `ColorHistogramClassifier._rerank_by_distance`
   (`infrastructure/evaluation/color_histogram_classifier.py:104-111`) sorts
   candidates by exact perceptual distance and then returns **only the labels**,
   throwing away the `(sample, distance)` ordering.

This feature makes retrieval a **measured** result: implement the `Retriever` port
for the color method (and for the embedding baseline) by surfacing the ranking the
classifier already computes, then route it through the existing
`calculate_retrieval_metrics` so committed reports carry a genuine MRR and
recall@k instead of a constant zero.

**Honest scope — this does not oversell.** The relevance signal in
`calculate_retrieval_metrics` is **class-label homogeneity**: a retrieved neighbour
is a "hit" when it shares the query's class
(`sklearn_metrics_calculator.py:59-65`, `compute_recall_at_k` `:73-85`,
`compute_reciprocal_rank` `:87-95`). This is the standard metric-learning
"Recall@K" convention and is legitimate, but it is **correlated with the k-NN
classification accuracy already reported** — it is a *ranking-quality view of the
same neighbourhoods*, not an independent result that rescues the thesis. This spec
introduces no graded relevance judgements the corpus does not have, and the report
must label these metrics as label-based retrieval so no reader mistakes them for
human-judged IR.

No new third-party dependency is introduced: `scikit-learn`, `hnswlib`, `pot`, and
`scipy` are already declared (`setup.cfg` `install_requires`). An optional graded
`ndcg_score` view (below) uses `sklearn.metrics`, already imported by
`sklearn_metrics_calculator.py:2`.

**Depends on:** none (foundational). **Unblocks:**
`026-matched-budget-baseline-eval`, which needs a real MRR/recall column for every
method.

## Core Domain Concepts

- **Retriever**: the existing `domain/service/retriever.py` port —
  `fit(training_samples)` then `search(query, k) -> List[(EvaluationSample,
  distance)]` returning the k nearest labelled documents in ranked order. Today an
  unimplemented abstraction; this feature gives it its first concrete adapters.
- **Ranked search result**: the ordered `(sample, distance)` list per query that
  `calculate_retrieval_metrics` consumes as `search_results`. The color classifier
  already builds this internally (`_rerank_by_distance`) and collapses it to a
  label; the retriever preserves it.
- **Label-based relevance**: relevance = "shares the query's class label". Recall@k
  is "a same-class document appears in the top k"; reciprocal rank is `1 / rank` of
  the first same-class hit. Correlated with k-NN classification by construction —
  reported as such, never as human-judged retrieval.
- **Retrieval-capable method**: a method that can rank a corpus by query similarity
  (color histograms via `DistanceCalculator`; embeddings via `hnswlib`). A pure
  classifier such as TF-IDF + logistic regression (`tfidf_classifier.py:9`) is
  **not** retrieval-capable and reports no MRR/recall (see Open Questions).

## User Stories

- As a researcher, I want the color method's MRR and recall@k to be **computed from
  real rankings** so the committed `reports/eval_results.md` stops printing a
  constant `0.0000` MRR.
- As a skeptic, I want the retrieval metric's relevance definition stated as
  **class-label homogeneity** so I am not misled into reading it as human-judged IR.
- As a maintainer, I want the dead `Retriever` port either **implemented and used**
  or deleted, so there is no zero-implementation abstraction left in `domain/`.
- As a contributor, I want the color classifier and the color retriever to **share
  one retrieval core** so the HNSW-candidate + exact-rerank logic is not duplicated.
- As a researcher, I want an embedding-space retriever too, so
  `026-matched-budget-baseline-eval` can put color retrieval next to embedding
  retrieval on the same queries.

## Acceptance Criteria

- [ ] Given a fitted color retriever and a query, when `search(query, k)` is called,
  then it returns exactly `min(k, corpus_size)` `(EvaluationSample, distance)` pairs
  in non-decreasing distance order (the ranking `_rerank_by_distance` already
  computes but currently discards).
- [ ] Given a query whose first same-class neighbour is at rank 2 in the returned
  ranking, when reciprocal rank is computed, then it is `0.5`
  (`compute_reciprocal_rank`, known-answer).
- [ ] Given a retrieval evaluation over a labelled corpus, when it runs, then the
  reported `EvaluationResult.mrr` and `recall_at_k` come from
  `calculate_retrieval_metrics` on real rankings and are **not** `0.0`/`{}` for a
  non-degenerate input.
- [ ] Given the color classification path, when it runs, then it produces the same
  predictions as before (the majority vote over the same re-ranked neighbours is
  unchanged; the retrieval refactor is behaviour-preserving for classification).
- [ ] Given the `Retriever` port, when the codebase is searched, then it has at
  least one concrete implementation wired into a use case (no zero-implementation
  abstraction remains), and `ColorHistogramClassifier` and the color retriever share
  a single retrieval core with no copy-pasted HNSW/rerank block.
- [ ] Given the retrieval report or CLI output, when MRR/recall@k are shown, then
  they are labelled as **label-based retrieval** (relevance = shared class), not
  presented as graded/human-judged IR.
- [ ] Given a method that cannot rank a corpus (TF-IDF logistic regression), when a
  retrieval evaluation is requested for it, then it is **skipped with an explicit
  reason**, not reported as MRR `0.0`.
- [ ] Given `tox` is run, then all eight quality gates pass and coverage stays 100%,
  every new test has one logical assertion named `test_should_..._when_...`, no
  comments exist, and the domain layer imports no `sklearn`/`hnswlib`/`ot`/`torch`.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/service/retriever.py` — the existing `Retriever` ABC is **kept and
  implemented** (not deleted); no signature change is required —
  `search(query, k) -> List[Tuple[EvaluationSample, float]]` is exactly the shape
  `calculate_retrieval_metrics` expects. If the label-only relevance needs a k-list,
  it is passed to the use case, not the port.
- `domain/model/evaluation_result.py` — reused unchanged; it already carries
  `recall_at_k: Dict[int, float]` and `mrr: float` (`:9-10`). No new model unless a
  distinct `RetrievalResult` is preferred over reusing `EvaluationResult` (Open
  Question).
- No `sklearn`/`hnswlib`/`ot` imports enter `domain/` (architecture test enforces
  it, mirroring the existing isolation rules in
  `tests/colors_of_meaning/test_synesthetic_architecture.py`).

### Application Layer (`src/colors_of_meaning/application/`)

- New `RetrievalEvaluateUseCase(retriever, metrics_calculator, dataset_repository,
  k_values)`: fits the retriever on the train split, calls `search` for each test
  query, assembles `search_results`, and returns
  `metrics_calculator.calculate_retrieval_metrics(queries, search_results,
  k_values, bits_per_token)`. Mirrors `EvaluateUseCase` (`evaluate_use_case.py`) but
  on the retrieval port instead of the classifier. `correlation-id` logging of
  `{dataset, method, k_values, mrr, recall_at_k}`.
- `EvaluateUseCase` is left for classification; whether it stops reporting a
  meaningless `mrr=0.0` (drop it from the classification result) or leaves it is an
  Open Question — the honest default is to report MRR only via the retrieval path.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- Factor a shared retrieval core out of `ColorHistogramClassifier` (the
  HNSW-candidate retrieve `:96-102` + exact-distance rerank `:104-111`) so both the
  classifier and a new retriever use it — no duplicated block (PREFER EDITING /
  no-duplication house rule). Shape: a `ColorHistogramRetriever(Retriever)` that
  owns the index + rerank and returns ranked `(sample, distance)` tuples; the
  classifier becomes "retrieve top-k, then majority-vote", delegating to the same
  core. Predictions must be unchanged.
- New `EmbeddingRetriever(Retriever)` wrapping the `hnswlib` L2 index already built
  by `HNSWClassifier.fit` (`hnsw_classifier.py:30-51`), exposing ranked neighbours
  from `knn_query` (`:59`) for the matched retrieval comparison in `026`.
- `SklearnMetricsCalculator` — no change needed to `calculate_retrieval_metrics`
  (`:27-49`); it becomes **reachable**. Optional: add a graded `ndcg_score` view
  (`sklearn.metrics.ndcg_score`, context7-grounded, see PLAN) computed per query
  from a same-class relevance vector and `-distance` scores; gated behind the report
  so it is additive, not a replacement for the label-based MRR.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval.py`: add `--task {classification,retrieval}` (default
  `classification` to preserve current behaviour). For `retrieval`, construct the
  matching `Retriever` and run `RetrievalEvaluateUseCase`; `_print_results` shows
  real MRR/recall@k with an explicit "label-based retrieval (relevance = shared
  class)" caption.
- No new top-level command is required if `--task` suffices; a sibling `retrieve`
  command is an alternative (Open Question).

### Shared Layer

No changes.

## API Contracts

None. Retrieval evaluation is an offline/CLI concern; the `POST /query/palette`
contract is unaffected. (Note: the query-by-palette endpoint is a *palette* lookup,
not this corpus-retrieval metric.)

## CLI Impact

- `eval`: add `--task {classification,retrieval}` and `--k-values` (e.g. `1 5 10`);
  `classification` is the unchanged default. `retrieval` prints MRR and recall@k
  captioned as label-based, and refuses (with a reason) for non-retrieval methods.

## Dependency Injection

The CLI constructs the chosen `Retriever` (color or embedding) and injects it into
`RetrievalEvaluateUseCase` alongside `SklearnMetricsCalculator` and the dataset
repository — the same constructor-injection pattern `eval.py`/`eval_suite.py`
already use for classifiers and distance calculators. No Lagom container or API
wiring changes.

## Observability

`correlation-id` structured logging: `RetrievalEvaluateUseCase` logs `{dataset,
method, k_values, mrr, recall_at_k, query_count}` once per run. No per-query logging
in the ranking loop (hot path). No new metrics/tracing.

## Open Questions

- **Reuse `EvaluationResult` or add `RetrievalResult`?** `EvaluationResult` already
  fits (`recall_at_k`, `mrr`); default is to reuse it and set `accuracy=macro_f1=0`
  for retrieval rows (as `calculate_retrieval_metrics` already does, `:43-44`),
  clearly labelled. A dedicated `RetrievalResult` is cleaner but adds a model.
- **Drop the meaningless classification MRR?** `calculate_classification_metrics`
  returns `mrr=0.0` (`:23`). Default: keep the field for the frozen dataclass but
  stop surfacing it on classification rows; report MRR only from the retrieval path.
- **`--task` flag vs a sibling `retrieve` CLI.** Default: a `--task` flag on `eval`
  to avoid a second entry point; revisit if the argument sets diverge.
- **Graded `ndcg_score`.** Add it as an *additional* column (graded ranking view) or
  omit for now? Default: add it behind the report as clearly-labelled extra signal,
  since it needs no new dependency; keep the label-based MRR as the primary number.
- **TF-IDF retrieval.** A k-NN retriever over TF-IDF vectors could exist, but the
  wired TF-IDF is a logistic-regression classifier (`tfidf_classifier.py:9`).
  Default: report TF-IDF as classification-only and skip its retrieval row with a
  reason (revisited in `026`).
