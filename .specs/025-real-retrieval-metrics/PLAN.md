# Plan: Real retrieval metrics — implement the `Retriever` port and stop reporting MRR = 0

## Implementation Strategy

Turn retrieval from an asserted pillar into a measured one by (1) implementing the
already-defined `Retriever` port so ranked neighbourhoods stop being discarded, (2)
routing those rankings through the already-implemented-but-never-called
`calculate_retrieval_metrics`, and (3) reporting the result honestly as label-based
retrieval, correlated with k-NN classification rather than independent of it.

The ranking already exists. `ColorHistogramClassifier` retrieves candidates from an
`hnswlib` cosine index (`color_histogram_classifier.py:96-102`) and re-ranks them by
exact perceptual distance (`_rerank_by_distance` `:104-111`) — then collapses the
sorted `(distance, label)` list to a majority vote, dropping the ranking. The fix is
structural, not algorithmic: extract that retrieve-then-rerank core, have the
retriever return the ranked `(sample, distance)` list, and let the classifier stay a
thin "retrieve top-k → majority vote" over the same core. Classification predictions
must not change.

On the metric side, nothing needs to be invented: `calculate_retrieval_metrics`
(`sklearn_metrics_calculator.py:27-49`) already computes recall@k and MRR correctly;
it is simply unreachable because `EvaluateUseCase` only calls the classification
method (`evaluate_use_case.py:34`). A new `RetrievalEvaluateUseCase` calls it.

**Library grounding (context7 / scikit-learn).** The corpus-retrieval metric here is
naturally *per-corpus with class-as-relevance and variable-length rankings*, which
`sklearn.metrics.label_ranking_average_precision_score` and `top_k_accuracy_score`
do **not** fit (both expect a fixed `(n_samples, n_labels)` score matrix over a label
set, not ranked corpus neighbours). So the existing hand-rolled MRR/recall — pinned
by known-answer tests — stays the primary implementation. `sklearn.metrics.ndcg_score`
*does* fit as an **optional graded view**: per query, build a relevance vector
(`1.0` for same-class candidates, else `0.0`) and a score vector (`-distance`), and
`ndcg_score([relevance], [scores], k=k)` yields a standard graded ranking number with
no new dependency (`sklearn` is already imported at `sklearn_metrics_calculator.py:2`).
It is additive; it does not replace the label-based MRR.

Three decisions keep it honest and clean:

1. **One retrieval core.** The classifier and retriever share the HNSW-candidate +
   exact-rerank logic; the classifier = retriever + majority vote. No duplicated
   block, and classification output is provably unchanged.
2. **Measure, don't relabel.** MRR/recall come from real rankings via the existing
   metric; the report captions them as label-based retrieval (relevance = shared
   class) so nothing is presented as human-judged IR.
3. **No zero-implementation abstractions.** The `Retriever` port gains concrete
   adapters and a use case; the dead-code smell my audit flagged is removed.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/service/retriever.py` — kept; no signature change. `search` already
  returns `List[Tuple[EvaluationSample, float]]`, exactly what
  `calculate_retrieval_metrics` consumes.
- `domain/model/evaluation_result.py` — reused; `recall_at_k`/`mrr` already present.
- Tests: none new here beyond what the model already has; architecture test asserts
  the retriever adapters live in `infrastructure/` and `domain/` stays framework-free.

### Application Layer (`src/colors_of_meaning/application/`)

- `application/use_case/retrieval_evaluate_use_case.py` (new) —
  `RetrievalEvaluateUseCase(retriever, metrics_calculator, dataset_repository,
  k_values)`; `execute(bits_per_token, max_samples, seed) -> EvaluationResult`:
  fit on `train`, `search` each `test` query for `max(k_values)` neighbours, pass
  `(queries, search_results, k_values)` to `calculate_retrieval_metrics`.
  `correlation-id` logging.
- Tests: with a stub `Retriever` returning known rankings and a stub dataset repo
  (no network), assert the use case forwards real rankings and returns non-zero MRR
  for a corpus where a same-class neighbour is retrieved.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- Extract a `_ColorHistogramRetrievalCore` (or a module-level helper) holding the
  index build (`color_histogram_classifier.py:41-68`), candidate retrieval
  (`:96-102`), and exact rerank (`:104-111`) returning the sorted
  `[(training_index, distance)]`. `ColorHistogramClassifier.predict` becomes
  "core.search → majority vote"; behaviour unchanged.
- `infrastructure/evaluation/color_histogram_retriever.py` (new) —
  `ColorHistogramRetriever(Retriever)` delegating to the shared core, mapping
  `training_index -> EvaluationSample` and returning ranked `(sample, distance)`.
- `infrastructure/evaluation/embedding_retriever.py` (new) —
  `EmbeddingRetriever(Retriever)` over the `hnswlib` L2 index that `HNSWClassifier`
  already builds (`hnsw_classifier.py:30-51`); `search` returns ranked neighbours
  from `knn_query` (`:59`) as `(sample, l2_distance)`.
- `infrastructure/evaluation/sklearn_metrics_calculator.py` — no change to
  `calculate_retrieval_metrics`. Optional: add a small `calculate_graded_ndcg`
  helper using `sklearn.metrics.ndcg_score` per the grounding above; kept separate so
  the frozen `EvaluationResult` contract is untouched (Open Question in SPEC).

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval.py` — add `--task {classification,retrieval}` (default
  `classification`) and `--k-values`. A `_create_retriever(method, ...)` helper
  builds the color or embedding retriever; `_print_results` captions retrieval rows
  as label-based. Non-retrieval methods raise a clear "retrieval not supported for
  {method}" message.
- Architecture test: add the two retriever modules to the infrastructure→domain
  rule; assert they may import `hnswlib`/`sklearn` while `domain/` may not.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes.

## Dependency Injection

CLIs build the chosen `Retriever` and inject it (plus `SklearnMetricsCalculator` and
the dataset repository) into `RetrievalEvaluateUseCase`, matching the existing
constructor-injection in `eval.py`/`eval_suite.py`. No Lagom or API container change.

## Task List

1. [ ] infrastructure: extract the shared color retrieval core from
   `ColorHistogramClassifier`; prove classification predictions are unchanged
   (characterisation test on a tiny synthetic corpus + codebook).
2. [ ] infrastructure: `ColorHistogramRetriever(Retriever)` + tests (ranked order
   non-decreasing; returns `min(k, corpus)` tuples; identical query retrieves itself
   at distance ~0).
3. [ ] infrastructure: `EmbeddingRetriever(Retriever)` + tests (ranked order; same
   `hnswlib` index as `HNSWClassifier`).
4. [ ] application: `RetrievalEvaluateUseCase` + tests (real MRR/recall from stubbed
   rankings; non-zero for a same-class hit; forwards `k_values`).
5. [ ] metric: known-answer tests for `calculate_retrieval_metrics` via the port
   (first same-class hit at rank 2 → RR 0.5; recall@k boundaries); optional
   `ndcg_score` graded helper + its own known-answer test.
6. [ ] interface: `eval --task/--k-values`, retrieval print path with the
   label-based caption, non-retrieval refusal; architecture-test wiring.
7. [ ] run `tox`; confirm 8 gates + 100% coverage; run `eval --task retrieval` on a
   small budget and confirm a non-zero MRR appears where the old path printed 0.0000.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...` names, no
network in unit tests (mock `SentenceEmbeddingAdapter` and the dataset repository;
use a small synthetic `ColorCodebook.create_uniform_grid`). Key tests:

- **Retriever (real logic):** on a tiny labelled corpus, `search(query, k)` returns
  ranked `(sample, distance)` in non-decreasing distance; the query's own document
  ranks first at ~0 distance.
- **Classifier unchanged:** the refactored classifier returns the *same* predictions
  as a captured baseline on a fixed synthetic corpus (behaviour-preserving refactor).
- **Metric known-answers:** first same-class neighbour at rank 2 → RR 0.5; recall@1
  is 0 and recall@2 is 1 for a corpus where the only same-class hit is second;
  empty rankings → 0.0 without raising.
- **Use case:** stubbed retriever + stubbed dataset repo produce a non-zero MRR
  `EvaluationResult`; no dataset is downloaded.
- **CLI branches:** `--task retrieval` selects the retriever and prints the
  label-based caption; a non-retrieval method raises the documented message.
- **ndcg (optional):** `ndcg_score` helper matches a hand-computed value on a
  known relevance/scores pair.

## Observability Plan

`correlation-id` logging: `RetrievalEvaluateUseCase` logs `{dataset, method,
k_values, mrr, recall_at_k, query_count}` once per run; no per-query logging in the
ranking loop. No new metrics/tracing.

## Risks and Mitigations

- **Refactor changes classification silently.** Mitigation: a characterisation test
  pins predictions before/after the core extraction; the classifier keeps its exact
  majority-vote over the same re-ranked neighbours.
- **Retrieval metric read as human-judged IR.** Mitigation: every surface (CLI
  caption, report column header) labels it "label-based retrieval (relevance =
  shared class)"; the SPEC states the correlation with classification explicitly.
- **Retriever/classifier duplication.** Mitigation: a single shared core; an
  architecture/test guard that both delegate to it.
- **hnswlib determinism.** Mitigation: reuse the existing `set_num_threads(1)` and
  fixed `random_seed=100` already in the classifiers; assert repeat-call stability.
- **Scope creep into graded relevance.** Mitigation: `ndcg_score` is optional and
  additive; the corpus has only class labels, so no fabricated graded judgements.
