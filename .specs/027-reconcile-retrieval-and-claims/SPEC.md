# Feature: Reconcile claims (round 2) — retrieval-vs-classification wording, stale counts, and design.md drift

## Overview

`016-p2-6-reconcile-docs` made the compression claim precise (`~1024:1`) and stopped
mislabelling the fixed grid as "VQ". A second round of documentation drift remains —
claims the *code and its own numbers* do not support — and this step reconciles it,
backed by the real metrics that `025-real-retrieval-metrics` and
`026-matched-budget-baseline-eval` produce. Five concrete discrepancies:

1. **Classification accuracy is relabelled as retrieval.** `README.MD:17` states the
   "colour method **retrieves** at **81.8%** accuracy at the 4,000-document scale",
   and `:13`/`:38` lead with "retrieve" as a first-class capability. But until `025`,
   MRR/recall were hardcoded to `0.0` (`sklearn_metrics_calculator.py:22-23`) and the
   `81.8%` figure is the **classification** accuracy from the scaled run, not a
   retrieval metric. The word "retrieves" must attach to the real MRR/recall from
   `025`/`026`, or the figure must be relabelled as classification accuracy.

2. **The marquee table conflates budgets.** `README.MD:141-147` shows the Color
   Method at a **400-sample** budget (`:147`, `:149`) beside TF-IDF/HNSW quoted on the
   **full test set**. The matched-budget three-way that belongs there is produced by
   `026`; this step replaces the table with that matched view (color/tfidf/hnsw at one
   budget) and removes the color@400-vs-baseline@full pairing.

3. **A self-contradicting corpus count.** `README.MD:20` describes a "22-author /
   **133-work**" corpus, while `:29` ("all **73** books verified") and `:55` ("each of
   the **73** books") cite 73. The committed gallery and authorship report reflect 133
   (`reports/documents_authorship.md`); `73` is stale and internally inconsistent.

4. **`docs/design.md` mislabels the distance and the training objective.**
   `design.md:28` calls it "**Wasserstein-2** distance", but the calculator uses a
   non-squared Euclidean ground cost with `ot.emd2` and no final square root — i.e.
   **Wasserstein-1** (`infrastructure/ml/wasserstein_distance_calculator.py`).
   `design.md:83` says "Training uses **random targets** (unsupervised)", which is
   stale since `002-p0-2-structure-preserving-training` replaced random-target MSE with
   a structure-preserving similarity-distillation objective. `design.md:90` lists
   "hue=topic, lightness=sentiment, saturation=concreteness" as if fully realised,
   which `020-falsifiable-interpretability-validation` showed is only weakly true and
   off-by-default for sentiment.

5. **The most direct "does structure survive" number is buried.** The
   structure-preservation Spearman `−0.3904` appears only in prose (`README.MD:182`,
   `:214`). |ρ| ≈ 0.39 is weak-to-moderate neighbourhood preservation — the single
   most direct measure of the thesis — and should be stated as a first-class,
   honestly-hedged result, not a step's side output.

This is a **documentation-only** reconciliation, mirroring `016`. No production code,
CLI, API, or test behaviour changes are required by the Definition of Done. As in
`016`, `.claude/CLAUDE.md` is treated conservatively: only factual edits, with no
rewrite of any mandatory rule, naming convention, or architectural constraint.

**Depends on:** `025-real-retrieval-metrics` and `026-matched-budget-baseline-eval`
(so corrected claims cite real MRR/recall and real matched numbers, not placeholders).

## User Stories

- As a reader, I want "retrieve" to name a **measured** retrieval result (MRR/recall)
  or the number to be labelled **classification accuracy**, so no metric is presented
  as a different one.
- As a skeptic, I want the performance table to show a **matched** color/tfidf/hnsw
  comparison at one budget, so I am not misled by color@400 beside baselines@full.
- As a contributor, I want the corpus described consistently (133 works, not 73), so
  the headline counts agree with the committed gallery and report.
- As a researcher, I want `docs/design.md` to say **Wasserstein-1** and
  **structure-preserving distillation**, matching the code, and to state the
  interpretability axes as *weakly* realised per `020`.
- As a reviewer, I want the structure-preservation `ρ ≈ −0.39` stated as a
  first-class, hedged result so the thesis's most direct evidence is not buried.

## Acceptance Criteria

- [ ] Given `README.MD:17` says the method "retrieves at 81.8%", when this step is
  complete, then either the figure is relabelled as **classification accuracy** or it
  is accompanied by the **real MRR/recall** from `025`/`026`, and no classification
  number is called "retrieval" anywhere in the README TL;DR.
- [ ] Given `README.MD:141-147` juxtaposes color@400 with baselines@full-test, when
  this step is complete, then the "Current Performance" table is the **matched** view
  from `026` (color/tfidf/hnsw at one budget, budget-labelled) and no full-vs-400
  pairing remains.
- [ ] Given `README.MD:29` and `:55` say "73 books" while `:20` says "133-work", when
  this step is complete, then all three agree on the committed count (133), and a
  `grep -nE "73 book" README.MD` returns nothing.
- [ ] Given `docs/design.md:28` says "Wasserstein-2", when this step is complete, then
  it says **Wasserstein-1** consistent with the `ot.emd2` + Euclidean-ground-cost
  implementation.
- [ ] Given `docs/design.md:83` says "random targets", when this step is complete,
  then it describes the **structure-preserving** objective from `002`, and
  `design.md:90`'s interpretability axes are qualified as weakly realised per `020`.
- [ ] Given the structure-preservation `ρ ≈ −0.39` (`README.MD:182`), when this step
  is complete, then it is stated as a first-class, hedged "weak-to-moderate
  neighbourhood preservation" result rather than only a reproduction side-note.
- [ ] Given any residual "retrieve/retrieval" wording that still refers to a
  classification number, when this step is complete, then it is corrected or removed.
- [ ] Given `tox` is run, then all eight quality gates pass and coverage remains 100%
  (documentation edits introduce no uncovered code paths); any optional doc-guard test
  added stays green.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. Code files are read only as the source of truth the docs are reconciled
against (e.g. the Wasserstein calculator's W1 ground cost).

### Application Layer (`src/colors_of_meaning/application/`)

No changes.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No changes.

### Interface Layer (`src/colors_of_meaning/interface/`)

No code changes. `reports/eval_results.md` is regenerated by `026`, not hand-edited
here; this step only points the README table at those committed rows.

### Shared Layer

No changes.

## API Contracts

None. No controller, route, or DTO is added or modified.

## CLI Impact

No CLI behaviour changes. Only README/`docs/design.md` prose (and, conservatively,
two factual lines in `.claude/CLAUDE.md` if the retrieval framing there needs the
same correction) is edited.

## Dependency Injection

No changes.

## Observability

No changes. Documentation edits have no runtime footprint.

## Open Questions

- **Relabel vs report retrieval.** For `README.MD:17`, prefer *reporting the real
  MRR/recall* (from `025`/`026`) alongside classification accuracy, so the "retrieve"
  pillar is finally substantiated rather than merely softened. Fallback: relabel the
  81.8% as classification accuracy. Default: report the real number.
- **A README consistency guard test?** `016` left this as an Open Question; a small
  grep test could assert README contains "133" and not "73 book". Default: add it only
  if it stays trivial and keeps `tox` green; otherwise defer.
- **Keep the aspirational-vs-implemented note style from `016`.** Where a blog claim
  (e.g. the fully-realised interpretability axes) exceeds what `020` measured, mark it
  "aspirational vs measured" rather than deleting, matching `016`'s convention.
- **`.claude/CLAUDE.md` scope.** Its "retrieval" framing (project purpose, baselines
  table) may echo the same relabel; per `016`, edit only factual lines and touch no
  mandatory rule. Default: correct only if a concrete factual claim there is wrong;
  otherwise leave the instruction file alone.
- **Structure-preservation sign/'|ρ|' presentation.** State it as `ρ = −0.3904`
  (minimised toward −1 for perfect preservation, so |ρ| ≈ 0.39) with a one-line
  explanation, so the negative sign is not misread as "anti-preservation".
