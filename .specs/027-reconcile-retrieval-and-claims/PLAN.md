# Plan: Reconcile claims (round 2) — retrieval-vs-classification wording, stale counts, and design.md drift

## Implementation Strategy

A documentation-only pass, in the mould of `016-p2-6-reconcile-docs`: make the prose
match the code and the committed numbers, now that `025-real-retrieval-metrics` and
`026-matched-budget-baseline-eval` have produced real MRR/recall and a matched
color/tfidf/hnsw comparison. Every edit is a factual correction with a `file:line`
anchor and a source-of-truth reference; nothing is softened vaguely and no metric is
left named as a different metric.

Order the edits by the source of truth each depends on:

1. **README table → matched view** (needs `026`'s refreshed
   `reports/eval_results.md`): replace the "Current Performance" block
   (`README.MD:141-147`, `:149`) with the matched color/tfidf/hnsw rows at one
   budget, budget-labelled, and delete the color@400-vs-baseline@full pairing.
2. **"Retrieves at 81.8%" → measured retrieval or relabelled** (needs `025`): at
   `README.MD:17` (and the `:13`/`:38` "retrieve" framing), attach "retrieve" to the
   real MRR/recall, or relabel `81.8%` as classification accuracy. Default: report the
   real retrieval number.
3. **Corpus count 73 → 133** (self-contained): fix `README.MD:29` and `:55` to agree
   with `:20` and `reports/documents_authorship.md`.
4. **design.md corrections** (self-contained, verified against code): `:28`
   Wasserstein-2 → Wasserstein-1; `:83` random targets → structure-preserving
   distillation (`002`); `:90` interpretability axes qualified as weakly realised
   (`020`).
5. **Structure-preservation ρ promoted** (self-contained): state `ρ = −0.3904`
   (`README.MD:182`, `:214`) as a first-class hedged result with a one-line sign
   explanation.

## Layer Changes

Documentation only — no `src/` layer changes. Files touched:

### `README.MD`
- `:141-147`, `:149` — swap the performance table for `026`'s matched rows; remove the
  budget-mismatch pairing; keep a pointer to `reports/eval_results.md`.
- `:17` (+ `:13`, `:38`) — "retrieves at 81.8%" becomes measured retrieval
  (MRR/recall from `025`) or is relabelled classification accuracy; the "retrieve"
  pillar wording is made truthful.
- `:29`, `:55` — "73 books" → "133 works" (matches `:20`).
- `:182`, `:214` — present `ρ = −0.3904` as a first-class, hedged
  structure-preservation result with a sign note.

### `docs/design.md`
- `:28` — "Wasserstein-2 distance" → "Wasserstein-1 distance" (Euclidean ground cost +
  `ot.emd2`, no final square root).
- `:83` — "random targets (unsupervised)" → structure-preserving similarity
  distillation (`002`).
- `:90` — interpretability axes marked weakly realised / off-by-default for sentiment,
  per `020`.

### `.claude/CLAUDE.md` (conservative, only if needed)
- Correct only a concrete factual claim that echoes the retrieval relabel; touch no
  mandatory rule, naming convention, or architectural constraint (per `016`).

## Dependency Injection

No changes.

## Task List

1. [ ] README: replace the "Current Performance" table with `026`'s matched
   color/tfidf/hnsw rows; remove the color@400-vs-baseline@full pairing; keep the
   `reports/eval_results.md` pointer.
2. [ ] README: correct the "retrieves at 81.8%" wording — report real MRR/recall from
   `025`, or relabel as classification accuracy; fix the `:13`/`:38` retrieve framing.
3. [ ] README: change "73 books" (`:29`, `:55`) to 133 works; confirm agreement with
   `:20` and `reports/documents_authorship.md`.
4. [ ] README: promote `ρ = −0.3904` to a first-class hedged result with a sign note.
5. [ ] design.md: Wasserstein-2 → Wasserstein-1; random targets → structure-preserving
   distillation; qualify the interpretability axes per `020`.
6. [ ] (optional) add a trivial README-grep guard test (contains "133", not "73
   book") only if it stays simple and keeps `tox` green.
7. [ ] run `tox`; confirm all 8 gates + 100% coverage; run
   `grep -nE "73 book|Wasserstein-2|random targets" README.MD docs/design.md` and
   confirm no stale claim remains.

## Testing Strategy

Documentation edits add no code paths, so coverage is unaffected. House rules still
apply to any optional guard test: one logical assertion, `test_should_..._when_...`,
no network. Verification is primarily by grep-based checks in the Task List:

- No "73 book" remains in `README.MD`; "133" is present and consistent with `:20`.
- No "Wasserstein-2" or "random targets" remains in `docs/design.md`.
- No README figure labelled "retrieve"/"retrieval" refers to a classification-only
  number; retrieval wording maps to an MRR/recall value or is relabelled.
- The performance table's rows all share one budget (no full-vs-400 mix).

## Observability Plan

None. Documentation has no runtime footprint.

## Risks and Mitigations

- **Docs re-drift as numbers change.** Mitigation: point the README table at the
  committed `reports/eval_results.md` rather than transcribing values; optionally add
  the grep guard test.
- **Over-editing `.claude/CLAUDE.md`.** Mitigation: as in `016`, restrict to factual
  corrections; leave all mandatory rules and conventions untouched.
- **Relabelling hides the "retrieve" pillar instead of substantiating it.**
  Mitigation: default to *reporting the real MRR/recall* from `025`/`026`, so the
  pillar is measured, not merely softened.
- **Editing ahead of the numbers.** Mitigation: this step depends on `025` and `026`
  landing; if sequenced earlier, the table/retrieval edits wait for their committed
  outputs and only the self-contained fixes (73→133, design.md, ρ) proceed.
