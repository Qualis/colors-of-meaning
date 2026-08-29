# Plan: Align the projector objective with the structure metric it is scored on

## Implementation Strategy

Turn the training criterion into an **injected seam**, then run a pre-registered A/B
across objective arms and ceiling controls, and let the committed decision rule — not
the result — decide whether the shipped projector changes.

Three decisions keep this honest and affordable:

1. **The baseline arm is today's code, moved verbatim.** `cosine_centred` is a
   cut-and-paste of `_structure_loss` into the new module and stays the default, so
   every existing committed number remains reproducible and the A/B has a real
   control rather than a re-implementation.
2. **Cheap metric for all arms, expensive metric for finalists.** Held-out ρ needs
   only pairwise arithmetic over evaluation embeddings, so all arms × 8 seeds is
   minutes. Downstream accuracy costs ~610 s per AG News color row, so it runs for
   the top-2 arms × 3 seeds only. This is what makes the sweep fit one overnight run.
3. **Controls before conclusions.** `noise` (floor), `pca3` (untrained linear
   reference) and `unconstrained_head` (no Lab gamut) bound what three dimensions can
   hold. Without them a ρ of 0.45 is uninterpretable; with them the residual is
   attributable.

Phase 0 runs first and independently, because its outcome decides which distance the
downstream arm should use and whether `reports/rate_distortion.md` needs a correction.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/model/objective_comparison.py`: frozen `ObjectiveArmResult` and
  `ObjectiveComparison`; `adopted_arm()` implements the 2σ-plus-accuracy-guard rule.
  Validates seed counts match and sd is non-negative. No torch, no sklearn.
- Tests: adoption returns the baseline when the margin is under threshold; returns the
  challenger when over; rejects a challenger that clears ρ but fails the accuracy guard.

### Application Layer (`src/colors_of_meaning/application/`)

- `application/use_case/compare_structure_objectives_use_case.py`:
  `CompareStructureObjectivesUseCase.execute(train_embeddings, eval_embeddings, arms,
  seeds, downstream_arms)`. Per (arm, seed): build mapper via the injected factory,
  delegate training to `TrainColorMappingUseCase`, score with the injected
  `StructurePreservationEvaluator`, collect. Aggregate to mean ± sd, then run the
  downstream evaluation for the nominated arms. `correlation-id` logging throughout.
- No new training logic here — it orchestrates existing use cases.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/ml/structure_objectives.py`:
  - `StructureObjective` `Protocol` — `(lab_output, teacher_embeddings) -> Tensor`.
  - `cosine_centred` — today's loss, moved unchanged.
  - `delta_e_correlation` — pairwise Euclidean Lab distance against `1 - cos(E)`,
    both off-diagonals standardised, loss `1 - pearson`.
  - `margin_ranking` — in-batch triplets ordered by teacher cosine, scored with
    `nn.MarginRankingLoss` on ΔE.
  - `offdiagonal_entries` moves here; `pytorch_color_mapper` and
    `supervised_pytorch_color_mapper` import it from the new home.
- `infrastructure/ml/pytorch_color_mapper.py`: `__init__(..., structure_objective=cosine_centred)`;
  `_structure_loss` delegates. `LabProjectorNetwork(..., constrain_to_lab: bool = True)`
  returns raw three-dim output when false.
- `infrastructure/evaluation/pca_projection_control.py`: fit `PCA(n_components=3)` on
  train embeddings, transform eval embeddings, min-max rescale each axis to
  L∈[0,100] and a,b∈[−127.5,127.5] using train statistics, return `List[LabColor]`.
- `infrastructure/visualization/matplotlib_figure_renderer.py`: `render_objective_comparison`
  — arms on x, mean ρ with sd error bars, controls drawn as horizontal reference lines.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/compare_objectives.py`: argument parsing, artifact loading, container
  construction, report writing. Follows `ablate.py`'s factory-closure style for the
  per-arm mapper factory.
- `tox.ini`: `[testenv:compare_objectives]`.
- `interface/cli/rate_distortion.py`: accept repeated `--distance` so Phase 0 needs no
  new environment.

### Shared Layer (`src/colors_of_meaning/shared/`)

- Unchanged. Seeding via `shared/determinism.seed_everything`.

## Dependency Injection

The CLI builds a `mapper_factory: Callable[[str, int], ColorMapper]` closing over the
arm's objective and seed, plus the evaluator, renderer, codebook and (for finalists) an
`evaluate_use_case_factory`. These are injected into the use case; no Lagom container or
API wiring changes, matching `ablate` and `eval_suite`.

## Task List

**Phase 0 — settle the 9→12-bit inversion (no algorithm change, no code change)**
1. Re-run the rate-accuracy axis under `jensen_shannon` and exact `wasserstein`,
   3 seeds, at the existing `--max-samples 200`. Both are constructible today
   (`rate_distortion.py:126-133`); `sliced` is not, and is not needed — exact EMD is the
   stronger test because it carries no proxy confound.
2. Write the finding into `reports/rate_distortion.md` — correction if the inversion
   is metric-induced, a measured sparsity note if it survives.
3. Only if the report needs both metrics in one committed command: accept repeated
   `--distance`, and optionally wire `sliced` into this CLI, with tests.

**Phase 1 — the seam (no behaviour change)**
4. `structure_objectives.py` with `cosine_centred` moved verbatim; relocate
   `offdiagonal_entries` and update both importers.
5. Inject `structure_objective` into `PyTorchColorMapper`; add `constrain_to_lab`.
6. Regression test: default construction reproduces the current loss value bit-for-bit
   on a fixed batch.

**Phase 2 — arms, controls and the comparison**
7. Implement `delta_e_correlation` and `margin_ranking`.
8. `ObjectiveComparison` domain model and the adoption rule.
9. `PcaProjectionControl`; wire `noise` and `unconstrained_head` (pre- and post-clamp).
10. `CompareStructureObjectivesUseCase`.
11. `render_objective_comparison`.
12. `compare_objectives` CLI and tox environment.

**Phase 3 — run and report**
13. Overnight run: all arms and controls × 8 seeds on ρ; top-2 × 3 seeds downstream.
14. Write `reports/structure_objective.md` and its figure.
15. Apply the adoption rule. If a challenger wins, retrain and recommit
    `artifacts/models/projector.pth` (plus the documents variants) and regenerate via
    `./bin/generate --retrain`; if not, commit the negative result and change no artifact.
16. Reconcile `README.md` — the ρ figure, the results table, and the bottleneck-vs-objective
    attribution sentence.

## Testing Strategy

- Domain: adoption-rule cases (under threshold, over threshold, accuracy-guard veto).
- Objectives: each arm returns a scalar, is finite, and decreases when the student
  ordering is made to agree with the teacher's; `cosine_centred` matches the archived
  reference value.
- `delta_e_correlation` is invariant to a uniform scale on the Lab output (the property
  that fixes the range mismatch) — a test the old objective fails.
- Control: `pca3` output lies inside the Lab ranges and quantizes without raising.
- Unconstrained head: output escapes the Lab ranges pre-clamp (proving the head is off).
- Use case: seeds are honoured (same seed → same ρ), and the downstream evaluation runs
  only for nominated arms (assert via a spy factory).
- CLI: report contains one row per arm and the adopted-arm line.
- Follow the stochastic-RNG discipline — no bare `torch.manual_seed` leaking under
  `--random-order`.

## Observability Plan

`correlation-id` on every arm/seed training line (arm, seed, rho, seconds) and on the
summary (adopted arm, margin in pooled sd, baseline rho). Report records library
versions, as the other reports do.

## Risks and Mitigations

- **Overnight budget overrun.** Mitigation: ρ-only for all arms; downstream limited to
  top-2 × 3 seeds; Phase 0 capped at the existing `--max-samples 200`. Exact EMD costs
  ~92 ms per call, so Phase 0's Wasserstein pass is the expensive half — budget roughly
  an hour or two for four budget points at that sample cap, and confirm before scaling.
- **The new objective wins on ρ but loses on accuracy.** This is a plausible and
  interesting outcome, not a failure — the accuracy guard in the adoption rule exists
  precisely for it, and the report states both axes.
- **Large regeneration diff.** Mitigation: `./bin/generate` is stage-scoped and
  byte-reproducible apart from a matplotlib PNG version stamp; regenerate stage by
  stage. Note `interpretability.md` is **not** affected (structured mapper owns its
  own axis losses), and `documents-*` stages plus `book` are local-only (git-ignored
  corpus, live API key).
- **Coverage regressions from the moved code.** Mitigation: clear the stale non-editable
  tox install and egg-info before trusting a local coverage number.
