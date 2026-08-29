# Feature: Align the projector objective with the structure metric it is scored on

## Overview

The committed headline for structure preservation is **ρ = −0.3904**, which the
README honestly calls "weak-to-moderate ... real structure survives the 12-bit
color, and a meaningful share is lost". The repository attributes that loss to the
**384→3 bottleneck**. Reading the training code against the evaluation code shows
that attribution has never been tested, because the two measure different things:

- **Training** (`infrastructure/ml/pytorch_color_mapper.py:146-176`) minimises MSE
  between the embedding cosine matrix and the cosine of **batch-mean-centred Lab**
  outputs.
- **Checkpoint selection and the reported number**
  (`infrastructure/evaluation/structure_preservation_evaluator.py:38-55`) compute
  Spearman between embedding cosine and **Euclidean ΔE**.

The gradient never sees the quantity that is selected on and published. Three
defects follow from the student term specifically:

1. **Angle discards magnitude.** Cosine scores two Lab vectors on the same ray from
   the batch mean as identical; a pale and a saturated variant of one hue can sit
   ~60 ΔE apart and cost nothing. The loss is blind to perceptual distance — the
   axis the thesis rests on.
2. **Batch-mean-centring is non-stationary.** `lab_output - lab_output.mean(dim=0)`
   makes a pair's target depend on which other samples share its batch.
3. **Range mismatch.** Teacher cosines are positively skewed and compressed;
   student cosine over centred 3-D vectors spans [−1, 1] broadly. MSE across those
   supports pulls the student toward the teacher's *mean* rather than its
   *structure*.

This feature replaces the objective with one that optimises the reported quantity,
and — critically — **bounds what three dimensions can hold at all**, so the residual
loss can finally be attributed to the bottleneck or to the objective rather than
assumed. It closes roadmap item **R-2** (unconstrained target space) as a control arm.

A prerequisite phase also resolves a second unexplained number. `reports/rate_distortion.md`
records accuracy 0.6450 @3b → 0.7150 @6b → **0.7600 @9b → 0.6300 @12b**: the headline
operating point is the worst above 3 bits. The sweep ran `--distance jensen_shannon`,
which **saturates on disjoint supports**; an AG News item is 1–3 sentences, so at
4,096 bins its histogram is ~3-hot and pairs share almost no bins. The main
evaluation reaches 81.75% at the same 12 bits using sliced Wasserstein, which reads
the Lab support geometry instead. The inversion is therefore plausibly a metric
artifact, not a property of the bit budget, and it must be settled before it is used
to interpret anything else.

## Core Domain Concepts

- **Structure objective**: the training criterion relating embedding dissimilarity
  to Lab dissimilarity. A swappable arm, not a fixed loss.
- **Objective arm**: one named (objective, head) configuration trained across seeds
  and scored on held-out structure preservation and downstream task metrics.
- **Ceiling control**: a non-arm reference bounding achievable structure at three
  dimensions — an untrained noise projector (floor), a PCA-3 linear projection
  (untrained reference), and an unconstrained-head variant (no Lab gamut clamp).
- **Pre-registered adoption rule**: the seed-spread threshold, fixed before the run,
  that decides whether a new arm replaces the committed projector.

## User Stories

- As a researcher, I want the projector trained on the metric it is judged by, so
  ρ measures the bottleneck's cost rather than an objective/metric mismatch.
- As a skeptic, I want a measured ceiling for three dimensions, so "|ρ| ≈ 0.39 is
  what 12 bits costs" is a demonstrated claim rather than an assumption.
- As a researcher, I want the 9→12-bit accuracy inversion explained, so the headline
  operating point is not quietly the worst one on the frontier.
- As a maintainer, I want every arm's result committed with seed error bars, so a
  ρ improvement can be distinguished from seed noise.
- As a reviewer, I want the adoption rule fixed in advance, so a negative result is
  reported rather than re-cut until the new objective wins.

## Acceptance Criteria

- A `StructureObjective` seam exists in `infrastructure/ml/`; `PyTorchColorMapper`
  receives one by injection and defaults to today's behaviour, so existing numbers
  stay reproducible.
- Arms implemented and measured: `cosine_centred` (today, baseline), `delta_e_correlation`
  (Pearson over standardised off-diagonal ΔE against 1 − cosine), `margin_ranking`
  (`nn.MarginRankingLoss` over teacher-ordered triplets).
- Controls measured: `noise` (floor), `pca3` (PCA-3 fitted on train, each axis
  min-max rescaled to the Lab ranges so it quantizes against the same codebook),
  `unconstrained_head` (Lab sigmoid/tanh head removed — roadmap R-2).
- `unconstrained_head` reports ρ **twice**: pre-clamp (isolating the gamut
  constraint) and post-clamp (what the pipeline would actually receive).
- Held-out Spearman ρ is reported per arm as **mean ± sd over 8 seeds**.
- Downstream AG News accuracy, macro F1, MRR and recall@5 at the matched 4,000
  budget are reported for the top-2 objective arms over 3 seeds.
- **Pre-registered adoption rule**: an arm replaces the committed projector only if
  its mean held-out |ρ| exceeds `cosine_centred`'s by more than **2× the pooled seed
  standard deviation** *and* its AG News accuracy is no more than **1.0 point** below
  it. Otherwise the committed artifact is unchanged and the negative result is
  published.
- A committed `reports/structure_objective.md` is regenerated by a committed command.
- Phase 0 re-runs the rate-accuracy axis under both `jensen_shannon` and `sliced`
  over 3 seeds and records whether the 9→12-bit inversion persists; the finding is
  written into `reports/rate_distortion.md` as a correction or a footnote.
- `tox` green, 100% coverage, one logical assertion per test, no comments, layer
  boundaries respected (no `torch` in `domain/`).

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/model/objective_comparison.py` → frozen `ObjectiveArmResult(arm, mean_rho,
  stdev_rho, seeds, accuracy, macro_f1, mrr, recall_at_k)` and
  `ObjectiveComparison(results, baseline_arm, adoption_threshold_sigma)` exposing
  `adopted_arm()` applying the pre-registered rule. Pure — no torch, no sklearn.
- No new port: the objective is a torch-level concern and a `domain/service` ABC
  over tensors would import torch into the domain, which the Definition of Done
  forbids.

### Application Layer (`src/colors_of_meaning/application/`)

- `CompareStructureObjectivesUseCase(mapper_factory, structure_preservation_evaluator,
  evaluate_use_case_factory=None)` → `execute(train_embeddings, eval_embeddings, arms,
  seeds) -> ObjectiveComparison`. Trains each arm at each seed, scores held-out ρ,
  optionally runs the downstream evaluation for the nominated arms, aggregates.
  `correlation-id` logging per arm and seed.
- Reuses `TrainColorMappingUseCase` for the train/select loop and `EvaluateUseCase`
  for the task axis; adds no new training logic.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/ml/structure_objectives.py` → the three objective callables plus a
  `StructureObjective` `Protocol`. `cosine_centred` is today's code moved verbatim so
  the baseline arm is provably unchanged.
- `infrastructure/ml/pytorch_color_mapper.py` → `__init__` accepts
  `structure_objective`, defaulting to `cosine_centred`; `LabProjectorNetwork` accepts
  `constrain_to_lab: bool = True` to expose the unconstrained head.
- `infrastructure/evaluation/pca_projection_control.py` → PCA-3 control producing
  Lab-ranged coordinates from train-split statistics.
- `infrastructure/visualization/matplotlib_figure_renderer.py` → `render_objective_comparison`
  (arms on x, mean ρ with sd error bars).

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/compare_objectives.py` → `--arms`, `--controls`, `--seeds`,
  `--downstream-arms`, `--dataset`, `--config`, `--budget`, `--output-path`
  (default `reports/structure_objective.md`), `--figure-path`. Writes the report.
- No API change; no existing CLI changes.

### Shared Layer

- None. Seeding reuses `shared/determinism.seed_everything`.

## API Contracts

No endpoint changes.

## CLI Impact

One new CLI (`compare_objectives`) and one new `tox -e compare_objectives`. The Phase 0
diagnosis needs **no code change at all** — it is the existing `tox -e rate_distortion`
invoked once per `--distance` value. Accepting repeated `--distance` values, and adding
`sliced` to that CLI, are conveniences for making the corrected report reproducible from
a single command.

## Dependency Injection

The CLI constructs the embedding adapter, codebook, evaluator, renderer and a mapper
factory closing over each arm's objective, then injects them into the use case —
matching the construction style of `eval_suite` and `ablate`. No new third-party
dependency: `torch`, `scikit-learn`, `scipy` and `matplotlib` are already declared.

## Observability

Per-arm and per-seed `correlation-id` log lines carrying arm name, seed, held-out ρ,
and elapsed seconds; one summary line carrying the adopted arm and the margin over
baseline in units of pooled sd.

## Open Questions

- **Pearson vs soft-Spearman.** Default: train on Pearson over standardised
  off-diagonals as a monotone surrogate and report Spearman, noting the gap. A
  differentiable soft-rank would close it exactly but needs a new dependency
  (`torchsort`), which the pinned-torch supply chain makes costly. Deferred.
- **Seed count.** Default 8, matching the authorship scaling sweep, so the sd is
  comparable to the one already published.
- **Triplet sampling for `margin_ranking`.** Default: all in-batch triplets whose
  teacher cosine gap exceeds a margin threshold; hard-negative mining is a future
  option.
- **Should `pca3` also report downstream accuracy?** It quantizes against the same
  codebook so it can. Default: ρ only, to keep the overnight budget bounded.
- **Do the supervised and structured mappers move too?** They own separate losses
  (`_contrastive_loss`; hue/lightness/chroma axes) and are out of scope here, which
  is why `reports/interpretability.md` does not regenerate.
- **Adoption rule strictness.** 2σ on ρ plus a 1.0-point accuracy guard is a
  judgement call; a stricter 3σ would risk discarding a real but modest gain at 8
  seeds.
