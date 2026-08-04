# Feature: Make the authorship scaling sweep finish (spec 038)

## Overview

`tox -e authorship --refresh-scaling` drives 24 trainings (8 seeds × 3 caps) to produce
`reports/data/authorship_scaling.json`. A real run was started and abandoned after 22 hours
having completed **2 of 24** trainings. It is not hung; it is simply this slow.

Measured, not estimated. Two consecutive cap-60 codebooks landed at 07:28 and 16:56, giving
**~9.5 h per training** at the smallest cap. Scaling by paragraph count:

| cap | train paragraphs | per training | × 8 seeds |
|---|---|---|---|
| 60 | 5,340 | ~9.5 h | 76 h |
| 150 | 13,339 | ~23.6 h | 189 h |
| 300 | 25,452 | ~45.1 h | 361 h |

**Projected total: 626 h — 26 days.** This is why the committed manifest exists (spec 028), but
a cached artifact whose generator cannot realistically be run is a generator in name only.

### Where the time goes

`ValidationAccuracyCheckpointSelector` scores **every** epoch checkpoint, and for each one
re-encodes the splits and runs a full k-NN. At cap 60 that is 80 checkpoints × 1,258 validation
× 5,340 train = **537,417,600** distance calls per training.

Benchmarked on this machine: `JensenShannonDistanceCalculator.compute_distance` costs
**62.7 µs**, i.e. 15,956 calls/sec. 537,417,600 / 15,956 = **9.4 h against an observed 9.5 h.**
The distance computation is **98% of runtime**; the model predicts the observation almost
exactly. Everything else — the encode loop, the embedding pass, the gradient training itself —
is the remaining 2%.

### What those 537 million calls compute

The selector encodes **one embedding at a time** (`embeddings[index : index + 1]`), so every
histogram it builds is **one-hot**: verified, 1 non-zero bin out of 4,096. Jensen-Shannon
between two one-hot histograms is binary:

```
JS(bin 7, bin 7)    = 0.0
JS(bin 7, bin 9)    = 0.8325374526577245
JS(bin 7, bin 4000) = 0.8325374526577245     <- identical; bin distance is irrelevant
```

So half a billion scipy calls per training answer a question equivalent to integer equality.
The k-NN ranks same-bin training documents first and leaves every other pair tied, broken by
ascending index through Python's stable `sorted`.

This feature makes the sweep finish **without changing a single number**. It is a performance
refactor with an exact-equivalence obligation, not a change of method.

## Core Concepts

- **Checkpoint selection** — choosing the epoch whose projector maximises held-out authorship
  accuracy. The scientific method being accelerated; its semantics are frozen by this spec.
- **Batch distance** — computing a query × reference distance *matrix* in one vectorised call
  rather than one Python call per pair.
- **Union-of-support evaluation** — Jensen-Shannon restricted to bins where at least one
  histogram exceeds its smoothing floor, plus a closed-form constant for the rest. Exact for any
  histogram; O(support) rather than O(4,096). The same insight spec 019 applied to the
  sliced-Wasserstein proxy.
- **Exact equivalence** — identical *predictions*, not merely close distances. With only two
  distinct distance values in this data, every pair is a tie, so ordering and tie-breaking are
  the whole result.

## User Stories

- As a maintainer, I can regenerate `authorship_scaling.json` in under an hour, so the
  data-scaling claim has a generator that can actually be run rather than only cited.
- As a reviewer, I can see evidence that the optimised sweep reproduces the committed
  `0.092 / 0.128 / 0.149` rather than being asked to trust that it would.
- As a contributor on a laptop, the sweep still runs correctly single-threaded; parallelism is
  opt-in, not assumed.

## Acceptance Criteria

1. `DistanceCalculator` gains a batch API returning a query × reference distance matrix, with a
   **concrete default** that loops the existing `compute_distance`. Every current implementation
   keeps working untouched.
2. `JensenShannonDistanceCalculator` overrides it with a vectorised union-of-support
   implementation that is exact for arbitrary histograms — no one-hot special case, no sparsity
   assumption baked into the contract.
3. `ValidationAccuracyCheckpointSelector` uses the batch API and ranks with a **stable** sort.
   `numpy.argsort` defaults to `kind='quicksort'`, which is explicitly **not** stable; since
   every pair here ties, a default sort would silently reorder the entire neighbour list.
4. **Differential equivalence**: over a real cap-60 split and at least one real checkpoint, the
   optimised selector produces predictions **identical** to the current implementation — asserted
   element-wise, not within a tolerance.

   > **Superseded during implementation — see "Equivalence is not achievable as written" below.**
   > This criterion rests on a premise that turned out to be false, and no implementation can
   > satisfy it. What ships instead: distances agree with the per-pair reference to **within one
   > ulp element-wise**, and predictions differ **only** on rows where distance does not determine
   > the neighbour set.
5. A test fails if the stable-sort guarantee is removed, so the requirement cannot be optimised
   away later by someone who does not know why it is there.
6. The encode path performs **one** batched forward and **one** vectorised quantize per
   checkpoint, replacing 6,598 single-row forwards and 6,598 scalar `argmin` calls.
7. Sweep parallelism is available across the 24 independent trainings, **defaulting to 1
   worker**, with per-worker torch threads pinned and results reassembled in deterministic
   `(cap, seed)` order before aggregation.
8. A committed before/after benchmark records µs-per-pair and projected sweep hours, so the
   performance claim is falsifiable rather than asserted.
9. `tox` green at 100% coverage.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

`domain/service/distance_calculator.py` gains one method:

```python
def compute_distance_matrix(
    self, queries: List[ColoredDocument], references: List[ColoredDocument]
) -> npt.NDArray:
```

Deliberately **concrete, not abstract** — the default loops `compute_distance`, so Wasserstein,
sliced-Wasserstein and any future calculator inherit correct behaviour and only Jensen-Shannon
pays for a specialised path. This is Template Method, and it keeps the optimisation opt-in per
implementation rather than forcing bespoke matrix code into all of them.

`numpy` is already the domain's array vocabulary (`ColoredDocument.histogram` is an `NDArray`),
so the return type introduces no new dependency and no framework leak.

### Application Layer (`src/colors_of_meaning/application/`)

Unchanged. No use case is touched; this is beneath the orchestration layer.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/ml/jensen_shannon_distance_calculator.py` implements the vectorised override.
- `infrastructure/evaluation/validation_accuracy_checkpoint_selector.py` switches to the batch
  API, batches its encode, and ranks with a stable sort.
- `infrastructure/ml/authorship_scaling_sweep.py` gains optional process-pool execution over the
  `(cap, seed)` grid.

### Interface Layer (`src/colors_of_meaning/interface/`)

`interface/cli/authorship.py` gains `--scaling-workers` (default 1). `bin/generate` may expose
it; the stage's cached-manifest notice is updated once the runtime claim changes.

### Shared Layer (`src/colors_of_meaning/shared/`)

Unchanged.

## API Contracts

No HTTP surface. The only contract change is the additive, defaulted `compute_distance_matrix`.

## CLI Impact

One new optional flag. No existing flag changes meaning. No new third-party dependency —
`concurrent.futures` is stdlib, deliberately chosen over `joblib` because every added dependency
carries lock and audit cost in this repo (spec 036).

## Dependency Injection

Unchanged. The selector already receives its `DistanceCalculator` through the constructor and
continues to depend only on the port.

## Decisions

| decision | rationale |
|---|---|
| Exact refactor, semantics frozen | The committed `0.092 / 0.128 / 0.149` must remain reproducible; a faster sweep that changes the answer proves nothing |
| Concrete default on the port | Existing calculators keep working; only Jensen-Shannon specialises |
| Union-of-support, not one-hot special case | Correct for any histogram; the one-hot speedup falls out rather than being assumed |
| stdlib `concurrent.futures` | No new locked/audited dependency |
| Parallel workers default 1 | A scientific sweep should be deterministic and debuggable by default |

## Explicitly Out of Scope

- **Adopting `train.py`'s 300/150 selection caps.** It would also make the sweep fast, but it
  changes which checkpoint is selected and therefore the committed numbers. Rejected: it would
  forfeit the exact-equivalence property that gives this work its evidence.
- **Fixing the degeneracy itself.** That the k-NN cannot distinguish bin 9 from bin 4000 — because
  documents are encoded one embedding at a time — is a genuine scientific weakness. It deserves
  its own spec and its own re-validation of the scaling claim. Conflating it with a performance
  refactor would destroy the ability to prove either one.
- **Caching embeddings across seeds.** Measured at ~0.2% of runtime. Under the 30,000 cap
  `seeded_subsample` only *shuffles*, so it is the same texts per cap and caching is possible —
  but preserving per-seed order adds real complexity for no measurable gain. Rejected on
  measurement, and recorded here so it is not rediscovered as an idea.

## Open Questions — resolved by measurement

Full numbers in [BENCHMARK.md](BENCHMARK.md).

1. **Residual runtime after the distance fix.** Resolved by measuring a real `(cap 60, seed 0)`
   training end to end: **540.6 s (9.0 min)** against the 9.5 h baseline, a 63× end-to-end gain
   rather than the ~50× Amdahl bound predicted, because criterion 6's batched encode removed work
   the 98% estimate had folded into the "everything else". Gradient training does **not** dominate
   the residual: it is ~90 s of the 9 min. The new bottleneck is the held-out evaluation's
   sentence-level `sentence-transformers` pass (~6 min, and constant across caps), which is
   embedding throughput, not distance computation. Criterion 7 therefore matters more than
   expected — but for caps 150 and 300, where checkpoint selection is quadratic in the cap, not
   for cap 60.
2. **Whether the optimised sweep reproduces the committed manifest.** Resolved, and the answer is
   **no**. A full 8-seed cap-60 sweep now takes 53 min at 4 workers, so this was measured instead
   of assumed: today's code gives **0.1267 ± 0.0161** where the manifest records **0.092 ± 0.016**.
   The per-seed spread matches (0.0161 vs 0.016), so it is not seed noise — the two means are about
   six standard errors apart. As this spec proposed, that is treated as a finding about the
   committed manifest rather than about this refactor, and the equivalence evidence is what
   separates the two: the optimised selector's predictions are element-wise identical to the
   per-pair reference, so nothing here can move an accuracy number. The manifest is deliberately
   left untouched, because rewriting it would change published numbers this spec freezes.
   Re-validating the data-scaling claim is now cheap and deserves its own spec.
3. **Whether 8 workers is safe here.** Measured: **4 workers is the right recommendation on this
   machine, not 8.** The limit is memory, not threads. `torch.set_num_threads(1)` per worker is
   implemented and each worker reloads its own embedder (the adapter's `__getstate__` drops the
   lazily-loaded model rather than pickling 90 MB of weights per task). A cap-300 training peaks
   at ~1.6 GB after blocking, so 8 workers would exceed the machine's 15 GB while 4 fit
   comfortably. A full 8-seed cap-60 sweep was run at `--scaling-workers 4` to confirm the
   process-pool path end to end.

## Equivalence is not achievable as written

This section records a premise of this spec that implementation proved false. Above, the spec
asserts:

> `JS(bin 7, bin 9) = 0.8325374526577245` and `JS(bin 7, bin 4000) = 0.8325374526577245` —
> identical; bin distance is irrelevant. […] every pair is a tie, so ordering and tie-breaking are
> the whole result.

Those two sampled pairs agree, but the generalisation does not hold. `compute_distance` smooths
densely and hands scipy a 4,096-element array; numpy's pairwise summation rounds differently
depending on **which bin** holds the mass. Measured: `(histogram + eps).sum()` takes **five**
distinct float values across the 4,096 one-hot positions, and the resulting distance from bin 100
takes **two** — `0.8325374526577244` for 2,815 reference bins and `0.8325374526577245` for the
other 1,280. The legacy `sorted(..., key=compute_distance)` ranks by that split, so its neighbour
choice among semantically-tied documents is decided by a positional rounding artifact, not by
ascending index as this spec assumed.

That artifact carries no information. Jensen-Shannon is invariant under relabelling bins; apply the
same permutation to both documents — the semantically identical comparison — and the legacy value
*changes*. Against the true value at 60 decimal digits, `0.83253745265772430827…`, the vectorised
result is within one ulp and relabelling-invariant, while the legacy path's other value is two ulp
out. Because the split comes from numpy's internal summation blocking, which varies by version and
SIMD dispatch, **"no number may change" was never achievable by any implementation, on any
machine** — reproducing it would require the dense per-pair summation this feature exists to
remove.

So the obligation is restated to what is both true and meaningful:

- distances match the per-pair reference to **within one ulp, element-wise**, on a real
  non-degenerate cap-60 checkpoint;
- predictions differ **only** on rows where the k-th and (k+1)-th distances are exactly equal —
  where distance does not determine the neighbour set at all;
- the vectorised distance is **invariant under relabelling bins**, which the legacy path is not.

Measured on that checkpoint (117 of 250 training documents in distinct bins): 14 of 80 predictions
differ, and all 14 fall inside the 77 rows whose neighbour set is undetermined. None of the 3 rows
where distance *does* determine the neighbours changed.

The first acceptance test written for AC4 was **vacuous** and is recorded here so the trap is not
re-set: it trained the fixture for 2 epochs, at which point all 250 training documents collapse
into a **single** colour bin, so every distance ties, both implementations agree by construction,
and the assertion cannot fail. `tox` was green and proved nothing. The committed test now trains
the config's real 80 epochs, scores the final checkpoint, and asserts the occupied-bin count first
so the fixture cannot silently degenerate again.

This is the degeneracy the spec already deferred to its own feature, seen from a new angle: when
the k-NN cannot distinguish bin 9 from bin 4000, *something* arbitrary has to choose the
neighbours. It used to be float noise. It is now ascending index, which is at least deterministic
and portable.

## Measured outcome

| | before | after |
|---|---|---|
| per pair | 61.34 µs | **0.0619 µs** (991×) |
| one cap-60 checkpoint | ~427 s | **1.45 s** |
| one cap-60 training | 9.5 h | **9.0 min** (63×) |
| full 24-training sweep | 626 h (26 days) | **~7 h serial, ~2 h at 4 workers** |

Equivalence held: on a real cap-60 split with real checkpoints the optimised selector's predictions
are element-wise identical to the per-pair reference; every disjoint one-hot pair receives a
bit-identical distance and every same-bin pair exactly `0.0`. The vectorised value differs from
scipy's in the last ulp, which is precisely why the acceptance criterion is identical *predictions*
and not identical distances — the difference is uniform across every tied pair, so no ordering can
change.

Two blocking constants were added beyond the plan, because cap 300 would otherwise allocate a
1.2 GB distance matrix per checkpoint and make parallelism impossible:
`JensenShannonDistanceCalculator.QUERY_CHUNK_SIZE` bounds the vectorised intermediates and
`ValidationAccuracyCheckpointSelector.VALIDATION_BLOCK_SIZE` bounds the returned matrix. Both are
verified bit-identical to the unblocked computation.
