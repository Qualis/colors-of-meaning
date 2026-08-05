# Benchmark: authorship scaling sweep, before and after (spec 038)

All figures measured on the development machine used for the abandoned 22-hour run: 8 cores,
15 GB RAM, CPU-only torch, numpy 2.4.6, scipy 1.17.1, Python 3.11.2. Reproduce the per-pair
comparison with:

```bash
tox -- -m benchmark tests/colors_of_meaning/infrastructure/ml/test_jensen_shannon_distance_calculator.py
```

The benchmark test asserts only a conservative ≥20× floor so it cannot flake on a loaded CI box;
the numbers below are what it actually produces here.

## Per-pair Jensen-Shannon distance

Realistic selector workload: 4,096-bin histograms, one non-zero bin each (the selector encodes one
embedding at a time, so every histogram it builds is one-hot).

| | µs per pair | pairs/sec |
|---|---|---|
| `compute_distance` (scipy, per pair) | **61.34** | 16,300 |
| `compute_distance_matrix` (vectorised, cap-60 shape) | **0.0619** | 16,200,000 |
| `compute_distance_matrix` (vectorised, cap-300 shape) | **0.0913** | 11,000,000 |

**991× faster per pair** at cap-60 shape. The spec's independently measured baseline was 62.7 µs;
61.34 µs reproduces it.

## One checkpoint's validation scoring, cap 60

1,258 validation × 5,340 train = 6,717,720 pairs.

| | seconds |
|---|---|
| distances only, before | 412 (6,717,720 × 61.34 µs) |
| distances only, after | **0.42** |
| whole checkpoint incl. encode, before | ~427 (observed 9.5 h ÷ 80 checkpoints) |
| whole checkpoint incl. encode, after | **1.45** |

## One training, cap 60, seed 0 — end to end

Real corpus, real embedder, 80 epochs, 80 scored checkpoints, held-out evaluation included.

| | duration |
|---|---|
| before (measured: two consecutive cap-60 codebooks landed 07:28 and 16:56) | **9.5 h** |
| after (measured end to end) | **540.6 s = 9.0 min** |

**63× faster per training.** The residual splits roughly as ~50 s embedding + ~90 s gradient
training + ~85 s checkpoint selection + ~6 min held-out evaluation, so checkpoint selection is no
longer the bottleneck — the held-out evaluation's sentence-level embedding pass now is, and it is
`sentence-transformers` throughput rather than distance computation.

## Full 24-training sweep

Cap 150 and cap 300 scale *quadratically* in the checkpoint-selection step, because both the
validation and the train split grow with the cap. Cap 60 is measured; caps 150 and 300 are projected
from the measured distance cost at their shapes plus the measured constant held-out evaluation.

| cap | train paragraphs | per training before | per training after | × 8 seeds after |
|---|---|---|---|---|
| 60 | 5,340 | 9.5 h | 9.0 min | 1.2 h |
| 150 | 13,339 | ~23.6 h | ~13 min | ~1.7 h |
| 300 | 25,452 | ~45.1 h | ~29 min | ~3.9 h |

| | total |
|---|---|
| before (spec's projection) | **626 h — 26 days** |
| after, `--scaling-workers 1` | **~7 h** (projected from the measured cap-60 training) |
| after, `--scaling-workers 4` | **under 7 h, but not 7/4** — see below |

The 4-worker path was run for real, not only projected: a full 8-seed cap-60 sweep completed in
**3,185.9 s (53 min)** as two waves of four workers. Ideal 4× scaling of the measured 9.0 min
training would have been 18 min, so **parallelism delivered about 1.4× here, not 4×** — the machine
was simultaneously running `tox` at load ~20, and cap 300 is memory-bound besides (~1.6 GB per
worker). Do not read "~7 h serial" and divide by the worker count. What the run does establish is
that the process pool, the `__getstate__` that keeps 90 MB of embedder weights out of every pickled
task, and the deterministic `(cap, seed)` reassembly all work end to end.

Only the cap-60 training and this 8-seed cap-60 sweep are measured. Every other row in the table
above is projected from them, and the projections are unvalidated at caps 150 and 300.

## Reproducing the committed manifest — it did not, so it was re-run

> **Resolved after this spec landed.** The full 24-training sweep was run on this code and
> `reports/data/authorship_scaling.json` was rewritten from its output:
>
> | cap | was | now |
> |---|---|---|
> | 60 | 0.092 ± 0.016 | **0.1267 ± 0.0161** |
> | 150 | 0.128 ± 0.022 | **0.1407 ± 0.0118** |
> | 300 | 0.149 ± 0.023 | **0.1833 ± 0.0178** |
>
> The cap-60 row is bit-identical to the 8-seed measurement recorded below, so the sweep reproduces
> itself.
>
> "More data helps" survives — the new curve is monotone at 2.79× → 3.10× → 4.03× chance — but it
> did **not** strengthen. Comparing old and new at a *fixed* cap (2.02× → 2.79× at cap 60) measures
> the level shift between two manifests, not the slope of the scaling curve. Read within each
> manifest, the trend is flat-to-weaker: the absolute gain across the range is unchanged (+0.0570
> old, +0.0566 new), the relative gain falls from 1.62× to 1.45×, and the shape moved — the cap
> 60 → 150 step collapsed from +0.036 to +0.014, inside the per-seed spread, while 150 → 300 grew
> from +0.021 to +0.043. This vindicates the caution recorded further down that "the *shape* of
> the 'more data helps' claim may be affected, not merely its offset".
>
> The rest of this section is the original finding, kept because it is what motivated the re-run.

### The original finding

Spec 038's Open Question 2 asked whether today's code reproduces
`reports/data/authorship_scaling.json`, whose cap-60 row was produced by an unknown historical run.
Now that the sweep is runnable, this was answered rather than assumed: a full 8-seed cap-60 sweep
was run at `--scaling-workers 4`.

| | mean accuracy | std | seeds |
|---|---|---|---|
| committed manifest, cap 60 | 0.092 | 0.016 | 8 |
| this code, cap 60, same config | **0.1267** | 0.0161 | 8 |

The per-seed spread matches almost exactly (0.0161 vs 0.016), so this is not seed noise: the
standard error of the mean is 0.016/√8 ≈ 0.006, putting the two means about six standard errors
apart. Today's code scores **higher** than the manifest claims, so the committed data-scaling
narrative is conservative rather than inflated — but note that 0.1267 at cap 60 is close to the
manifest's cap-150 figure of 0.128, so the *shape* of the "more data helps" claim may be affected,
not merely its offset.

**How much of this gap is the refactor?** Honestly: not separable from this run alone, and it would
cost ~76 h of legacy-path compute to separate. What can be said is bounded. The refactor changes
predictions only on rows where distance does not determine the neighbour set — 14 of 80 on the
measured checkpoint — so it can move an accuracy number, and the previous section explains why no
implementation could have avoided that. But a six-standard-error gap is far larger than a
tie-break rule plausibly accounts for, and other candidates are known and larger: the corpus grew
from 73 works to 133 in spec 027, and the supervised mapper and split have both changed since the
manifest was written. Identifying the cause needs a bisect, not a benchmark, and that is a spec of
its own.

**The manifest was left untouched by spec 038 itself**, because rewriting it changes published
numbers that this spec freezes. It was re-run immediately afterwards as a separate, deliberate step —
see the note at the top of this section. That is the whole point of the feature: re-validating the
data-scaling claim went from 26 days to ~2 h at 4 workers, so a stale manifest became something you
fix by running the generator rather than something you document as a known defect.

## Memory

The optimisation trades memory for time, so the sweep's per-worker footprint is worth stating
before anyone raises `--scaling-workers`:

| stage | peak RSS |
|---|---|
| cap-60 training | ~1.4 GB |
| cap-300 distance matrix, unblocked | 2.66 GB |
| cap-300 distance matrix, blocked (shipped) | ~1.6 GB |

Two blocking constants bound this. `JensenShannonDistanceCalculator.QUERY_CHUNK_SIZE` (512) bounds
the vectorised intermediates; `ValidationAccuracyCheckpointSelector.VALIDATION_BLOCK_SIZE` (1024)
bounds the returned matrix. Both are verified to be bit-identical to the unblocked computation —
blocking partitions rows and no row's value depends on another's. On 15 GB, 4 workers is
comfortable at every cap and 8 workers is not, at cap 300.

## Equivalence — what is actually true

The spec set out to prove element-wise identical predictions. That turned out to be impossible for
*any* implementation, and the reason is worth stating precisely, because it is a fact about the
old code rather than the new.

`compute_distance` smooths densely and hands scipy 4,096 elements; numpy's pairwise summation
rounds differently depending on which bin holds the mass:

| | measured |
|---|---|
| distinct values of `(histogram + eps).sum()` over 4,096 one-hot positions | **5** |
| distinct legacy distances from bin 100 to the other 4,095 bins | **2** (2,815 × `…244`, 1,280 × `…245`) |
| distinct vectorised distances, same comparison | **1** |
| true value at 60 decimal digits | `0.83253745265772430827…` |

The legacy k-NN ranks by that 1-ulp split, so its neighbour choice among semantically-tied
documents is a positional rounding artifact. The artifact is provably meaningless: Jensen-Shannon
is invariant under relabelling bins, yet applying the same permutation to both documents changes
the legacy value (2 of 6 random trials). The vectorised value is invariant, and is the one within
one ulp of truth; legacy's other value is two ulp out. Since the split originates in numpy's
internal summation blocking — version- and SIMD-dependent — the manifest that was committed *at the
time this was written* (`0.092 / 0.128 / 0.149`) was never bit-reproducible in principle. The
manifest committed now was produced by this code and does reproduce; see the top of the previous
section.

What is asserted instead, all committed as tests:

- distances match the per-pair reference **to within one ulp, element-wise**, on a real
  non-degenerate cap-60 checkpoint (117 of 250 training documents in distinct bins);
- predictions differ **only** where the k-th and (k+1)-th distances are exactly equal — measured
  as **14 of 80** rows differing, every one of them inside the 77 rows whose neighbour set is
  undetermined, and **none** of the 3 rows where distance does determine the neighbours;
- the vectorised distance is **relabelling-invariant** (exactly, for one-hot documents; to within
  one ulp for arbitrary ones), which the legacy path is not;
- every same-bin pair receives **exactly** `0.0`.

`numpy.argsort` defaults to an unstable sort, and since the mathematically-tied pairs really are
tied under the exact computation, the default would silently reorder the entire neighbour list. The
selector pins `kind="stable"`, and a committed guard test fails when that is removed — confirmed by
mutating the call site and watching the test go red.

### A vacuous test, recorded so the trap is not re-set

The first acceptance test written for this trained its fixture for 2 epochs. At 2 epochs all 250
training documents quantize into a **single** colour bin, so every distance ties, both
implementations agree by construction, and the assertion cannot fail. It passed; `tox` was green;
it proved nothing. The committed test now trains the config's real 80 epochs, scores the final
checkpoint, and asserts the occupied-bin count **first**, so a fixture that silently degenerates
fails loudly instead of certifying itself.
