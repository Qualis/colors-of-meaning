# Plan: Make the authorship scaling sweep finish

## Implementation Strategy

Four independently-committable stages, ordered by measured payoff and stop-safe at every point.

Stage 1 is 98% of the win and carries all of the correctness risk, so it lands **alone** and is
gated on a differential test before anything else is built on it. Stages 2–4 are each optional:
if Stage 1 alone takes the sweep from 26 days to ~4.4 h, that already makes the generator
runnable and the feature could stop there.

The governing constraint throughout: **no number may change.** Every stage is verified by the
full `tox` at 100% coverage, and Stage 1 additionally by element-wise prediction equality.

## Layer Changes

Domain: one concrete, defaulted method on `DistanceCalculator`. Infrastructure: the Jensen-Shannon
override, the selector rewrite, the sweep's optional pool. Interface: one CLI flag. Application
and shared layers untouched.

### Stage 1 — batch distance API and vectorised Jensen-Shannon (independently committable)

The load-bearing stage. Do the measurement harness **first**, so equivalence is demonstrable
before the fast path exists.

- Add `compute_distance_matrix(queries, references)` to `domain/service/distance_calculator.py`
  as a **concrete** method whose body loops `compute_distance`. Every existing calculator now has
  a correct batch API for free; nothing breaks.
- Override it in `JensenShannonDistanceCalculator` using the union-of-support formulation:
  `JSD(p,q) = (N − |U|)·f(a_p, a_q) + Σ_{i∈U}[ −m_i·ln m_i + ½p_i·ln p_i + ½q_i·ln q_i ]`,
  where `U` is the union of non-floor bins and `a_p` is the document's constant smoothing
  background. Return the **distance** (`sqrt`), matching `scipy.spatial.distance.jensenshannon`.
- Rewrite `_predict` in the selector to consume the matrix and rank with
  `np.argsort(row, kind="stable")`. **This is the single most dangerous line in the feature.**
  `np.argsort` defaults to `kind='quicksort'`, which numpy documents as not stable; with only two
  distinct distance values present, *every* pair is a tie and a default sort would reorder the
  entire neighbour list. Preserve `Counter(...).most_common(1)` for the label tie-break, which
  resolves by first-insertion order and therefore depends on the neighbour ordering above.
- **Acceptance gate:** a differential test that builds a real cap-60 split, takes at least one
  real checkpoint, and asserts the old and new selectors produce identical prediction lists —
  element-wise, not `allclose`. Plus a guard test that fails if the sort stops being stable.
- Record the before/after benchmark (62.7 µs/pair baseline) into the spec.

### Stage 2 — batched encode in the selector (independently committable)

Only measurable once Stage 1 lands; worth ~70 s per training against a then-~11 min baseline.

- Replace the per-document `_encode` loop with a single `embed_batch_to_lab` over all embeddings
  plus a vectorised quantize, so each checkpoint costs one forward instead of 6,598.
- `ColorCodebook.quantize` is currently per-colour `np.argmin`; add a batch path rather than
  looping it. Watch the frozen-dataclass/`cached_property` constraint from spec 011 — do not add
  `__slots__`.
- Verify: predictions unchanged (reuse Stage 1's differential test), and one real timing.

### Stage 3 — optional sweep parallelism (independently committable)

- Execute the `(cap, seed)` grid through `concurrent.futures.ProcessPoolExecutor`, gated on
  `--scaling-workers` with default 1.
- **Pin `torch.set_num_threads(1)` in each worker.** torch currently opens 32 threads to achieve
  ~107% CPU; 8 workers × 32 threads on 8 cores would thrash badly. This is the stage's main trap.
- Reassemble results in deterministic `(cap, seed)` order before `np.mean`/`np.std`, so float
  summation order — and therefore the manifest — cannot depend on completion order.
- Verify: a 2-worker run over a reduced grid produces byte-identical manifest values to a
  1-worker run.

### Stage 4 — expose and document (independently committable)

- `--scaling-workers` on `interface/cli/authorship.py`; optionally surfaced by `bin/generate`.
- Update the cached-manifest notice in `bin/generate`, which currently says the sweep takes
  "hours", once the real figure is known.
- README/ROADMAP note that the manifest now has a runnable generator.

## Dependency Injection

Unchanged. The selector receives its `DistanceCalculator` by constructor injection and continues
to depend only on the domain port; the fast path is selected by which implementation is injected,
not by a flag or an isinstance check.

## Task List

1. [ ] benchmark harness recording µs/pair and projected sweep hours
2. [ ] concrete `compute_distance_matrix` default on the domain port
3. [ ] port-level tests: default matches per-pair `compute_distance` for every calculator
4. [ ] vectorised union-of-support override in `JensenShannonDistanceCalculator`
5. [ ] equivalence tests vs scipy across one-hot, multi-bin, disjoint and identical histograms
6. [ ] selector consumes the matrix; `np.argsort(..., kind="stable")`
7. [ ] guard test failing if the sort is not stable
8. [ ] **differential test: identical predictions on a real cap-60 checkpoint**
9. [ ] record before/after benchmark in the spec
10. [ ] batched encode + vectorised codebook quantize
11. [ ] `ProcessPoolExecutor` over the `(cap, seed)` grid, `torch.set_num_threads(1)` per worker
12. [ ] deterministic result reassembly before aggregation
13. [ ] `--scaling-workers` flag, default 1
14. [ ] update `bin/generate`'s runtime notice with the measured figure
15. [ ] `tox` green at 100% coverage; `shellcheck -x bin/*` clean

## Testing Strategy

Unit tests per layer, one logical assertion each, `test_should_..._when_...`. The vectorised
Jensen-Shannon is tested against `scipy.spatial.distance.jensenshannon` directly — the existing
implementation is the oracle, so equivalence is checkable rather than argued.

Property-based testing earns its place here: the union-of-support formula must equal the dense
scipy result for *arbitrary* non-negative histograms, which is a property, not an example. Cover
the degenerate shapes explicitly — identical documents, disjoint supports, one-hot pairs, and
documents of differing total mass (where the constant-background term stops cancelling and a
naive "outside the union contributes zero" shortcut would be wrong).

No network, no dataset download. The differential test needs a real corpus split, so it must be
marked and skipped when `./documents/` is absent, matching how other documents-dependent tests
behave — CI must stay green on a clean clone.

## Observability Plan

The sweep logs per-training progress with a `correlation_id`, cap, seed and elapsed seconds, so a
long run reports where it is instead of appearing hung — the exact failure that prompted this
spec. Stdlib logger only; the project has no metrics or tracing backend and none is added.

## Risks and Mitigations

| risk | mitigation |
|---|---|
| `np.argsort` default is unstable and every pair ties, silently changing every prediction | `kind="stable"` plus a guard test that fails without it; called out in the SPEC, the PLAN and at the call site's test |
| Vectorised distances differ in the last ulp, turning ties into non-ties and changing rankings | Acceptance is identical *predictions*, not distances within tolerance |
| Union-of-support shortcut assumes equal background mass | Property test includes documents of differing total mass, where `f(a_p, a_q) ≠ 0` |
| 8 workers × 32 torch threads thrash 8 cores | `torch.set_num_threads(1)` per worker; benchmark 1 vs 2 vs 8 before recommending a default |
| Parallel completion order perturbs float aggregation | Reassemble by `(cap, seed)` before aggregating; verified by a 1-worker vs 2-worker manifest comparison |
| The committed manifest may not be reproducible by today's code at all | Open Question 2 — verify one training end to end and treat a mismatch as a finding about the manifest, not a failure of this refactor |
| Optimising the sweep tempts a semantics change alongside it | Semantics are frozen by the SPEC; the degeneracy fix is explicitly deferred to its own spec |

## Validation against the spec

AC1–AC3 are Stage 1 code plus port-level and guard tests. AC4–AC5 are Stage 1's acceptance gate.
AC6 is Stage 2, reusing the same differential test. AC7 is Stage 3, verified by the 1-worker vs
2-worker manifest comparison. AC8 is task 9. AC9 is the standing project gate.
