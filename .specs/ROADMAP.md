# Colors of Meaning — Remediation & Delivery Roadmap

**Last updated:** 2026-07-17
**Status:** Delivered — P0–P2 complete (specs 001–016); 11 extensions shipped (017–027); only the §7 research extensions remain
**Scope:** Sequence the changes needed to turn the repository into a valid empirical test of the [Colors of Meaning](https://www.qual.is/posts/colors-of-meaning) thesis, then extend it.

---

## 1. Why this roadmap exists

The blog post is conceptual and reports **no** empirical results — it explicitly says the open questions "require empirical investigation." This repository was meant to *be* that investigation. The engineering is excellent (hexagonal layers enforced by `pytest-archon`, Lagom DI, 725 tests at 100% coverage, 8 quality gates), but the three mechanisms the thesis depends on are each broken or decorative, and the headline experiment has never been validly run:

| # | Mechanism | Current state | Evidence |
|---|---|---|---|
| 1 | Projector training | MSE regression onto **uniform-random** Lab targets | `infrastructure/ml/pytorch_color_mapper.py:124-132`; `supervised_pytorch_color_mapper.py:175-180,200-208` |
| 2 | Perceptual distance | 1-D `scipy.stats.wasserstein_distance` over **bin index** `0…4095`; ignores Lab geometry | `infrastructure/ml/wasserstein_distance_calculator.py:13-16` |
| 3 | Codebook | Fixed 16³ uniform grid, **~88% out-of-gamut** bins; not learned VQ | `domain/model/color_codebook.py:43-57` |
| 4 | Reported result | `Color Method = TBD`; no trained artifacts exist | `README.MD:33` |

This roadmap fixes the chain end-to-end (P0), then addresses the blog's secondary claims (P1) and engineering/honesty debt (P2).

> **Delivery status (2026-07-17).** §1 — the table above and its diagnosis — describes the **pre-remediation baseline** that motivated this roadmap; every mechanism it flags has since been fixed. **P0–P2 are complete**, each milestone mapped to a committed spec:
>
> | Milestone | Specs | Status |
> |---|---|---|
> | P0-1…P0-6 — real, correct experiment | `001`–`006` | ✅ |
> | P1-1…P1-4 — secondary claims | `007`–`010` | ✅ |
> | P2-1…P2-6 — engineering & honesty | `011`–`016` | ✅ |
> | Beyond the roadmap | `017`–`027` (A4 image, lossless codec, scaled/matched-budget eval, falsifiable interpretability, rate–distortion frontier, authored corpus, narrative compass, book generation, real retrieval metrics, claims reconciliation) | ✅ |
>
> The only un-started items are the **§7 research extensions** (R-1 other modalities · R-2 learned/unconstrained target space · R-3 cross-system consistency).

---

## 2. Definition of Done (applies to every task)

A task is complete only when:

- [ ] `tox` is green — all gates pass (flake8, black, bandit, semgrep, xenon, radon, mypy, pip-audit).
- [ ] Coverage remains **100%**; new tests are meaningful, **one logical assertion each**, named `test_should_..._when_...`.
- [ ] **No comments**; code is self-documenting.
- [ ] Layer boundaries respected — domain stays pure (no `sklearn`/`torch`/`ot` imports in `domain/`); new adapters live in `infrastructure/` behind a `domain/service` port and are wired with Lagom.
- [ ] Any new endpoint returns a **Pydantic DTO**, never a dict.
- [ ] New dependencies are added to `setup.cfg`/`pyproject.toml` and pass `pip-audit`.

---

## 3. Sequence at a glance

```
P0  Make the experiment real and correct  (ship as one milestone; a partial bundle still yields a meaningless number)
    P0-1 POT Lab-EMD distance ─┐
    P0-2 structure-preserving training objective ─┤
    P0-3 shuffle/stratify dataset sampling ─┼──► P0-6 run end-to-end, fill the AG News table
    P0-4 eval reaches every mapper + ef>=k fix ─┤
    P0-5 seeding + structure-preservation metric ─┘

P1  The blog's secondary claims        (depends on a working, measurable P0 pipeline)
    P1-1 learned VQ codebook
    P1-2 honest interpretable mapper (real sentiment / concreteness)
    P1-3 real compression comparison (matched rate-distortion)
    P1-4 ablations (quantization levels, distance metrics)

P2  Engineering & honesty               (independent; can interleave, lower priority)
    P2-1 API: mount query / retire "coconut"
    P2-2 security: hash + de-commit credentials
    P2-3 real health checks
    P2-4 unify the two config systems
    P2-5 tests that assert science, not plumbing
    P2-6 reconcile docs (384 vs 768, "VQ" naming)

R   Research extensions                 (optional, only after P0 proves the core idea)
    R-1 other modalities · R-2 unconstrained-from-human-perception mapping · R-3 cross-system consistency
```

---

## 4. Milestone P0 — Make the experiment real and correct

> **Ship P0 as a unit.** Each task below removes one independent reason the color-method number would be invalid; the AG News result is only trustworthy once all five land. P0-1…P0-5 can be developed in parallel; P0-6 integrates them.

### P0-1 — Replace 1-D index Wasserstein with a true Lab EMD (POT)

- **Files:** `infrastructure/ml/wasserstein_distance_calculator.py`; port `domain/service/distance_calculator.py`; DI wiring in CLI/`interface`.
- **Change:** Compute Earth-Mover distance over the **perceptual** ground cost between codebook colors instead of `|index_i − index_j|`.
- **Approach (context7 / POT-grounded):**
  - Inject the `ColorCodebook` (or a precomputed cost matrix) into the calculator via Lagom — do **not** rebuild per call.
  - `coords = np.array([[c.l, c.a, c.b] for c in codebook.colors])` (4096×3, `float32`).
  - `M = ot.dist(coords, coords, metric='euclidean')` — precompute once (~67 MB `float32`); `metric='euclidean'` gives W1 (true EMD), not the `sqeuclidean` default.
  - `return float(ot.emd2(doc1.histogram, doc2.histogram, M))`; offer `ot.sinkhorn2(..., method='sinkhorn_log')` for corpus-scale retrieval, gated by a new `distance.sinkhorn_reg` config field.
  - Validate both histograms share the **same** codebook (today only `num_bins` is checked).
  - Note: `distance.smoothing_epsilon` (1e-8) is consumed only by Jensen-Shannon and is **not** a Sinkhorn regulariser — keep them distinct.
- **Acceptance:** a test asserting that moving mass between **perceptually close** codebook colors yields a *small* distance and between **perceptually distant** colors a *large* one (the current index version fails this). Add POT to deps; `pip-audit` green.
- **Effort:** M · **Depends on:** none.

### P0-2 — Structure-preserving training objective

- **Files:** `infrastructure/ml/pytorch_color_mapper.py` (and `supervised_pytorch_color_mapper.py`).
- **Change:** Delete `_generate_targets` (random Lab) and train the projector so **similar meanings map to similar colors**.
- **Approach (context7 / sentence-transformers-grounded):**
  - Per batch, build the frozen-teacher target `gold = util.cos_sim(emb, emb).detach()`.
  - Build a student similarity from the 3-D color outputs (e.g. `1 − normalized ΔE`, or cosine of mean-centred Lab).
  - `loss = MSE(student_offdiag, gold_offdiag)` — the structure-preserving analogue of `CosineSimilarityLoss`.
  - For the supervised variant, replace random-target MSE with `ContrastiveLoss`/`TripletMarginLoss` on color outputs keyed by class label; if keeping a multi-task sum, **normalise the Lab loss scale** (L→[0,1], a/b→[−1,1]) or use learnable uncertainty weights so the classification term is not swamped (today MSE is O(10³–10⁴) vs 0.1·CE at O(1)).
- **Acceptance:** training reduces a held-out similarity-discrepancy metric (see P0-5); a test asserts that two near-duplicate inputs land at near-identical Lab colors (would fail under random targets).
- **Effort:** L · **Depends on:** none (pairs with P0-5).

### P0-3 — Shuffle/stratify dataset sampling

- **Files:** `infrastructure/dataset/{ag_news,imdb,newsgroups}_dataset_adapter.py`.
- **Change:** `max_samples` currently takes a contiguous **head slice** with no shuffle; because IMDB is ordered neg-then-pos, the first 12 000 rows are **single-class on both train and test** → a meaningless 100%.
- **Approach:** seeded shuffle (and stratify by label where available) before truncating; thread the configured `seed` through.
- **Acceptance:** a test asserting both classes are present in an IMDB `max_samples` slice.
- **Effort:** S · **Depends on:** none.

### P0-4 — Make `eval` reach every mapper + fix candidate retrieval

- **Files:** `interface/cli/eval.py`, `application/use_case/evaluate_use_case.py`, `infrastructure/evaluation/color_histogram_classifier.py`, `infrastructure/evaluation/hnsw_classifier.py`.
- **Change:**
  - `eval.py:73` hardcodes the unconstrained `PyTorchColorMapper` and has no `--mapper-type`; loading a structured/supervised checkpoint into it **key-mismatches and raises**. Add `--mapper-type` and construct the matching network (a small mapper factory keeps this DRY and DI-friendly).
  - Fix `ef >= k`: `ef=50 < num_candidates=100` violates hnswlib's documented requirement → degraded/under-filled retrieval. Set `ef = max(ef, num_candidates)`; call `set_num_threads(1)` for deterministic builds.
- **Acceptance:** tests that `--mapper-type structured` evaluates without a state-dict error, and that `ef >= num_candidates` holds at query time.
- **Effort:** S/M · **Depends on:** P0-2 (mapper variants worth evaluating).

### P0-5 — Determinism + a structure-preservation metric

- **Files:** training bootstrap (e.g. `application/use_case/train_color_mapping_use_case.py` or the mappers); new `infrastructure/evaluation/structure_preservation_evaluator.py` behind a `domain/service` port.
- **Change:**
  - Honour the configured `seed`: `torch.manual_seed(cfg.seed)`, seed a `torch.Generator` and pass it to `torch.rand`/`torch.randperm`, `np.random.seed`; optional `torch.use_deterministic_algorithms(True)` behind a flag.
  - Add a metric reporting **Spearman correlation between embedding-space cosine similarity and color-space distance** (the `EmbeddingSimilarityEvaluator` idea, reimplemented for 3-D color) — this *directly* measures the blog's claim instead of inferring it from accuracy, and drives best-checkpoint selection.
- **Acceptance:** identical seeds reproduce identical Lab outputs; the evaluator returns a correlation in [−1, 1] and is used for checkpointing.
- **Effort:** S · **Depends on:** none (metric pairs with P0-2).

### P0-6 — Run end-to-end and fill the table

- **Change:** `train → encode → eval` on AG News; replace `README.MD:33` `Color Method | TBD | TBD` with a real **Accuracy / Macro-F1** next to TF-IDF 90.63% / HNSW 91.99%. Commit the run config and the structure-preservation correlation. **This is the deliverable the blog has been missing.**
- **Acceptance:** a reproducible command (documented in README) regenerates the reported numbers within tolerance.
- **Effort:** S · **Depends on:** P0-1…P0-5.

---

## 5. Milestone P1 — The blog's secondary claims

### P1-1 — Learned VQ codebook

- **Files:** new `infrastructure/ml/learned_color_codebook_factory.py` (KMeans lives in infrastructure to keep `domain` pure); `domain/model/color_codebook.py` already accepts any `List[LabColor]`; wire `train_color_mapping_use_case.py:32` to optionally build from data.
- **Approach:** fit `MiniBatchKMeans(n_clusters=num_bins, random_state=seed, n_init='auto')` on the **projected** Lab colors; store `cluster_centers_` as the palette. Vectorise quantization (`argmin(cdist)` / `kmeans.predict`) instead of the O(4096) Python loop. Keep `create_uniform_grid` as a baseline for A/B reconstruction.
- **Why:** reclaims the ~88% of bins wasted on out-of-gamut/empty cells; makes "vector quantization" honest.
- **Effort:** M · **Depends on:** P0-2 (meaningful Lab distribution), P0-1 (benefits retrieval).

### P1-2 — Honest interpretable mapper

- **Files:** `infrastructure/ml/structured_pytorch_color_mapper.py`.
- **Change:** today lightness = normalized **mean** of embedding dims and chroma = normalized **variance** — neither is sentiment or concreteness. Drive **lightness from real sentiment** (IMDB labels / a sentiment model) and **chroma from a real concreteness signal** (e.g. Brysbaert concreteness norms). Order hue clusters so hue adjacency ≈ semantic adjacency (e.g. 1-D ordering of cluster centroids) instead of the arbitrary `2π·idx/K`.
- **Why:** substantiates the blog's interpretability claim and enables measuring the **interpretability ↔ performance tradeoff** it explicitly poses.
- **Effort:** L · **Depends on:** P0.

### P1-3 — Real compression comparison

- **Files:** `application/use_case/compress_document_use_case.py`, `compression_comparison_use_case.py`, `infrastructure/ml/pq_compression_baseline.py`.
- **Change:** put color-VQ on the **same rate-distortion axis** as gzip/PQ: a genuine reconstruction error (mean ΔE of dequantized colors), an artifact that actually shrinks (today the "compressed" pickle stores the full color sequence), and removal of the fabricated `original_bits = num_tokens*8*10` and the non-decodable variable-width RLE. Fix PQ's in-sample k-means fit.
- **Effort:** M · **Depends on:** P0-1, P1-1.

### P1-4 — Ablations the blog names

- **Change:** sweep quantization levels (1024 vs 4096 vs learned) and distance metrics (EMD vs Jensen-Shannon vs cosine); report accuracy + structure-preservation per setting.
- **Effort:** M · **Depends on:** P0, P1-1.

---

## 6. Milestone P2 — Engineering & honesty

| ID | Change | Files | Effort |
|---|---|---|---|
| P2-1 | Mount `create_query_controller` (currently dead code) **or** delete it; retire/justify the content-free **"coconut"** CRUD so the API is about colors | `interface/api/main.py`, `interface/api/controller/query_controller.py` | S/M |
| P2-2 | Hash credentials and remove committed plaintext `admin/password` | `infrastructure/security/basic_authentication.py:25`, `resources/application.properties` | S |
| P2-3 | Make health checks real: readiness verifies codebook/model presence; liveness not hardcoded `True` | `infrastructure/system/health_checks.py` | S |
| P2-4 | Unify (or explicitly document) the two config systems — the science `synesthetic_config` never reaches API runtime | `shared/configuration.py`, `shared/synesthetic_config.py` | M |
| P2-5 | Add tests that assert **science** (structure preservation, distance ordering, an accuracy floor) — the suite would stay green if the projector emitted noise (`test_pytorch_color_mapper.py:68` is `assert True`); fix no-comment leaks in `hnsw_classifier.py` | `tests/...` | M |
| P2-6 | Reconcile docs: repo is 384-dim/1024:1 vs blog's 768-dim/2000:1; stop labeling the uniform grid "VQ" until P1-1 lands | `README.MD`, `.claude/CLAUDE.md` | S |

---

## 7. Research extensions (optional — after P0 proves the core idea)

- **R-1** Other perceptual modalities (soundscapes, tactile, taste) behind the same `ColorMapper`/`DistanceCalculator` ports.
- **R-2** Free the mapping from human perceptual constraints (learned target space) and compare.
- **R-3** Cross-system consistency: do independently trained synesthetic mappings discover similar structure?

---

## 8. Recommended first PR

**P0-1 (POT Lab-EMD distance)** — self-contained, high-signal, and unblocks honest retrieval immediately. It implements the existing `DistanceCalculator` port, needs only DI wiring and one new dependency, and ships with a test that pins perceptual geometry (which the current implementation provably fails). Pair it with **P0-2** in the same milestone branch so the first end-to-end run (P0-6) has both a real objective and a real metric.

---

## 9. Evidence appendix

| Finding | Location |
|---|---|
| Unconstrained mapper trains to random targets | `infrastructure/ml/pytorch_color_mapper.py:124-132` |
| Supervised mapper: dominant loss also random-target MSE | `infrastructure/ml/supervised_pytorch_color_mapper.py:175-180,200-208` |
| Structured axes are mean/variance proxies, not sentiment/concreteness | `infrastructure/ml/structured_pytorch_color_mapper.py:221-245` |
| Wasserstein is 1-D over bin index | `infrastructure/ml/wasserstein_distance_calculator.py:13-16` |
| `delta_e` (CIE76) defined but never called in `src/` | `shared/lab_utils.py` (no call sites) |
| Codebook is a fixed uniform grid; ~473/4096 in-gamut | `domain/model/color_codebook.py:43-57` |
| `max_samples` head-slice → IMDB single-class train & test | `infrastructure/dataset/imdb_dataset_adapter.py:17-19` |
| `eval` hardcodes unconstrained mapper; no `--mapper-type` | `interface/cli/eval.py:73` |
| hnswlib `ef=50 < k=100` | `infrastructure/evaluation/color_histogram_classifier.py:24-26,67` |
| No `torch.manual_seed` despite `seed: 42` | `shared/synesthetic_config.py:27`; absent in `src/` |
| Query endpoint never mounted; only "coconut"+health are | `interface/api/main.py` |
| Plaintext committed credentials | `infrastructure/security/basic_authentication.py:25`; `resources/application.properties` |
| Fabricated compression accounting / non-shrinking artifact | `application/use_case/compress_document_use_case.py:71`; `domain/model/colored_document.py:66-72` |
| Color Method result never produced | `README.MD:33` |
