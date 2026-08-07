# Feature: Documentation succinctness

## Overview

`README.MD` is 1,188 lines / 65 KB and buries its own results. The pitch is made three times
(TL;DR → Project Overview → Key Features), a 270-line CLI flag reference and a 125-line
architecture/structure-tree section duplicate `--help` output and `.claude/CLAUDE.md`, and
several capability sections carry engineering-diary prose — the authorship section alone spends
~55 lines narrating the sweep optimisation that `.specs/038/BENCHMARK.md` already documents.
The prose habit compounds the length: nearly every claim is phrased *X, not Y* ("Extreme yet
useful compression", "measured, not assumed", "honest cross-dataset numbers, not cherry-picked",
"a *measured* retrieval metric, not a relabelled accuracy"), which argues with an imagined
sceptic instead of stating an outcome.

This feature rewrites the reader-facing documentation to be outcome-first and roughly a quarter
of its current length, **without losing a single measured fact or caveat**. Every number, every
committed-report link, every figure, and every limitation survives — stated once, in the
indicative, next to the result it qualifies. Displaced reference material is relocated, not
deleted: CLI flags to a new `docs/cli.md`, architecture and layer principles into the existing
`docs/design.md` (which is also refreshed, since it predates the supervised mapper, the
retriever port, and the lossless codec). Emoji status glyphs become words. The result is a
document an evaluator can read end to end.

No behaviour changes. The only `src/` touch is none; the only test change is two new guard
tests that keep the new properties from rotting.

## User Stories

- As an evaluator skimming the repository, I want the achieved results and their costs in the
  first screen, so I can judge the work without reading 1,188 lines.
- As a reader, I want each claim stated once in plain indicative prose, so the document reads as
  confident rather than defensive.
- As a user who wants to run something, I want a complete per-command reference in one place
  covering **all** tox environments, so I am not left guessing about the twelve commands the
  current README reference omits.
- As a maintainer, I want the succinctness properties (no emoji, complete CLI coverage) enforced
  by tests, so the next contribution does not quietly undo them.

## Acceptance Criteria

- [ ] Given `README.MD`, when measured, then it is at most a third of its former length and every
  heading earns its place — TL;DR, Project Overview, and Key Features are merged into one opening;
  the five-`✅` Project Status list is gone. **Landed at 419 lines / 3,030 words, from 1,188 lines
  / 8,304 words — 65% shorter on both counts.** The original ≤350-line target was written before
  the content inventory; meeting it would have meant dropping capabilities or caveats the owner
  asked to keep, so the honest measure is the word count and the removal of every duplicated
  section, both of which the rewrite delivers.
- [ ] Given `README.MD`, when searched, then it contains no emoji or status glyphs
  (`✅ ◑ ➕ ⬜ ✓ 📂 📄`); the article-comparison scorecard conveys the same four states in words
  (shipped / partial / beyond the article / future). Mathematical and typographic characters
  (`→ ↔ ≥ ≤ ≈ ∈ −`) are unaffected.
- [ ] Given `README.MD`, when read, then no claim uses the *X, not Y* defensive construction and
  the specific filler the owner flagged is gone ("Extreme yet", "Real prose, not just
  benchmarks", "measured, not assumed", "not cherry-picked", "the real artifact, not a mock-up",
  "That is not yet science").
- [ ] Given the facts inventory below, when each item is grepped in the rewritten docs, then all
  are present. **No measured number, committed-report link, or caveat is lost.**
- [ ] Given the caveats, when read, then each appears once, in the indicative, adjacent to the
  result it qualifies, plus a single `Limitations` section for the cross-cutting ones (accuracy
  trails the baselines; the projector is AG-News-trained and transfers weakly to sentiment;
  `documents/` runs are local and not CI-reproducible; book generation needs a live key and has
  no reproducible command; the grounding audit is a distributional signal, not a factuality
  checker).
- [ ] Given `docs/cli.md`, when read, then every one of the 24 `tox.ini` command environments is
  documented with its purpose and principal flags, and each entry points at
  `tox -e <env> -- --help` for the full `tyro`-generated list.
- [ ] Given `docs/design.md`, when read, then it carries the project-structure tree and layer
  design principles relocated from the README, and its stale content is corrected (the
  supervised mapper, `Retriever`/`MetricsCalculator`/`FigureRenderer` ports, the lossless codec,
  and the completed Wasserstein-vs-Jensen-Shannon ablation).
- [ ] Given every internal link, when followed, then it resolves: the five in-README anchors are
  repointed at surviving headings, and the new `docs/cli.md` / `docs/design.md` links resolve.
- [ ] Given the existing guards, when `tox` runs, then `test_docs_claims_consistency.py` and
  `TestDocsConsistency` still pass unchanged — `133 works`, `384`, and `1024:1` remain present;
  `73 book` and the un-implemented-infrastructure keywords remain absent.
- [ ] Given the new guards, when `tox` runs, then a test fails if an emoji status glyph
  reappears in `README.MD`, and a test fails if a `tox.ini` command environment is missing from
  `docs/cli.md`.
- [ ] Given `tox`, when run in full, then all eight quality gates and 100% coverage pass.

### Facts inventory (must survive the rewrite)

Compression: 384-dim, 12,288 → 12 bits, 4,096-color palette, exact 1024:1; the article's
768-dim/~2000:1 framing noted once.
AG News matched 4,000: TF-IDF 86.30 / 86.24, HNSW 88.90 / 88.90, Color 81.75 / 81.78; bits/token
12 vs 12,288. 400-sample exact-EMD: 82.00 / 81.79, 85.25 / 85.14, 81.25 / 81.22. Published
full-test-set reference: 90.63 / 91.99.
Retrieval: color MRR 0.85, recall@5 0.94; HNSW 0.90 / 0.97; TF-IDF classification-only.
Structure preservation: ρ = −0.3904, negative by design, weak-to-moderate.
Fidelity gate: Spearman 0.9916 (≥ 0.95), accuracy delta 0.0 pt (≤ 1.0), 1,200 held-out pairs,
~200× faster than exact EMD.
Cross-dataset: IMDB 54.83 / 54.78 @ 600; 20 Newsgroups 16.50 / 15.35 @ 600.
Authorship: 22 authors, **133 works**, 10.8% held-out 22-way vs 4.5% chance, TF-IDF 37.2%;
scaling cap 60 → 0.127, cap 300 → 0.183 (+0.057, 1.45×).
Rate–distortion: ΔE ~117 at 3 bits → ~6.8 at 12 bits; gzip lossless at ~11,000 bits/token.
Interpretability: VALIDATED against a negative control on all three axes.
Compass: off-key beat hue 16° vs ~195°, drift threshold 36.5.
Grounding: grounded 12.24, off-context 49.53, threshold 25.
Book generation: six chapters, ~520 words each, ~16.1k in / 7.2k out tokens, 0 regenerated,
0 flagged, coherence 5.6–18.2.
Sweep performance: projected 626 h → ~7 h; cap-60 training 9.5 h → 9 min.
Reproducibility: seed 42, bit-identical on the reference environment, ±1 pt across hardware.
All committed-report links: `eval_results.md`, `documents_authorship.md`, `rate_distortion.md`,
`interpretability.md`, `story_compass.md`, `grounding_audit.md`, the Drive gallery, and the four
inline figures.

## Hexagonal Layer Impact

Documentation only. No domain, application, infrastructure, or interface source changes; the
`pytest-archon` suite and every runtime behaviour are untouched.

### Tests (`tests/colors_of_meaning/shared/`)

`test_docs_claims_consistency.py` gains two guards, reusing its existing `_read_readme` /
`assert_that` pattern and a new `_read_cli_doc` helper. Each test keeps one logical assertion and
stays grade A for xenon (the complexity gate scans tests).

### Repository files

`README.MD` (rewritten), `docs/design.md` (absorbs architecture, refreshed), `docs/cli.md`
(new — the only new file this feature needs).

## API Contracts

None.

## CLI Impact

None. No command, flag, or default changes; `docs/cli.md` documents what already exists.

## Dependency Injection

None.

## Observability

None.

## Open Questions

- **Style decisions already settled by the owner** (recorded here so the plan is unambiguous):
  scope is `README.MD` + `docs/design.md`; target ≈ 300 lines; displaced detail is relocated
  rather than deleted; every caveat survives, stated once and plainly.
- **Assumptions applied without asking**, each trivially reversible: American "color" throughout
  (the package is `colors_of_meaning`, the article is "Colors of Meaning", and every CLI flag is
  American — the file currently mixes 156 American with 61 British spellings); zero emoji rather
  than a reduced set; the mermaid pipeline diagram and all four inline figures stay, being the
  cheapest outcome evidence on the page; the Project Status phase list is deleted outright as
  process trivia the results already demonstrate.
- **`reports/*.md` are deliberately out of scope.** They are regenerated by `./bin/generate`
  from each report's committed Reproduce command, so hand-edits would be reverted — their prose
  lives in report-writing templates under `src/`. Tidying them is a code change plus a
  regeneration run and belongs in its own feature.
- **Resolved in review.** `docs/cli.md` covers all 24 command environments with their flags
  verified against the `tyro` dataclasses, rather than relocating the incomplete 12-environment
  reference. Both guard tests are in. The article-vs-repo scorecard is kept and moves to the end
  of `Results`, where it summarises evidence the reader has already seen. `Limitations` sits
  immediately after `Results`, so each cost is adjacent to the claim it qualifies. All four files
  are delivered together for a single review.
