# Plan: Documentation succinctness

## Implementation Strategy

Rewrite `README.MD` around a single spine — *what it achieves, what that costs, how to run it* —
and relocate everything that is reference rather than argument. Three files change and one is
created:

| File | Action | Lines before → after |
|---|---|---|
| `README.MD` | rewritten in place | 1,188 → ~310 |
| `docs/design.md` | absorbs architecture, refreshed for drift | 94 → ~170 |
| `docs/cli.md` | **new** — the complete per-environment command reference | — → ~180 |
| `tests/…/test_docs_claims_consistency.py` | two guards added | 55 → ~75 |

The rewrite is a replacement, not a sequence of edits: the current text's problem is its shape,
and 1,188 lines of surgical `Edit` calls would preserve the shape. Facts move across
mechanically from the inventory in `SPEC.md`, which is checked item by item afterwards.

`docs/cli.md` is the one new file. It is justified rather than convenient: the README's current
"CLI Reference" documents 12 of the 24 command environments in `tox.ini`, so porting it verbatim
would carry a half-complete reference into a new home. The new document covers all 24, records
each environment's purpose and principal flags, and defers exhaustive flag lists to
`tox -e <env> -- --help`, which `tyro` generates from the CLI dataclasses and which therefore
cannot drift.

## Target README structure

~310 lines, in this order. Line budgets are the ceiling, not a target to fill.

| § | Content | Lines |
|---|---|---|
| Title, badges, one-line hook | unchanged badges; hook states the outcome, not the aspiration | 8 |
| **What this is** | the mapping in two paragraphs: 384-dim → 12-bit color over a 4,096-color Lab palette (exact 1024:1), documents as color distributions; article link; the 768-dim/~2000:1 difference in one clause | 18 |
| **Results** | one section replacing today's three. Sub-parts: matched-budget AG News table (+ retrieval line + ρ = −0.3904); cross-dataset table; authorship on real prose; rate–distortion; interpretability; then the article-vs-repo scorecard as a closing summary, its four states in words. Each result carries its cost in the same sentence as its number. Links to the six committed reports | 87 |
| **Limitations** | the five cross-cutting caveats, stated once, plainly | 14 |
| **Gallery** | the t-SNE, A4 gallery, and lossless-barcode figures with the decode command; Drive link | 20 |
| **Pipeline** | mermaid diagram, two-sentence caption | 40 |
| **Quick start** | environment, train + eval the color method, the two baselines | 26 |
| **What you can do with it** | ten capability entries — palette query, corpus comparison, A4 semantic sheets, lossless barcodes, rate–distortion sweep, interpretability check, narrative compass, grounding audit, book generation, your own corpus. Each is outcome + one command, ~5 lines | 60 |
| **Reproducing the results** | condensed from today's three overlapping sections: run config, the three commands, artifact paths, the seed/tolerance note, the fidelity gate | 30 |
| **Development** | tested configuration, vagrant instructions, the commands table (retained — it is the best index on the page), test/format/run, API authentication | 35 |
| **Architecture** | five layer bullets, then out to `docs/design.md`, `docs/cli.md`, `.claude/CLAUDE.md` | 10 |
| **Contributing** | six existing points, condensed | 8 |

### Cut outright

- **Project Overview** and **Key Features** (§ 125–144) — the third and second restatements of
  the opening. Their only unique content is the supervised mapper and the ARM64/hnswlib note,
  which move to *What this is* and *Architecture*.
- **Project Status** (§ 310–318) — five phases, all `✅`. Process trivia; the results prove it.
  The Quality Gates line moves to *Development*.
- **Quick Start** (§ 320–344) as a separate section — it duplicates *Reproducing the Color
  Method Result* command for command.
- **Generate Visualizations** (§ 631–645) — verbatim duplicate of the CLI-reference visualization
  block.
- **Query by Color Palette** appears at both § 397 and § 897 — one survives.
- The sweep-performance narrative inside the authorship section (§ 496–525, ~30 lines) → two
  sentences plus the existing link to `.specs/038/BENCHMARK.md`, where it already lives in full.
- **Claude Code Skills** (§ 1166–1177) → one line pointing at `.claude/skills/README.md`.

### Prose rules applied throughout

State the outcome and its cost in the indicative. Delete the *X, not Y* construction wherever it
appears — the metric's name and number already carry the distinction:

> before: "…**retrieves same-class documents at MRR 0.85 / recall@5 0.94** — ranked by the same
> perceptual Wasserstein distance, a *measured* retrieval metric, not a relabelled accuracy —
> against matched-budget TF-IDF / HNSW at **86.3% / 88.9%**"
>
> after: "Ranked by perceptual Wasserstein distance, color retrieval scores MRR 0.85 and
> recall@5 0.94, against HNSW's 0.90 and 0.97."

Bold only where a term is being defined. One idea per sentence; no three-clause em-dash stacks.
American spelling throughout.

## Layer Changes

None. No file under `src/colors_of_meaning/` is modified, so domain purity, layer boundaries,
and the `pytest-archon` suite are unaffected by construction.

### Tests (`tests/colors_of_meaning/shared/test_docs_claims_consistency.py`)

Two guards, following the module's existing helper + `assert_that` shape:

- `test_should_not_use_status_emoji_when_reading_readme` — asserts a module-level
  `STATUS_EMOJI` tuple (`✅ ◑ ➕ ⬜ ✓ 📂 📄`) has no member in the README. Deliberately a literal
  tuple rather than a Unicode range, so `→ ↔ ≥ ≤ ≈ ∈ −` cannot be caught by accident.
- `test_should_document_every_command_environment_when_reading_cli_doc` — parses
  `[testenv:<name>]` headers from `tox.ini`, drops the non-command envs
  (`format`, `watch`, `clean`, `build`, `publish`), and asserts none is missing from
  `docs/cli.md`. The filtering happens in a module-level helper so both the test and the
  xenon grade-A requirement are satisfied (the complexity gate scans `tests/`).

## Dependency Injection

None.

## Task List

1. [ ] docs: draft the rewritten `README.MD` against the structure table above, moving facts
   across from the `SPEC.md` inventory.
2. [ ] docs: create `docs/cli.md` covering all 24 command environments; verify every documented
   flag against its CLI dataclass in `src/colors_of_meaning/interface/cli/` (they are `tyro`
   dataclasses, so the fields are the flags) and note any drift found rather than copying it
   forward.
3. [ ] docs: move the project-structure tree and Design Principles into `docs/design.md`;
   correct its stale content — the supervised mapper, the `Retriever` / `MetricsCalculator` /
   `FigureRenderer` ports, the lossless codec, and ablation item 4 (Wasserstein vs
   Jensen-Shannon), which has since been run.
4. [ ] docs: repoint the five in-README anchor links and add the `docs/cli.md` / `docs/design.md`
   links; confirm the four inline figure paths still resolve.
5. [ ] tests: add the two guards to `test_docs_claims_consistency.py`.
6. [ ] verify: walk the `SPEC.md` facts inventory item by item against the new files; grep the
   flagged filler phrases to confirm each is gone.
7. [ ] verify: run `tox` in full — all eight gates plus 100% coverage.

## Testing Strategy

- One logical assertion per test, `assertpy`'s `assert_that`, names following
  `test_should_<behaviour>_when_<condition>`.
- The two new guards join the four existing docs-consistency guards; nothing existing changes.
- No mocks or fixtures needed — the guards read files from the repository root, as the module
  already does.
- Verification is `tox`, never bare `pytest`.
- Coverage is unaffected: no `src/` line is added or removed.

## Observability Plan

None — no runtime code changes.

## Risks and Mitigations

- **Risk: a measured fact is silently lost in a 1,188 → 310 line rewrite.** → The `SPEC.md`
  facts inventory is a checklist walked item by item in task 6, and the pre-existing guards
  independently pin `133 works`, `384`, and `1024:1`.
- **Risk: the rewrite trips a guard test.** → The five enforced literals are enumerated in the
  spec; `tox` in task 7 is the gate. Note the guards read `README.MD` (uppercase extension) —
  the filename must not change, and `pyproject.toml` ships it as the package long description.
- **Risk: `docs/cli.md` is born stale.** → Task 2 verifies each flag against its `tyro`
  dataclass, the document defers exhaustive lists to `--help`, and the new coverage guard fails
  if an environment goes undocumented.
- **Risk: cutting the defensive framing reads as over-claiming**, weakening the repository's
  main differentiator. → Every caveat survives; only its repetition and its arguing-with-a-
  sceptic phrasing are removed. The new `Limitations` section makes the cross-cutting ones more
  prominent than they are today, where they are scattered across nine sections.
- **Risk: the owner wants a different tone than the sample rewrite implies.** → Task 1 produces
  the draft README for review before tasks 2–7 proceed; the tone is settled on one file rather
  than four.
