# Feature: Committed generators for the lossless corpus figures (`tox -e visualize_lossless`)

## Overview

The project's standing rule is that every artifact under `reports/` regenerates from a
committed command (`006`, `019`, `022`, `026`, `028`). `./bin/generate` now makes that rule
checkable end to end: twelve stages drive the CLIs with each report's own `## Reproduce`
command, and a verification pass confirmed every `.md` and all 133 `reports/figures/a4/*.png`
sheets come back byte-identical.

Building it surfaced the residual gap. Four files under `reports/` have **no generator anywhere
in the repository** — a repo-wide search finds nothing in `src/`, `bin/`, `tox.ini` or a
Makefile that writes them:

| file | what it is |
|---|---|
| `figures/documents_lossless_gallery.png` | captioned contact sheet of every book's lossless barcode |
| `figures/documents_two_ways.jpg` | titled side-by-side: one book's semantic sheet vs its barcode |
| `figures/pipeline_preview.png` | a Mermaid render of the README flowchart |
| `linkedin_post.md` | hand-written prose |

This feature adds committed generators for the first two. Both are pure composition over PNGs
that committed commands already produce — `visualize_documents` writes the per-book semantic
sheets, and `bin/generate`'s `lossless-corpus` stage writes the per-book barcodes via
`encode_lossless` — so neither needs a projector, codebook, embedding model or dataset.

Building them also exposed that `reports/figures/lossless/` had drifted: it held **73 distinct
works**, the pre-`027` corpus count, against today's 133. The `lossless-corpus` stage now
regenerates all 133 (144 files; ten books span multiple pages). This feature depends on that
stage having run.

**`figures/pipeline_preview.png` is explicitly out of scope.** Rendering it needs `mermaid-cli`
(Node + Puppeteer), an external toolchain outside the uv/Python dependency closure, for a single
preview image. `linkedin_post.md` is authored prose, not a machine artifact. `bin/generate`
already names both in `--help` so the script never implies coverage it does not have.

## Core Domain Concepts

- **Lossless barcode page** (`018`) — a model-free A4 colour barcode whose cells are data
  symbols, decoding back byte-for-byte. Carries no meaning; the pixels are compressed bytes.
- **Semantic A4 sheet** (`028`) — the lossy ~1024:1 rendering, horizontal bands of the book's
  palette sized by how often the projector maps its sentences to each colour.
- **Contact sheet** — an author-ordered grid of per-book images, already implemented for the
  semantic sheets as `render_a4_gallery`.
- **Two-ways composite** — the two renderings of one book side by side, captioned to make the
  lossy/lossless distinction legible at a glance. This is the figure that carries the project's
  central claim visually: the same book, compressed for *meaning* and compressed for *bytes*.

## User Stories

- As an evaluator, I can regenerate every figure the README and the write-up link, so no image
  in the repository is an unreproducible one-off.
- As a maintainer, `./bin/generate` covers the lossless figures too, so corpus growth cannot
  silently leave a figure describing an older corpus — the failure `lossless/` already had.
- As a contributor, I can render the comparison for any book in the corpus, not just the one
  that happens to be committed.

## Acceptance Criteria

1. `tox -e visualize_lossless` writes `reports/figures/documents_lossless_gallery.png` — an
   author-ordered contact sheet over `reports/figures/lossless/*.png`, each tile captioned
   `<author>/<work>`.
2. The same command writes `reports/figures/documents_two_ways.png` — a titled two-panel
   composite of one book's semantic A4 sheet and its lossless barcode, with the captions
   `meaning · lossy ~1024:1` and `exact bytes · lossless, byte-for-byte`.
3. Neither figure loads a projector, codebook, embedding model or dataset. The command runs
   with `artifacts/` absent.
4. Missing source images fail with an actionable message naming the stage that produces them,
   never a bare `FileNotFoundError` or a silently half-empty grid.
5. Multi-page books (`<work>_p01.png`, `_p02.png`) contribute one tile per page, captioned so
   the page is identifiable.
6. `documents_two_ways.png` is committed (un-ignored in `.gitignore`). The gallery stays
   git-ignored: it is 6.1 MB today and roughly 11 MB at 133 works, of incompressible noise.
7. `render_a4_gallery`'s existing callers are unaffected — `reports/figures/documents_a4_gallery.png`
   regenerates byte-identically apart from the known matplotlib version stamp.
8. `./bin/generate` gains a stage for the new command, ordered after `documents-figures` and
   `lossless-corpus`, and skipping cleanly when the corpus is absent.
9. `tox` passes with 100% coverage; `shellcheck -x bin/*` stays clean.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

`domain/service/figure_renderer.py` changes in two ways, both port-level only:

- `render_a4_gallery` gains an optional `labels: Optional[List[str]] = None` argument. Defaulted,
  so every existing caller is untouched and the committed A4 gallery keeps its exact bytes.
- One new abstract method for the composite, shaped as data rather than layout —
  `render_image_comparison(panels: List[Tuple[str, str]], title: str, output_path: str)`, where
  each panel is `(image_path, caption)`. Two panels today; the signature does not forbid more.

No new business rule. The `<author>__<work>` filename convention is already established by
`028` and `022`.

### Application Layer (`src/colors_of_meaning/application/`)

New `application/use_case/visualize_lossless_use_case.py`, mirroring
`VisualizeDocumentsUseCase`: it receives the `FigureRenderer` port plus a resolved list of
source image paths and delegates. It performs no filesystem globbing and no I/O itself, so it
stays testable without a temp directory.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

`infrastructure/visualization/matplotlib_figure_renderer.py` implements both:

- `_draw_gallery_tile` gains an optional caption drawn beneath the tile, matching the existing
  artifact's `<author>/<work>` style.
- `render_image_comparison` lays the panels out on one row with a figure title and per-panel
  captions.

Both read PNGs through `plt.imread`, as `render_a4_gallery` already does. Note the
`PIL.Image.open` `ResourceWarning` trap from `021` — the suite runs under `-W error`.

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/visualize_lossless.py` and `[testenv:visualize_lossless]`, taking
`--lossless-dir`, `--sheets-dir`, `--book`, `--title`, `--columns`, `--gallery-path`,
`--comparison-path` and `--dpi`. It resolves and sorts the source paths, derives captions from
filenames, and calls the use case. Filename→caption derivation is the only parsing it does.

`bin/generate` gains a `lossless-figures` stage. The README command table gains the environment.

### Shared Layer (`src/colors_of_meaning/shared/`)

A pure helper converting `<author>__<work>[_pNN].png` to its `author/work` caption and ordering
tiles by author then work. Framework-free, so it can live beside `shared/document_corpus.py`.

## API Contracts

None. No HTTP surface; this is a CLI-and-figure feature.

## CLI Impact

One new `tox` environment. No existing CLI flag changes. No new third-party dependency —
matplotlib and Pillow are already in the closure.

## Dependency Injection

The CLI constructs `MatplotlibFigureRenderer` and injects it into the use case as the
`FigureRenderer` port, exactly as `visualize_documents` does. The use case never names a
concrete renderer.

## Observability

Both figure writes log at INFO with a `correlation_id` (`uuid4`), the source count, the output
path and the book, matching the sibling visualize CLIs. No metrics or tracing backend exists in
this project and none is added.

## Decisions

Settled before drafting:

| question | decision |
|---|---|
| which figures | both |
| CLI shape | one new env; pure composition, no model |
| commit which | `documents_two_ways.png` only; gallery stays git-ignored |
| captions | optional `labels` on the shared `render_a4_gallery` port |
| composite format | PNG, not the existing JPEG |
| `pipeline_preview.png` | out of scope |

## Open Questions

1. **Default book for the composite.** Proposed `darwin/origin_of_species`, matching the
   existing artifact and the write-up's "Darwin's bright-red science prose" framing.
2. **Default title.** The existing figure reads *"On the Origin of Species — two ways to colour
   a book"*, which cannot be derived from the filename `origin_of_species.txt`. Proposed: a
   `--title` flag defaulting to `"<Work Name> — two ways to colour a book"` with the work name
   title-cased from the filename, so the committed wording is passed explicitly.
3. **Stale `documents_two_ways.jpg`.** It is untracked and has no generator, so deleting it
   before the replacement is verified would destroy the only copy. Proposed: leave it in place
   and remove it once the PNG is accepted.
4. **Gallery columns.** The existing sheet is 11 columns; `visualize_documents` defaults to 12.
   Proposed: default 12 for consistency, exposed as `--columns`.
