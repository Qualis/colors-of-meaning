# Plan: Committed generators for the lossless corpus figures

## Implementation Strategy

Land in four independently-committable stages, ordered so the risky part comes first. The one
load-bearing risk is the shared port: `render_a4_gallery` is already used to produce a
**committed** artifact (`reports/figures/documents_a4_gallery.png`), so Stage 1 changes that
signature and proves the existing bytes are unchanged before anything new depends on it. If
Stage 1 cannot preserve those bytes, fall back to a separate captioned-gallery port method
without touching the existing one.

Stages 2 and 3 are additive — new use case, new renderer method, new CLI — and cannot regress
an existing artifact. Stage 4 is wiring and docs.

Every stage is verified by the full `tox` at 100% coverage, per the project gate.

## Layer Changes

Domain: one optional argument on an existing port method, one new port method. Application: one
new use case. Infrastructure: two renderer implementations. Interface: one CLI plus a tox env.
Shared: one pure filename helper. No dataset, embedding, projector or codebook code is touched.

### Stage 1 — optional gallery captions (independently committable)

- `domain/service/figure_renderer.py`: `render_a4_gallery` gains
  `labels: Optional[List[str]] = None`.
- `infrastructure/visualization/matplotlib_figure_renderer.py`: `_draw_gallery_tile` draws a
  caption beneath the tile when one is supplied. Keep the function grade A — `xenon` scans
  tests as well as `src`, so extract any filter into a helper rather than nesting.
- Tests: a captioned gallery renders without error; an uncaptioned call is unchanged; a
  `labels` list shorter than `sheet_paths` raises rather than silently mislabelling.
- **Verify the no-regression claim directly**: regenerate `documents_a4_gallery.png` via
  `./bin/generate --only documents-figures` and confirm `cmp -l` reports exactly the 5-byte
  matplotlib version stamp and nothing more. This is the stage's acceptance gate.

### Stage 2 — the comparison renderer (independently committable)

- `domain/service/figure_renderer.py`: add `render_image_comparison(panels, title, output_path)`.
- Implement in the matplotlib adapter: one row of panels, figure title, per-panel caption.
- Tests: panel count drives the axes count; the title and captions reach the figure; an empty
  `panels` list raises. Use `mocker` against `plt` as the sibling renderer tests do, plus one
  test that writes a real file to a `tmp_path` — mocked-renderer tests missed real crashes in
  `visualize_corpus` that only an adversarial review caught, so at least one test must exercise
  the real matplotlib path.
- Watch the `PIL.Image.open` `ResourceWarning` from `021`: the suite runs under `-W error`.

### Stage 3 — use case, filename helper and CLI (independently committable)

- `shared/`: pure helper mapping `<author>__<work>[_pNN].png` to its `author/work` caption and
  sorting tiles by author then work.
- `application/use_case/visualize_lossless_use_case.py`: takes the `FigureRenderer` port and
  resolved paths; delegates. No globbing, no I/O.
- `interface/cli/visualize_lossless.py` + `[testenv:visualize_lossless]`: resolve `--lossless-dir`
  and `--sheets-dir`, derive captions, call the use case. Missing sources raise with a message
  naming the producing stage, per acceptance criterion 4.
- Tests: the CLI passes sorted paths and derived captions to the use case; a missing source
  directory raises the actionable error; the use case delegates to the port. Follow the
  `tyro`-dataclass CLI test pattern of the sibling `visualize_*` modules.

### Stage 4 — wiring, gitignore and docs (independently committable)

- `bin/generate`: add the `lossless-figures` stage after `lossless-corpus`, with the same
  `documents_corpus_available` guard and a `stage_description` entry.
- `.gitignore`: `!reports/figures/documents_two_ways.png`. Note `028`'s trap — `reports/*` and
  `reports/figures/*` swallow new committed report files unless explicitly un-ignored.
- Generate the committed `documents_two_ways.png` and commit it.
- README: add `tox -e visualize_lossless` to the command table; update the lossless-codec
  section to reference the composite. Leave `documents_two_ways.jpg` on disk until the PNG is
  accepted (Open Question 3).

## Dependency Injection

The CLI builds `MatplotlibFigureRenderer` and injects it as the `FigureRenderer` port. The use
case is constructed with that port and nothing else, so its tests need no filesystem.

## Task List

1. [ ] domain: optional `labels` on `render_a4_gallery`
2. [ ] infrastructure: caption drawing in `_draw_gallery_tile`
3. [ ] tests: caption behaviour, length mismatch, uncaptioned unchanged
4. [ ] verify: `documents_a4_gallery.png` diff is the 5-byte version stamp only
5. [ ] domain: `render_image_comparison` port method
6. [ ] infrastructure: implement it, including one real-file test
7. [ ] shared: filename → caption helper and tile ordering
8. [ ] application: `VisualizeLosslessUseCase`
9. [ ] interface: `visualize_lossless.py` CLI and `[testenv:visualize_lossless]`
10. [ ] interface: actionable missing-source error
11. [ ] bin/generate: `lossless-figures` stage after `lossless-corpus`
12. [ ] .gitignore: un-ignore `documents_two_ways.png`
13. [ ] generate and commit the composite
14. [ ] README: command table row and lossless-section reference
15. [ ] `tox` green at 100% coverage; `shellcheck -x bin/*` clean

## Testing Strategy

Unit tests per layer, one logical assertion each, named `test_should_..._when_...`. Renderer
tests mock `plt` for behaviour and write one real file to `tmp_path` for the smoke path. Use
case tests inject a mock port. CLI tests drive the `tyro` dataclass directly, as the sibling
`visualize_*` tests do. No network, no dataset, no model load anywhere in the suite — this
feature's whole point is that it needs none.

Architecture tests already enforce the layer rules; the new use case must not import matplotlib
and the domain port must stay framework-free.

## Observability Plan

INFO log with `correlation_id`, source count, output path and book on each figure write, using
the stdlib logger. The project has no metrics or tracing backend and none is added.

## Risks and Mitigations

| risk | mitigation |
|---|---|
| Changing `render_a4_gallery` regresses the committed A4 gallery | Stage 1 gates on a `cmp -l` diff of exactly 5 bytes; the argument is optional so existing calls are untouched |
| Mocked renderer tests hide a real matplotlib crash | At least one test per renderer method writes a real file to `tmp_path` |
| `-W error` turns a Pillow `ResourceWarning` into a failure | Known from `021`; open images in a context manager |
| Composite depends on two prior stages' output | Actionable error naming the producing stage; `bin/generate` orders them |
| Gallery grows unbounded with the corpus | Not committed, so repository size is unaffected; `--columns` controls layout |
| `xenon` grade-A gate trips on test helpers | Extract comprehension and filter logic into helpers, per the repo's existing practice |

## Validation against the spec

Criteria 1–2 are covered by Stage 3's CLI tests plus a real run. Criterion 3 is provable by
running the command with `artifacts/` moved aside. Criterion 4 is a Stage 3 test. Criterion 5
falls out of the filename helper's `_pNN` handling and is unit-tested. Criteria 6 and 8 are
Stage 4. Criterion 7 is Stage 1's acceptance gate. Criterion 9 is the standing project gate.
