# Feature: Narrative color compass — a story arc as a measurable color trajectory

## Overview

The project's neural network is a **lossy encoder**: `LabProjectorNetwork` /
`StructuredLabProjectorNetwork`
(`src/colors_of_meaning/infrastructure/ml/structured_lab_projector_network.py`)
map a 384-dim embedding to a 3-dim Lab colour and never run in reverse — there is
no decoder and no text-generation path anywhere in the codebase. So the network
cannot *write* a book. But the very same machinery can **plan and audit** one: an
ordered story outline can be turned into a measured colour trajectory, and drift
in that trajectory is a cheap, interpretable coherence signal.

This feature makes **an ordered list of narrative beats (chapters or scenes,
provided as text) a first-class analysable object**. It introduces an
`AnalyzeNarrativeArcUseCase` that, for each beat, produces two complementary
readings, both built from components that already exist:

- **Interpretable axis reading (the "story shape").** Each beat's *whole text* is
  embedded once (`SentenceEmbeddingAdapter.encode`) and read through the
  `StructuredColorMapper` (the honest interpretable mapper from feature 008), whose
  axes are real signals — **lightness = sentiment, chroma = concreteness,
  hue = topic cluster** (validated in feature 020). Plotting lightness, chroma, and
  hue over the beat index shows the emotional arc, the action↔reflection texture,
  and topic wander at a glance.
- **Drift / coherence reading (the QA signal).** Each beat is also turned into a
  `ColoredDocument` (`domain/model/colored_document.py`) — a histogram over the
  4,096-colour codebook built from the beat's sentences — and the perceptual
  Earth-Mover distance (feature 001; sliced-Wasserstein / Jensen–Shannon from
  feature 019) is computed **between consecutive beats** and **between each beat and
  the whole-book palette**, via the existing `DistanceCalculator` port. Beats whose
  drift exceeds a configurable threshold are flagged.

Critically, the per-beat colour is read from the beat's **document embedding**, not
the mean of its sentence colours: averaging Lab across sentences cancels chroma
(`sqrt(a² + b²)`) and destroys the concreteness signal — the documented aggregation
trap the earlier corpus work already hit. The drift reading deliberately uses a
*histogram* (a distribution, not an average), which is exactly what `ColoredDocument`
represents.

The result is a committed `reports/story_compass.md` table and a
`reports/figures/story_compass.png` figure regenerable from one command over a
checked-in sample outline. **No new third-party dependency is introduced** — the
embedder, structured mapper, codebook, distance calculators, and
`MatplotlibFigureRenderer` (`domain/service/figure_renderer.py`) already exist. This
feature is the foundation that feature 024 (Claude-backed book generation) consumes
for coherence QA, but it stands alone as a story-planning instrument and — unlike
024 — is fully CI-reproducible because it never calls an external service.

## Core Domain Concepts

- **Narrative beat**: one ordered unit of a story (a chapter or scene), a block of
  prose or synopsis text with an index and an optional title. The atomic input.
- **Story outline**: the ordered sequence of beats for one work; the unit the
  compass analyses end to end.
- **Beat colour (document-level)**: the single Lab colour of a beat, read from the
  beat's *whole-text* embedding through the structured mapper — decomposed into
  lightness (sentiment), chroma (concreteness), and hue angle (topic). Never the
  mean of the beat's sentence colours.
- **Beat histogram**: the beat rendered as a `ColoredDocument` — a distribution over
  codebook colours built from its sentences — used only for drift/coherence, where a
  distribution (not an average) is the correct object.
- **Whole-book palette**: the mixture histogram over all beats' histograms; the
  reference distribution each beat's coherence is measured against.
- **Drift**: the `DistanceCalculator` distance between consecutive beat histograms —
  how far the story moves, beat to beat.
- **Coherence**: the distance between a beat's histogram and the whole-book palette —
  how far a beat sits from the story's centre of mass; large values flag a beat that
  clashes with the rest.
- **Narrative arc**: the ordered `NarrativeArc` value object holding one `ArcPoint`
  per beat (beat, beat colour, histogram), the drift series, the coherence series,
  and the set of threshold-exceeding flagged beats.

## User Stories

- As a story planner, I want to see the emotional arc (lightness) and the
  action↔reflection texture (chroma) across my outline, so I can judge the shape of
  the story before writing it.
- As a story planner, I want beats that clash in topic or tone with the rest of the
  book flagged automatically, so I can catch a wandering or off-key beat early.
- As a story planner, I want a rendered arc figure and a drift table from a single
  command over an outline file, so the shape of the story is a shareable artifact.
- As a maintainer, I want the compass to reuse the existing embedder, structured
  mapper, codebook, distance, and renderer ports rather than reimplementing any of
  them, and to be CI-reproducible from a checked-in sample outline.
- As a contributor, I want a clean `NarrativeArc` object with a drift/coherence API
  that feature 024's generator can audit its own output against.

## Acceptance Criteria

- [ ] Given an ordered outline of N beats, when the use case runs, then the returned
  `NarrativeArc` has exactly N `ArcPoint`s in input order, each carrying the beat, its
  document-level Lab colour, and its `ColoredDocument` histogram.
- [ ] Given a beat whose sentences carry opposing hues, when its beat colour is read,
  then the colour comes from the beat's document embedding and retains non-trivial
  chroma (it is **not** the chroma-cancelling mean of the sentence colours) — a
  falsifiable test that the aggregation trap is avoided.
- [ ] Given the arc, when drift and coherence are computed, then the drift series is
  the `DistanceCalculator` distance between each consecutive beat pair and the
  coherence series is each beat's distance to the whole-book palette, both via the
  injected port (no distance maths in the use case).
- [ ] Given a configured drift threshold, when a beat's coherence exceeds it, then
  that beat appears in `flagged_beats` with its index and value.
- [ ] Given the arc, when a figure is rendered, then lightness/chroma/hue curves, the
  drift series, and the beat-colour swatch strip are drawn at **exact, non-cropped**
  axes and written to `reports/figures/story_compass.png`, with the table written to
  `reports/story_compass.md`.
- [ ] Given the same outline and seed, when the compass runs twice, then every
  recorded number (colours, drift, coherence, flags) is identical (`seed_everything`).
- [ ] Given a checked-in sample outline, when the CLI runs in CI, then the committed
  report is reproducible and `tox` stays green at 100% coverage with no network call.
- [ ] Given an outline containing an empty beat, a whitespace-only beat, or a beat
  below the minimum length, when parsed, then those beats are skipped per the
  documented rules without raising.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

New `domain/model/narrative_arc.py`: frozen `NarrativeBeat(index, title, text)`,
`ArcPoint(beat, colour: LabColor, histogram: ColoredDocument)`, and a `NarrativeArc`
value object holding the ordered points plus `drift_series`, `coherence_series`, and a
`flagged_beats(threshold)` view. `NarrativeArc` reuses `LabColor`
(`domain/model/lab_color.py`) and `ColoredDocument`; its drift/coherence series are
supplied by the application layer (computed through the `DistanceCalculator` port), so
the model stays pure with no framework or I/O dependency. No new domain service — the
existing `DistanceCalculator`, `ColorMapper`, and `FigureRenderer` ports are enough.

### Application Layer (`src/colors_of_meaning/application/`)

New `application/use_case/analyze_narrative_arc_use_case.py`:
`AnalyzeNarrativeArcUseCase(embedding, structured_mapper, codebook,
distance_calculator)` with `execute(beats) -> NarrativeArc`. For each beat it embeds
the whole-beat text (document-level, per the anti-aggregation rule) to read the
lightness/chroma/hue colour, encodes the beat's sentences into a `ColoredDocument`
histogram (mirroring `EncodeDocumentUseCase`), then computes the drift and coherence
series against consecutive beats and the whole-book palette via the port.
`correlation-id` logging. Depends only on abstractions.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No new ML. `MatplotlibFigureRenderer`
(`infrastructure/visualization/matplotlib_figure_renderer.py`) gains
`render_narrative_arc(arc, output_path)` behind a new `FigureRenderer` port method —
a fixed-axis multi-panel plot (lightness/chroma/hue curves, drift series, and a
horizontal strip of the beats' RGB-converted colours). The renderer keeps the domain
port as the boundary, matching how feature 021 added `render_rate_distortion`.

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/compass.py` (tyro `@dataclass`, `main(args)`, `__main__`): reads an
outline file, builds the embedder/structured-mapper/codebook/distance/renderer from a
config or trained artifacts, runs the use case, prints the per-beat drift/coherence
table, writes `reports/story_compass.md`, and renders
`reports/figures/story_compass.png`. New `[testenv:compass]`. The README gains a
"Narrative compass" section with a `tox -e compass -- --outline <file>` example and
the committed sample figure. No API endpoint in this feature.

### Shared Layer (`src/colors_of_meaning/shared/`)

New `shared/outline_parser.py` — pure, no I/O: `parse_outline(text, min_chars) ->
List[NarrativeBeat]`, splitting a Markdown/plain outline into ordered beats (heading-
or blank-line-delimited) and dropping too-short blocks. Framework-free, reused by this
CLI and by feature 024's generator. `shared/synesthetic_config.py` gains a `compass`
section: drift threshold, minimum beat length, mapper choice (structured default,
plain fallback), and report/figure paths.

## API Contracts

No API contract changes. The compass is an offline/CLI planning instrument; the
`POST /query/palette` contract is unaffected. (An arc-analysis endpoint is noted as a
future option in Open Questions, not built here.)

## CLI Impact

One new CLI, `compass`: `--outline` (path to an outline file), `--config`,
`--min-beat-chars`, `--drift-threshold`, `--mapper {structured,plain}`, `--metric
{sliced,jensen_shannon,exact}`, `--output-path` (default `reports/story_compass.md`),
`--figure-path` (default `reports/figures/story_compass.png`). No existing CLI
changes.

## Dependency Injection

The CLI constructs the `SentenceEmbeddingAdapter`, the structured `ColorMapper` and
`ColorCodebook` from trained artifacts, the chosen `DistanceCalculator`, and the
`FigureRenderer`, then injects them into `AnalyzeNarrativeArcUseCase` — matching the
construction style of `encode`/`visualize`. No Lagom container or API wiring changes.

## Observability

`correlation-id` logging in the use case: number of beats, per-beat
`{lightness, chroma, hue}`, the drift/coherence summary, the flagged-beat count, and
the whole-book palette entropy. No new metrics or tracing.

## Open Questions

- **Beat granularity.** Chapter-level vs scene-level beats change the arc's
  resolution. Default: whatever the outline file delimits (one beat per top-level
  section); sub-scene splitting is a future option.
- **Mapper default and fallback.** Default to the structured mapper for interpretable
  axes; fall back to the plain `PyTorchColorMapper` (still a valid colour, no
  axis semantics) when no structured artifact is present. Should a missing structured
  artifact hard-fail instead?
- **Distance metric default.** Sliced-Wasserstein (feature 019) is cheap; exact EMD
  (feature 001) is ~92 ms/call and only affordable for short outlines; Jensen–Shannon
  is the cheapest. Default: sliced, with `--metric` to override.
- **Palette definition.** Uniform mixture over beat histograms vs length-weighted by
  beat word count. Default: uniform (each beat contributes equally to "what the book
  is about"); length-weighting is a future option.
- **API surface.** A `POST /compass/arc` endpoint returning a Pydantic arc DTO is a
  natural extension but is deferred; the compass is CLI-first in this feature.
