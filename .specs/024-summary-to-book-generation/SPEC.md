# Feature: Summary-to-book generation — an LLM writes, the color compass audits

## Overview

The project's own network cannot generate prose — it is a lossy
embedding→Lab encoder with no decoder (established in feature 023's overview). Writing
a comprehensible book from a summary requires the opposite kind of model: an
**autoregressive language model**. This feature adds one behind a clean port and pairs
it with the narrative colour compass (feature 023) so that **an LLM writes the book and
the colour pipeline plans and audits it** — the honest division of labour, and the one
role in which this project's differentiated machinery is genuinely useful around a
generator.

The flow: a `GenerateBookUseCase` takes a `BookSpecification` (a summary plus target
shape) and (1) asks a `TextGenerator` for a structured **outline** of beats, (2)
generates each beat's prose in order with the preceding context, and (3) after each
beat, runs `AnalyzeNarrativeArcUseCase` (feature 023) over the beats so far to measure
**coherence drift**; a beat whose colour signature clashes with the emerging book
palette beyond a threshold is **regenerated once or twice with a corrective note**
(bounded retry budget) or flagged. The output is a `GeneratedBook` — ordered chapters
plus the final `NarrativeArc` report and figure.

The generator is the Anthropic Claude API, reached through the official `anthropic`
Python SDK behind a `TextGenerator` domain port, so the domain and application layers
stay provider-agnostic. Per the project's model guidance the adapter defaults to
**`claude-opus-4-8`** with **adaptive thinking** (`thinking={"type": "adaptive"}`) and
`output_config={"effort": "high"}`; the outline is produced with **structured outputs**
(`messages.parse` / `output_config.format`) so it is a validated list of beat briefs,
and each beat's prose is produced with **streaming**
(`client.messages.stream(...).get_final_message()`) because long, high-`max_tokens`
output must stream to avoid SDK HTTP timeouts. Refusals (`stop_reason == "refusal"`)
are handled before content is read, and API errors use the SDK's typed-exception
chain.

This is the **first feature to introduce a third-party runtime dependency** (`anthropic`)
and the **first that cannot run in CI** — generation needs a live API key. So the unit
suite mocks the SDK client (no network, consumer-driven contract test on the request
shape), CI never calls Anthropic, and any generated book is a local/opt-in artifact,
not a committed CI-reproducible one. The colour-QA loop, by contrast, reuses feature
023 unchanged and is fully testable with a stub generator.

## Core Domain Concepts

- **Book specification**: the input — a summary/premise, the target number of beats,
  target words per beat, an optional tone, and optional per-beat colour targets
  (desired lightness/chroma/hue as a tone spec). Frozen value object.
- **Beat brief**: one planned beat from the outline — index, title, synopsis, and
  optional target colour. The unit the generator expands into prose and the compass
  measures.
- **Text generator (port)**: the provider-agnostic capability to `generate_outline` and
  `generate_beat_prose`. A domain abstraction with **no SDK import**, so the business
  logic never depends on Anthropic.
- **Coherence-drift QA**: after generating a beat, the compass measures its colour
  signature against the emerging book palette; beats beyond the drift threshold are
  regenerated with a corrective note or flagged. The compass is the audit instrument,
  not the author.
- **Corrective note**: the guidance fed back to the generator on a drift-triggered
  retry (e.g. "stay closer to the established sombre, concrete tone of the preceding
  chapters"), turning the colour axes into a control signal.
- **Generated book**: the ordered chapters (`BeatBrief` + prose), the final
  `NarrativeArc`, and generation metadata (tokens, retries, flagged beats).

## User Stories

- As a writer, I want to give a one-paragraph summary and get back a full, ordered,
  comprehensible draft book, so I can go from premise to draft in one command.
- As a writer, I want the tool to plan an outline first and then write each chapter
  with the earlier chapters in context, so the book has structure and continuity.
- As a writer, I want chapters that drift off the book's established tone/topic to be
  caught and regenerated automatically, so the draft stays coherent without manual
  review of every chapter.
- As a writer, I want to steer a scene's tone with an explicit colour target (sombre /
  concrete / a given topic hue), so the colour axes are a controllable spec.
- As a maintainer, I want the LLM reached through a `TextGenerator` port so the domain
  and application layers never import the SDK, the provider is swappable, and the unit
  tests run with no network via a mocked client.

## Acceptance Criteria

- [ ] Given a `BookSpecification`, when generation runs, then the `TextGenerator`
  produces a validated outline of the requested number of `BeatBrief`s (structured
  output), and one chapter of prose is generated per brief in order.
- [ ] Given each beat, when its prose is generated, then the call includes the
  preceding chapters as context so continuity is maintained.
- [ ] Given a generated beat whose coherence drift (feature 023) exceeds the
  configured threshold, when QA runs, then the beat is regenerated with a corrective
  note up to the retry budget, and if still over threshold it is recorded in the
  book's flagged beats rather than silently accepted.
- [ ] Given the assembled book, when generation completes, then a `GeneratedBook` is
  returned with ordered chapters, the final `NarrativeArc`, and metadata (tokens,
  retries, flagged beats), and the arc figure is rendered via feature 023's renderer.
- [ ] Given the adapter, when it calls Claude, then it uses `claude-opus-4-8` with
  adaptive thinking, requests the outline via structured outputs, and streams prose via
  `messages.stream().get_final_message()` — asserted by a consumer contract test
  against a **mocked** SDK client (no network).
- [ ] Given the API returns `stop_reason == "refusal"`, when the adapter reads the
  response, then it surfaces a domain-level generation error before indexing
  `content`, rather than raising an IndexError.
- [ ] Given `tox` runs, then all eight quality gates pass at 100% coverage with **no
  network call and no API key**, the `anthropic` dependency passes `pip-audit`, and the
  domain/application layers contain no `anthropic` import.
- [ ] Given no live API key, when the unit suite runs, then every generator test uses a
  stub/mock; a real end-to-end book is an opt-in local run, not a CI artifact.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

New `domain/service/text_generator.py`: `TextGenerator` ABC with
`generate_outline(spec) -> list[BeatBrief]` and `generate_beat_prose(brief,
preceding_context, corrective_note) -> str` — **pure interface, no SDK import**. New
models `domain/model/book_specification.py` (frozen `BookSpecification`, `BeatBrief`
with optional target colour) and `domain/model/generated_book.py`
(`GeneratedChapter(brief, prose)`, `GeneratedBook(spec, chapters, arc, metadata)`).
`GeneratedBook` reuses feature 023's `NarrativeArc`. No framework dependency; the
domain never learns which provider writes the text.

### Application Layer (`src/colors_of_meaning/application/`)

New `application/use_case/generate_book_use_case.py`:
`GenerateBookUseCase(text_generator, analyze_narrative_arc_use_case)` with
`execute(spec) -> GeneratedBook`. It orchestrates outline → per-beat prose (with
preceding context) → compass QA → bounded corrective regeneration → assembly. It
depends only on the `TextGenerator` port and feature 023's use case — no SDK, no I/O.
A tracing decorator wraps `execute`; `correlation-id` logging records the outline size,
per-beat token spend, retries, and flags.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

New `infrastructure/generation/anthropic_text_generator_adapter.py` implementing
`TextGenerator` with the `anthropic` SDK. Client built from the environment
(`Anthropic()` resolves `ANTHROPIC_API_KEY` or an `ant` profile — never a hardcoded
key). Model `claude-opus-4-8`, `thinking={"type": "adaptive"}`,
`output_config={"effort": "high"}`. Outline via structured outputs (`messages.parse`
with a Pydantic `BeatBrief` schema / `output_config.format`); prose via
`client.messages.stream(...).get_final_message()` with a generous streaming
`max_tokens`. The stable book "bible" (summary + outline) is sent once with
`cache_control` prompt caching so per-beat calls only pay for the delta. Handles
`stop_reason == "refusal"` (surfaced as a generation error), and wraps the SDK's typed
exceptions (`RateLimitError` → `APIStatusError` → `APIConnectionError`); the SDK's
built-in retry covers 429/5xx. This is the only layer that imports `anthropic`.

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/generate.py` (tyro `@dataclass`, `main(args)`, `__main__`): reads a
summary (arg or file) and shape flags, builds the adapter + feature 023 pipeline + use
case, writes chapters to `reports/book/` plus the arc report and figure. New
`[testenv:generate]`. Optionally a `interface/api/controller/generation_controller.py`
with Pydantic DTOs (`GenerateBookRequest`, `GeneratedBookResponse`) — but generation is
long-running, so the synchronous endpoint is marked a future/async option in Open
Questions and the CLI is the primary surface. README gains a "Generate a book" section
noting the required API key and that generation is not run in CI.

### Shared Layer (`src/colors_of_meaning/shared/`)

`shared/synesthetic_config.py` gains a `generation` section: model id, effort,
streaming `max_tokens`, number of beats, target words per beat, retry budget, and the
drift threshold (shared with feature 023's `compass` config). `setup.py`/`setup.cfg`
gain the `anthropic` runtime dependency.

## API Contracts

No change to the existing `POST /query/palette` contract. If the optional generation
endpoint is built, it MUST return a Pydantic `GeneratedBookResponse` DTO (never a raw
dict) with a producer contract test — but it is deferred; the CLI is the contract-
bearing surface in this feature. The **consumer** contract is the Claude Messages API
request/response shape, pinned by a CDCT test against a mocked client.

## CLI Impact

One new CLI, `generate`: `--summary` / `--summary-file`, `--num-beats`,
`--words-per-beat`, `--tone`, `--retry-budget`, `--drift-threshold`, `--model`
(default `claude-opus-4-8`), `--effort`, `--output-dir` (default `reports/book`),
`--figure-path`. No existing CLI changes.

## Dependency Injection

The CLI constructs the `AnthropicTextGeneratorAdapter` and injects it as the
`TextGenerator` port into `GenerateBookUseCase`, alongside feature 023's
`AnalyzeNarrativeArcUseCase` (itself built from the embedder/mapper/codebook/distance
adapters). Tests inject a stub `TextGenerator`; production injects the Anthropic
adapter. If wired into the API, the Lagom container binds `TextGenerator ->
AnthropicTextGeneratorAdapter` once at import, mirroring the existing container setup.

## Observability

`correlation-id` logging across generation: outline size, per-beat
`{tokens_in, tokens_out, retries, drift, flagged}`, and a final summary (total tokens,
total retries, flagged beats). Metrics: counters for beats generated / retried /
flagged and a histogram of per-beat latency and token spend. A tracing decorator wraps
the use case so a book generation is one traceable request flow. No secret is ever
logged.

## Open Questions

- **Retry policy on persistent drift.** Default: regenerate up to the retry budget with
  a corrective note, then flag and keep the best attempt. Should a persistently
  drifting beat instead re-plan the outline? Deferred.
- **Colour targets as hard vs soft constraints.** Default: soft — target colours become
  corrective-note guidance, not a rejection gate, since the mapper is a lossy read.
- **Synchronous vs async API endpoint.** A full book can take minutes; a synchronous
  `POST /generate` risks HTTP timeouts. Default: CLI-first; if exposed, the endpoint
  should be job-based (submit → poll), not synchronous.
- **Coherence QA cadence.** Per-beat (catches drift early, more compass calls) vs once
  at the end (cheaper, later). Default: per-beat, since early correction is the point.
- **Committed artifact.** Because generation needs a live key, no generated book is
  committed or reproduced in CI; feature 023's compass report (from a checked-in
  outline) remains the CI-reproducible colour artifact.
- **Provider swap.** The `TextGenerator` port admits other backends; only the Anthropic
  adapter is built here, and the model-specific request shape lives entirely in it.
