# Plan: Summary-to-book generation — an LLM writes, the color compass audits

## Implementation Strategy

Add text generation behind a **provider-agnostic `TextGenerator` port** and orchestrate
it with feature 023's narrative compass so the LLM writes and the colour pipeline
audits. The domain and application layers never import the SDK; only one infrastructure
adapter knows it is Claude. The use case is a bounded loop: plan an outline → write each
beat with preceding context → measure coherence drift with the compass → regenerate a
drifting beat with a corrective note up to a budget → assemble a `GeneratedBook` with
its final `NarrativeArc`.

The adapter follows the project's model guidance: `claude-opus-4-8`, adaptive thinking,
`effort: high`; the **outline** via structured outputs (a validated list of
`BeatBrief`s, so parsing can't drift), and **prose** via streaming
(`messages.stream().get_final_message()`) because long high-`max_tokens` output must
stream to avoid HTTP timeouts. The stable summary + outline is sent once under prompt
caching so per-beat calls pay only for the delta. Refusals and typed API errors are
handled explicitly.

Three decisions keep it clean and testable:

1. **Port in the domain, SDK in one adapter.** `TextGenerator` is a pure ABC; the
   `anthropic` import lives only in `infrastructure/generation/`. Architecture tests pin
   this — the domain/application layers stay provider-free.
2. **Compass QA reuses feature 023 unchanged.** The generator consumes
   `AnalyzeNarrativeArcUseCase`; the drift threshold is shared config. The audit is not
   reimplemented.
3. **No network in tests, no CI generation.** Every unit test mocks the SDK client; a
   CDCT test pins the request shape; the QA loop is exercised with a stub generator.
   A real book is a local, opt-in run — never a committed CI artifact.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/service/text_generator.py`: `TextGenerator` ABC — `generate_outline(spec)`,
  `generate_beat_prose(brief, preceding_context, corrective_note)`. No SDK import.
- `domain/model/book_specification.py`: frozen `BookSpecification` (summary, num_beats,
  words_per_beat, tone, optional palette targets) and `BeatBrief` (index, title,
  synopsis, optional target colour).
- `domain/model/generated_book.py`: `GeneratedChapter(brief, prose)` and
  `GeneratedBook(spec, chapters, arc, metadata)`, reusing feature 023's `NarrativeArc`.

### Application Layer (`src/colors_of_meaning/application/`)

- `application/use_case/generate_book_use_case.py`:
  `GenerateBookUseCase(text_generator, analyze_narrative_arc_use_case).execute(spec) ->
  GeneratedBook`. Orchestrates outline → per-beat prose with accumulating context →
  compass QA → bounded corrective regeneration → assembly. Depends only on the port and
  feature 023's use case. Tracing decorator + `correlation-id` logging.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/generation/anthropic_text_generator_adapter.py`: implements
  `TextGenerator` with the `anthropic` SDK. `Anthropic()` from the environment;
  `claude-opus-4-8`; `thinking={"type": "adaptive"}`; `output_config={"effort":
  "high"}`. Outline via `messages.parse` against a Pydantic `BeatBrief` schema; prose
  via `messages.stream(...).get_final_message()` with a generous streaming
  `max_tokens`; summary + outline cached with `cache_control`. Guards
  `stop_reason == "refusal"` (→ domain error) before reading `content`; maps the SDK's
  typed exceptions to domain errors; relies on the SDK's built-in 429/5xx retry. The
  only module importing `anthropic`.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/generate.py`: tyro `@dataclass`; builds the adapter + feature 023
  pipeline + use case; writes `reports/book/NN_<title>.md` chapters, the arc report,
  and the figure.
- `tox.ini`: new `[testenv:generate]`.
- (Optional, deferred) `interface/api/controller/generation_controller.py` +
  `interface/api/data_transfer_object/generation_dto.py` (Pydantic
  `GenerateBookRequest` / `GeneratedBookResponse`) with a producer contract test — only
  if the endpoint is built; the CLI ships first.
- `README.MD`: "Generate a book" section (required `ANTHROPIC_API_KEY`, not run in CI).

### Shared Layer (`src/colors_of_meaning/shared/`)

- `shared/synesthetic_config.py`: add a `generation` section (model, effort, streaming
  `max_tokens`, num_beats, words_per_beat, retry budget, shared drift threshold).
- `setup.py` / `setup.cfg`: add the `anthropic` runtime dependency (scanned by
  `pip-audit`).

## Dependency Injection

The CLI constructs `AnthropicTextGeneratorAdapter` and injects it as the
`TextGenerator` into `GenerateBookUseCase`, together with feature 023's
`AnalyzeNarrativeArcUseCase`. Unit tests inject a stub `TextGenerator` and a real (or
lightly mocked) compass use case. If the API endpoint is built, the Lagom container
binds `TextGenerator -> AnthropicTextGeneratorAdapter` once at import, matching the
existing container wiring.

## Task List

1. [ ] domain: `text_generator.py` port, `book_specification.py`, `generated_book.py`
   + tests (frozen models; `GeneratedBook` composes a `NarrativeArc`).
2. [ ] application: `GenerateBookUseCase` + tests with a **stub** `TextGenerator` and
   the real compass over synthetic beats — assert outline→prose ordering, context
   accumulation, drift-triggered regeneration within budget, and flagging of a beat
   that stays over threshold.
3. [ ] infrastructure: `AnthropicTextGeneratorAdapter` + CDCT tests against a **mocked**
   `anthropic.Anthropic` — assert model `claude-opus-4-8`, adaptive thinking, structured
   outline request, streaming prose call, prompt-cache on the bible, refusal handling,
   and typed-exception mapping. No network.
4. [ ] shared/config: `generation` config section + `anthropic` dependency; tests.
5. [ ] interface: `generate` CLI + tests (mock adapter + use case); new
   `[testenv:generate]`; README section (key required, not in CI).
6. [ ] architecture tests: `TextGenerator` lives in domain with no `anthropic` import;
   the adapter (infrastructure) implements it; the use case depends only on the port and
   the compass; `anthropic` is imported nowhere in domain/application.
7. [ ] (optional) API controller + Pydantic DTOs + producer contract test, only if the
   endpoint is in scope.
8. [ ] run `tox`; confirm 8 gates + 100% coverage with no network/key; `pip-audit`
   clean on `anthropic`.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...` names,
`assert_that` for the entity/DTO tests, plain `assert`/`pytest.raises` for the
adapter/use-case paths, **no network and no API key in any test**, hermetic `tmp_path`
for file output. Key tests:

- **Domain models:** `BookSpecification`/`BeatBrief`/`GeneratedBook` are frozen and
  compose a `NarrativeArc`.
- **Use case (stub generator):** beats generated in outline order; each prose call
  receives the accumulated preceding context; a beat mocked to drift is regenerated with
  a corrective note up to the budget; a beat that stays over threshold lands in flagged
  beats; metadata totals tokens/retries.
- **CDCT consumer contract (mocked SDK):** the adapter sends `claude-opus-4-8`, adaptive
  thinking, a structured-output outline request, and a streaming prose request; the
  mocked stream's `get_final_message()` is parsed into prose; the bible carries
  `cache_control`.
- **Refusal / errors:** a mocked `stop_reason == "refusal"` yields a domain error before
  `content` is indexed; a mocked `RateLimitError` / `APIStatusError` /
  `APIConnectionError` maps to the right domain error via the typed chain.
- **CLI:** builds the pipeline from args and writes chapters + report + figure (mock
  adapter + use case).
- **Architecture:** no `anthropic` import outside `infrastructure/generation/`; the port
  is in domain; the use case depends only on abstractions.

## Observability Plan

`correlation-id` logging across generation: `{outline_size, per_beat:{tokens_in,
tokens_out, retries, drift, flagged}, totals:{tokens, retries, flagged}}`. Metrics:
counters for beats generated/retried/flagged; histograms for per-beat latency and token
spend. Tracing decorator on `execute` so one book is one request flow. Secrets are never
logged.

## Risks and Mitigations

- **Layer leakage** (SDK bleeding into domain/application). Mitigation: `TextGenerator`
  port in domain; `anthropic` imported only in `infrastructure/generation/`;
  architecture tests fail on any other import.
- **No CI reproducibility** (generation needs a live key). Mitigation: mock the SDK in
  every test; CDCT pins the request shape; no generated book is committed or asserted in
  CI; the compass report (feature 023) remains the reproducible colour artifact.
- **Refusal / index errors** (`stop_reason == "refusal"` returns empty content).
  Mitigation: check `stop_reason` before reading `content`; a test drives the refusal
  path.
- **Timeouts on long output** (non-streaming high `max_tokens` hits SDK timeouts).
  Mitigation: prose always streams via `get_final_message()`; a synchronous API endpoint
  is deferred in favour of a job-based shape.
- **Cost / token blow-up** (per-beat context grows). Mitigation: prompt-cache the stable
  bible; bound the retry budget; log per-beat token spend; `words_per_beat` and
  `num_beats` cap the work.
- **Dependency risk** (new `anthropic` runtime dep). Mitigation: `pip-audit` in `tox`;
  pin via the project's normal dependency policy; the port keeps the provider swappable.
- **Incoherent drafts despite QA** (colour drift is a coarse proxy, not a plot checker).
  Mitigation: the compass is honest QA on tone/topic signature, not a guarantee of
  narrative quality; flagged beats are surfaced, not hidden, and colour targets are soft
  guidance — stated plainly in the spec's Open Questions.
