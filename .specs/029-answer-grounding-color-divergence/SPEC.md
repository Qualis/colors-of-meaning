# Feature: Answer grounding as color-distribution divergence — a cheap RAG faithfulness signal

## Overview

A retrieval-augmented answer is supposed to *stay inside what was retrieved*. This
feature makes that measurable with machinery the project already has: encode the
**answer** and the **retrieved context** as `ColoredDocument` histograms over the
4,096-colour codebook, then measure the perceptual Earth-Mover distance between them
through the existing `DistanceCalculator` port. A small distance means the answer's
semantic-colour mass sits inside the context's; a large distance means the answer has
wandered off the retrieved evidence — a cheap, interpretable **off-context / drift
signal** for RAG.

This is the exact analogue of feature 023's *coherence* reading (each narrative beat
measured against the whole-book palette), pointed at retrieval instead of at a story:
the **answer histogram** is measured against the **mixture of retrieved-passage
histograms**. It reuses the embedder (`SentenceEmbeddingAdapter`), the codebook, the
`EncodeDocumentUseCase` histogram path, and the perceptual EMD / sliced-Wasserstein /
Jensen–Shannon distances (features 001 and 019). **No new third-party dependency is
introduced.** It composes with the retrieval work (features 025 / 026), which is where
the retrieved context comes from in a live system.

**What this is not — stated up front, because the project does not overclaim.** This is
a *distributional* grounding signal, **not a factuality or citation checker**. A large
divergence is a strong smell that the answer is topically or semantically unsupported by
the retrieved passages (a fabricated-topic or hallucinated answer). A small divergence
does **not** prove the answer is factually correct — it proves the answer's colour
distribution lies within the context's. The signal has high recall for *gross* topic
drift and is deliberately blind to subtle factual errors that stay on-topic. It inherits
the 384→3 lossy bottleneck of the projector, so it is a coarse triage instrument, not a
verdict. The threshold is corpus- and metric-dependent (as feature 023's drift threshold
had to be pinned empirically for its committed demo), not a universal constant.

The deliverable is a committed `reports/grounding_audit.md` regenerable from one command
(`tox -e grounding`) over a small checked-in RAG sample (a question, its retrieved
passages, and two candidate answers — one grounded, one off-context), demonstrating the
grounded answer below and the off-context answer above the flagging threshold. Like the
compass, the unit tests are hermetic (fake distance, synthetic histograms, no network);
only the artifact generator loads the real embedder.

## Core Domain Concepts

- **Answer**: the text produced by a generator (RAG answer, summary, LLM completion),
  encoded to a `ColoredDocument` histogram from its sentences. The object under audit.
- **Retrieved context**: the passages a retriever returned for the query. Each passage is
  a `ColoredDocument`; together they define what the answer is allowed to be "about".
- **Context palette**: the uniform mixture of the retrieved-passage histograms — the
  reference distribution the answer's grounding is measured against. A distribution, not
  an average of Lab colours (see the aggregation trap below).
- **Grounding divergence**: the `DistanceCalculator` distance between the answer
  histogram and the context palette — how far the answer's semantic-colour mass sits
  from the retrieved evidence.
- **Grounding verdict**: whether the divergence is within a configured threshold
  (`is_grounded`). Above threshold flags an answer for review.
- **Grounding report**: the `GroundingReport` value object holding the divergence, the
  threshold, and the derived verdict for one answer/context pair.

The per-context reading is a **histogram** (a distribution), and the palette is the
**mixture** of those histograms — never a mean of Lab colours. Averaging Lab across
passages cancels chroma (`sqrt(a² + b²)`) and would silently destroy the concreteness
signal — the documented aggregation trap the corpus and compass work already hit.

## User Stories

- As a RAG engineer, I want each generated answer scored for how far it drifts from the
  passages that were retrieved for it, so I can flag likely off-context answers for
  review before they reach a user.
- As a RAG engineer, I want the flag to come from a single divergence number against a
  configurable threshold, so I can tune sensitivity and wire it into a guardrail.
- As an evaluator, I want to score one answer against the *mixture* of all retrieved
  passages (not just the top one), so multi-passage grounding is measured correctly.
- As a maintainer, I want the audit to reuse the existing embedder, codebook, encode
  path, and distance ports rather than reimplement any of them, and to be
  CI-reproducible from a checked-in sample.
- As a contributor, I want a clean `GroundingReport` object and an
  `EvaluateGroundingUseCase` that a future API guardrail or the book-generation compass
  (feature 024) can call to audit its own output against its sources.

## Acceptance Criteria

- [ ] Given an answer histogram and a context histogram over the **same** codebook, when
  the use case runs, then the returned `GroundingReport.divergence` equals the injected
  `DistanceCalculator` distance between them (no distance maths in the use case), and
  `is_grounded` is true iff `divergence <= threshold`.
- [ ] Given an answer built from the **same** text as the context and a second answer on
  a **different** topic, when both are scored against that context, then the same-topic
  answer has strictly **lower** divergence — the falsifiable directional claim; a noise
  mapper would fail it.
- [ ] Given **multiple** retrieved passages, when the answer is scored against them, then
  it is compared against the **uniform mixture** of the passage histograms (verified by
  capturing the document handed to the port), **not** a chroma-cancelling Lab mean.
- [ ] Given an empty list of retrieved passages, when scored, then the use case raises
  rather than fabricating an empty palette.
- [ ] Given the same answer, contexts, and seed, when the audit runs twice, then every
  recorded number is identical (`seed_everything`).
- [ ] Given the checked-in RAG sample, when `tox -e grounding` runs, then
  `reports/grounding_audit.md` is regenerated with the grounded candidate below and the
  off-context candidate above the flagging threshold, and `tox` stays green at 100%
  coverage with no external-service call.
- [ ] Given a `GroundingReport`, when constructed with a negative divergence or negative
  threshold, then it raises (distances and thresholds are non-negative).

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

New `domain/model/grounding_report.py`: a frozen `GroundingReport(divergence: float,
threshold: float)` value object with an `is_grounded` property (`divergence <=
threshold`) and non-negativity validation in `__post_init__`. Pure; no framework, no
I/O, no distance maths inside the model. Reuses no other domain type directly but is
produced from a `ColoredDocument` comparison in the application layer. No new domain
service — the existing `DistanceCalculator` port (`domain/service/distance_calculator.py`)
is sufficient.

### Application Layer (`src/colors_of_meaning/application/`)

New `application/use_case/evaluate_grounding_use_case.py`:
`EvaluateGroundingUseCase(distance_calculator: DistanceCalculator)` with
`execute(answer: ColoredDocument, context: ColoredDocument, threshold) -> GroundingReport`
and `execute_against_contexts(answer, contexts: List[ColoredDocument], threshold)` which
mixes the passage histograms into a uniform-mixture palette (mirroring
`AnalyzeNarrativeArcUseCase._mix_palette`) and delegates to `execute`. Divergence comes
only from the injected port; the use case does no distance arithmetic. `correlation-id`
logging of `{answer_id, context_id, metric, divergence, threshold, is_grounded}`. Depends
only on abstractions.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No new ML and no new distance calculator — the audit reuses the existing
`WassersteinDistanceCalculator` (exact Lab EMD, feature 001),
`SlicedWassersteinDistanceCalculator` / `JensenShannonDistanceCalculator` (feature 019)
behind the `DistanceCalculator` port. The context7-grounded note below records how the
underlying POT primitives scale to auditing one answer against many contexts; that
scaling is an option for a future corpus-scale audit, not built here.

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/grounding.py` (tyro `@dataclass`, `main(args)`, `__main__`): reads a
checked-in RAG sample (a question, retrieved passages, and candidate answers), builds the
embedder / mapper / codebook / distance calculator from a config or trained artifacts,
encodes the answers and passages to histograms via the `EncodeDocumentUseCase` path, runs
`EvaluateGroundingUseCase`, prints a per-answer divergence/verdict table, and writes
`reports/grounding_audit.md`. New `[testenv:grounding]`. The README gains a "Grounding
audit" section with a `tox -e grounding -- --sample <file>` example and the committed
result. No API endpoint in this feature (a `POST /grounding/audit` guardrail is a natural
extension, deferred — see Open Questions).

### Shared Layer (`src/colors_of_meaning/shared/`)

`shared/synesthetic_config.py` gains an optional `grounding` section: flagging threshold,
distance metric (`sliced` default / `jensen_shannon` / `exact`), sample path, and report
path. No new shared parser is required — the sample is a small structured file the CLI
reads directly.

## API Contracts

No API contract changes. The audit is CLI-first, matching feature 023. A
`POST /grounding/audit` endpoint returning a Pydantic grounding DTO (divergence, threshold,
verdict per answer) is a natural guardrail extension but is deferred and listed as an Open
Question.

## CLI Impact

One new CLI, `grounding`: `--sample` (path to the RAG sample file), `--config`,
`--threshold` (flagging threshold), `--metric {sliced,jensen_shannon,exact}`,
`--output-path` (default `reports/grounding_audit.md`). No existing CLI changes.

## Dependency Injection

The CLI constructs the `SentenceEmbeddingAdapter`, the `ColorMapper` and `ColorCodebook`
from trained artifacts, and the chosen `DistanceCalculator`, then injects the distance
port into `EvaluateGroundingUseCase` — matching the construction style of
`encode` / `visualize` / `compass`. No Lagom container or API wiring changes.

## Observability

`correlation-id` structured logging in the use case: per audited answer, the
`{answer_id, context_id, metric, divergence, threshold, is_grounded}` record, so a
flagged answer is traceable. No new metrics or tracing.

## Distance metric (context7 / POT-grounded)

The divergence reuses the project's existing `DistanceCalculator` implementations; this
records the POT semantics they rest on so the spec is precise:

- Exact grounding divergence is the **W1 Earth-Mover distance** `ot.emd2(answer,
  context, M)` over the perceptual Lab ground cost `M = ot.dist(coords, coords,
  metric='euclidean')` (feature 001) — Euclidean cost, no final root, i.e. W1 not W2.
- For scaling **one answer against many** retrieved sets or many candidate answers, POT's
  entropic-regularised `ot.sinkhorn2(a, b, M, reg)` accepts a **matrix** `b` and returns a
  **vector** of distances in one call — the natural primitive for a corpus-scale audit.
  This is noted as the scaling path, not built in this feature; the default metric here is
  sliced-Wasserstein (feature 019), which is cheap, with exact EMD (~92 ms/call) opt-in
  for small samples.

## Open Questions

- **Threshold calibration.** A fixed absolute threshold vs a percentile of a
  same-corpus divergence distribution vs a per-query relative threshold (answer-to-context
  divergence compared to passage-to-passage divergence within the retrieved set). Default:
  a configurable absolute threshold, pinned empirically for the committed demo and
  documented as calibration, not a universal constant.
- **Palette weighting.** Uniform mixture of retrieved passages vs weighting by retrieval
  score or passage length. Default: uniform (each retrieved passage contributes equally to
  "what was retrieved"); score-weighting is a future option.
- **Answer granularity.** Whole-answer histogram (from the answer's sentences) vs
  per-claim / per-sentence grounding (flag the specific sentence that drifts). Default:
  whole-answer; per-sentence localisation is a future extension.
- **API surface.** A `POST /grounding/audit` guardrail endpoint returning a Pydantic DTO
  is a natural extension but is deferred; the audit is CLI-first in this feature.
- **Scope of the committed feature.** The MVP is the `GroundingReport` value object and
  `EvaluateGroundingUseCase` with their tests (tasks 1–2 of the plan); the CLI, config
  block, and committed `reports/grounding_audit.md` (tasks 3–6) complete it to the repo's
  "committed artifact + generator" convention and may be deferred to a follow-up if a
  smaller first landing is preferred.
