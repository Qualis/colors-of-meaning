# Command reference

Every command runs through a tox environment:

```bash
tox -e <environment> -- [flags]
```

The CLIs are [`tyro`](https://brentyi.github.io/tyro/) dataclasses, so `--help` is generated from
the source and is always authoritative:

```bash
tox -e eval -- --help
```

Flags are the dataclass fields with underscores written as dashes (`max_samples` →
`--max-samples`). Booleans are switches: `--compare-baselines`, `--with-accuracy`,
`--refresh-scaling`, `--deterministic`. List flags take repeated values: `--budgets 2 4 8 16`.

Every environment below is documented with its purpose and principal flags. Defaults are shown in
parentheses; where a default names an artifact, that artifact comes from a previous `train` run.

---

## Core pipeline

### `train`

Train the embedding-to-color projector and build its codebook. Prints the
structure-preservation correlation and, with `--select-on validation`, selects the best
checkpoint on a held-out split.

```bash
tox -e train -- --config configs/base.yaml --dataset-path data/sample_train.txt
tox -e train -- --config configs/agnews_run.yaml --mapper-type supervised
tox -e train -- --config configs/interpretability.yaml --mapper-type structured \
  --output-model artifacts/models/structured_projector.pth --output-codebook interpretability_codebook
```

- `--config` (`configs/base.yaml`) — experiment configuration
- `--mapper-type` (`unconstrained`) — `unconstrained`, `structured`, or `supervised`
- `--dataset-path` (`data/train.txt`) — line-delimited corpus for the mapper types that train on
  free text. The default file is not committed; `data/sample_train.txt` is a five-line smoke run
- `--output-model` (`artifacts/models/projector.pth`), `--output-codebook` (`codebook_4096`)
- `--codebook-mode` (`uniform`) — `uniform` fixed Lab grid, or a learned k-means palette
- `--deterministic` (off) — `torch.use_deterministic_algorithms(True)` for stricter cross-run
  reproducibility, at some cost in speed
- `--select-on` (`structure`) — select the checkpoint on the structure metric or on `validation`
- `--selection-train-samples` (`300`), `--selection-validation-samples` (`150`), `--selection-k` (`5`)
- `--source` (`dataset`), `--documents-dir` (`./documents`), `--min-paragraph-chars` (`200`),
  `--paragraphs-per-work`, `--split-strategy` (`work`), `--validation-fraction` (`0.2`),
  `--test-fraction` (`0.2`) — the authored-corpus source, shared with `eval`, `rate_distortion`,
  `authorship`, and `visualize_documents`

The structured mapper trains a multi-objective loss: angular hue loss, MSE lightness, MSE chroma,
weighted by alpha/beta/gamma from the config. Hue targets are k-means clusters ordered by
centroid so neighbouring hues carry neighbouring meanings; lightness comes from dataset sentiment
labels when `structured_mapper.sentiment_source: labels` is set, and chroma from the bundled
Brysbaert concreteness lexicon. Without those inputs each head falls back to a constant target
and learns no signal. The supervised mapper adds `classification_weight × CrossEntropyLoss` to
the projection loss and discards its classification head after training.

### `encode`

Encode a dataset split to color histograms and persist them for query, comparison, or image
rendering.

```bash
tox -e encode -- --config configs/base.yaml --split test
```

- `--config` (`configs/base.yaml`), `--split` (`test`), `--dataset-path` (`data/test.txt`)
- `--model-path` (`artifacts/models/projector.pth`), `--codebook-name` (`codebook_4096`)
- `--output-path` (`artifacts/encoded/test_documents.pkl`)

### `compare`

Find the nearest documents to one query document in an encoded corpus, by color distance.

```bash
tox -e compare -- --config configs/base.yaml --k 5
```

- `--encoded-documents` (`artifacts/encoded/test_documents.pkl`), `--query-index` (`0`), `--k` (`5`)

### `query`

Retrieve documents matching a color distribution given directly as Lab colors with weights. The
palette is quantized into a histogram and ranked by Wasserstein distance. The same query is
available at `POST /query/palette` when the API is running.

```bash
tox -e query -- \
  --palette-json '[{"l": 75, "a": 20, "b": -10, "weight": 1.0}, {"l": 30, "a": -5, "b": 40, "weight": 0.5}]' \
  --codebook-name codebook_4096 --k 5
```

- `--palette-json` — JSON array of Lab colors: `l` 0–100, `a` and `b` −128–127, optional `weight`
  (default `1.0`)
- `--encoded-documents` (`artifacts/encoded/test_documents.pkl`), `--codebook-name`
  (`codebook_4096`), `--k` (`5`)

### `compress`

Report compression ratio and bits per token for the color path, or compare against gzip and
Product Quantization on raw float32 embeddings.

```bash
tox -e compress -- --config configs/base.yaml
tox -e compress -- --compare-baselines --embeddings-path artifacts/encoded/test_embeddings.npy
```

- `--compare-baselines` (off) — run gzip (lossless) and Product Quantization (M=48 subspaces,
  k=256 centroids), reporting ratio, bits per token, and reconstruction MSE for each
- `--embeddings-path` (`artifacts/encoded/test_embeddings.npy`), `--model-path`, `--codebook-name`

Without `--compare-baselines` the analysis quantizes projector outputs against the fixed uniform
Lab grid by nearest-color assignment — a static palette, not a learned vector-quantization
codebook.

---

## Evaluation

### `eval`

Evaluate one dataset and method, for classification or retrieval.

```bash
tox -e eval -- --dataset ag_news --method tfidf
tox -e eval -- --dataset ag_news --method color --task retrieval --k-values 1 5 10 --config configs/agnews_run.yaml
tox -e eval -- --dataset newsgroups --method hnsw --k-neighbors 10
```

- `--dataset` (`ag_news`) — `ag_news`, `imdb`, `newsgroups`
- `--method` (`color`) — `color`, `tfidf`, `hnsw`
- `--task` (`classification`) — or `retrieval`
- `--distance` (`wasserstein`) — `wasserstein`, `sliced`, `sinkhorn`, `jensen_shannon`
- `--k-neighbors` (`5`), `--k-values` (`1 5 10`) — neighbours for k-NN, cut-offs for recall@k and
  NDCG@k
- `--max-samples`, `--mapper-type` (`unconstrained`), `--model-path`, `--codebook-path`
  (`codebook_4096`)
- the authored-corpus source flags listed under `train`

Retrieval metrics are label-based: relevance is a shared class label, recall@k is "a same-class
document appears in the top k", and MRR is `1 / rank` of the first same-class hit. This is the
standard metric-learning Recall@K convention. `tfidf` is classification-only and is skipped, with
a reason, under `--task retrieval`.

### `eval_suite`

Run the fidelity gate, then evaluate every (dataset, method, budget) cell and write
`reports/eval_results.md`. The gate compares the fast proxy against exact Earth-Mover distance on
held-out pairs and fails the run unless the rank correlation and accuracy delta stay inside their
thresholds.

```bash
tox -e eval_suite -- --datasets ag_news imdb newsgroups --distance sliced \
  --budgets 4000 600 600 --config configs/agnews_full.yaml --mapper-type unconstrained
```

- `--datasets` (`ag_news imdb newsgroups`), `--methods` (`color tfidf hnsw`)
- `--budget` (`4000`) — uniform sample budget; `--budgets` sets one per dataset instead
- `--task` (`classification`) — or `both`, which adds in-suite retrieval metrics
- `--distance` (`sliced`), `--k-values` (`1 5 10`), `--k-neighbors` (`5`)
- `--threshold-spearman` (`0.95`), `--max-accuracy-delta` (`1.0`) — the gate's bounds
- `--fidelity-dataset` (`ag_news`), `--fidelity-samples` (`1000`), `--fidelity-pairs` (`1500`)
- `--output-path` (`reports/eval_results.md`)

### `ablate`

Sweep codebook against distance metric and write the grid to JSON. Wasserstein is
codebook-specific, so a distance calculator is built per cell.

```bash
tox -e ablate -- --dataset ag_news --metrics wasserstein jensen_shannon cosine
```

- `--codebooks` (`grid1024=codebook_1024 grid4096=codebook_4096 learned=codebook_learned`) —
  `label=codebook_name` pairs
- `--metrics` (`wasserstein jensen_shannon cosine`), `--k-neighbors` (`5`)
- `--model-path`, `--mapper-type` (`unconstrained`), `--output-path`
  (`artifacts/ablations/sweep.json`)

### `rate_distortion`

Sweep the bit budget to measure a rate–distortion frontier: color-VQ across grid resolutions
(`bins_per_dimension` ∈ {2, 4, 8, 16} → 3/6/9/12 bits), Product Quantization matched to the same
bits, and gzip as one data-dependent point. Records native distortion — ΔE for color-VQ, MSE for
gzip and PQ — and optionally downstream accuracy at each budget.

```bash
tox -e rate_distortion -- --dataset ag_news --budgets 2 4 8 16 --methods color_vq gzip pq \
  --with-accuracy --distance jensen_shannon --max-samples 200 --config configs/base.yaml
tox -e rate_distortion -- --source documents --documents-dir ./documents --with-accuracy
```

- `--budgets` (`2 4 8 16`) — bins per dimension, not bits
- `--methods` (`color_vq gzip pq`), `--with-accuracy` (off), `--distance` (`wasserstein`;
  repeatable, one of `wasserstein`, `sliced`, `jensen_shannon`), `--seeds` (the configured seed),
  `--k-neighbors` (`5`), `--max-samples` (`400`)
- `--output-path` (`reports/rate_distortion.md`), `--figure-path`
  (`reports/figures/rate_distortion.png`)
- the authored-corpus source flags listed under `train`

Passing more than one `--distance`, or more than one `--seeds`, adds a rate-accuracy diagnosis
section when `--with-accuracy` is also set: the accuracy axis is re-measured under every distance and
seed at a fixed projector, so a
peak that moves with the distance can be told apart from one that belongs to the bit budget. The
first (distance, seed) pair produces the primary tables and the figure; the later sweeps measure the
colour codec alone. A single (distance, seed) cell writes no diagnosis, because one metric cannot
attribute the shape of the curve.

### `compare_objectives`

Train the projector under each candidate structure objective and score every arm on the held-out
Spearman correlation it is actually judged by, with untrained, linear and unconstrained-head
controls bounding what three dimensions can hold. A pre-registered rule — a margin of more than
2 pooled seed standard deviations on |ρ| plus an accuracy guard — decides whether the committed
projector is replaced.

```bash
tox -e compare_objectives -- --dataset ag_news --arms cosine_centred delta_e_correlation \
  margin_ranking --controls noise pca3 unconstrained_head unconstrained_head_preclamp committed \
  --seeds 42 43 44 45 46 47 48 49 --downstream-top-k 2 --downstream-controls committed --budget 4000
```

- `--arms` (`cosine_centred delta_e_correlation margin_ranking`), `--controls` (`noise pca3
  unconstrained_head unconstrained_head_preclamp committed`), `--seeds` (`42`–`49`) — the
  `committed` control reads the shipped projector off disk and scores it on the same held-out slice
- `--downstream-arms` (empty), `--downstream-top-k` (`2`), `--downstream-controls` (`committed`),
  `--downstream-seeds` (`42 43 44`), `--budget` (`4000`), `--distance` (`sliced`), `--k-neighbors` (`5`),
  `--k-values` (`1 5 10`) — the nominated arms are the baseline plus the strongest challengers, and every
  listed control is measured downstream too, so the decision can be read against the shipped artifact
- `--train-samples` (the configured `dataset.max_samples`), `--selection-samples` (`256`),
  `--structure-samples` (`256`) — the selection and structure slices are disjoint halves of one
  held-out draw from the test split
- `--adoption-threshold-sigma` (`2.0`), `--max-accuracy-drop` (`0.01`), `--model-dir`
  (`artifacts/objectives`), `--codebook-path` (`codebook_4096`), `--committed-model-path`
  (`artifacts/models/projector.pth`)
- `--output-path` (`reports/structure_objective.md`), `--figure-path`
  (`reports/figures/structure_objective.png`)

### `interpretability`

Measure hue↔topic, lightness↔sentiment, and chroma↔concreteness on a held-out split for the
structured mapper and for a negative control, and write a per-axis verdict. An axis whose margin
over the control misses its threshold is reported as a falsification.

```bash
tox -e interpretability -- --config configs/interpretability.yaml --dataset imdb \
  --structured-model artifacts/models/structured_projector.pth --control noise
```

- `--structured-model` (`artifacts/models/structured_projector.pth`)
- `--control` (`noise`) — an untrained noise projector, or `unconstrained` for a mapper never
  trained toward these axes; `--control-model` (`artifacts/models/projector.pth`)
- `--hue-topic-margin` (`0.02`), `--lightness-sentiment-margin` (`0.05`),
  `--chroma-concreteness-margin` (`0.05`)
- `--max-samples` (`500`), `--output-path` (`reports/interpretability.md`)

### `authorship`

Write `reports/documents_authorship.md` from a real held-out train and evaluation over the
authored corpus. The multi-seed data-scaling table is read from the committed manifest unless
`--refresh-scaling` re-runs the sweep.

```bash
tox -e authorship -- --config configs/documents.yaml \
  --model-path artifacts/models/projector_documents_valsel.pth \
  --codebook-name codebook_documents_valsel --mapper-type supervised --paragraphs-per-work 60
```

- `--methods` (`color tfidf`), `--distance` (`jensen_shannon`), `--k-neighbors` (`5`)
- `--refresh-scaling` (off) — re-run the seeds × caps training sweep instead of reading the
  manifest. This is the expensive path; see the benchmark note in the README
- `--scaling-seeds` (`8`), `--scaling-caps` (`60 150 300`), `--scaling-workers` (`1`),
  `--scaling-manifest` (`reports/data/authorship_scaling.json`),
  `--scaling-scratch-dir` (`artifacts/scaling_sweep`)
- `--paragraphs-per-work` (`60`), `--output-path` (`reports/documents_authorship.md`)

---

## Visualization

### `visualize`

Render pipeline figures to `--output-dir`.

```bash
tox -e visualize -- --visualization-type codebook
tox -e visualize -- --visualization-type histograms --dataset ag_news --max-samples 50
tox -e visualize -- --visualization-type projection --dataset ag_news --max-samples 200
tox -e visualize -- --visualization-type confusion_matrix --dataset ag_news --method color
```

- `--visualization-type` (`codebook`) — `codebook`, `histograms`, `projection`,
  `confusion_matrix`
- `--dataset` (`ag_news`), `--method` (`color`), `--k-neighbors` (`5`), `--max-samples` (`500`)
- `--model-path`, `--codebook-name`, `--output-dir` (`reports/figures`)

### `visualize_corpus`

Compare several labelled text corpora by color: a per-corpus mean color signature and a shared
t-SNE with points colored by corpus. Writes `corpus_color_signatures.png` and
`corpus_color_tsne.png`. Project Gutenberg boilerplate is stripped and paragraphs are sampled
from the body.

```bash
tox -e visualize_corpus -- \
  --corpus-specs "Darwin=data/darwin.txt,Smith=data/smith.txt,Austen=data/austen.txt" \
  --paragraphs-per-corpus 60
```

- `--corpus-specs` (`sample=data/sample_train.txt`) — comma-separated `label=path` pairs
- `--paragraphs-per-corpus` (`60`), `--min-paragraph-chars` (`200`), `--top-colors` (`24`)
- `--model-path`, `--codebook-name`, `--output-dir` (`reports/figures`)

### `visualize_documents`

Regenerate the authored-corpus figures: the author t-SNE, per-author color signatures, one A4
sheet per book under `reports/figures/a4/<author>__<work>.png`, and the contact-sheet gallery.
Output is deterministic and byte-identical across runs.

```bash
tox -e visualize_documents -- --config configs/documents.yaml \
  --model-path artifacts/models/projector_documents_valsel.pth \
  --codebook-name codebook_documents_valsel
```

- `--figure-split` (`train`), `--max-figure-samples` (`2000`), `--leading-paragraphs` (`40`)
- `--columns` (`12`), `--top-colors` (`24`), `--dpi` (`150`), `--output-dir` (`reports/figures`)
- the authored-corpus source flags listed under `train`

### `visualize_lossless`

Tile the lossless barcode pages into a contact sheet and render the semantic-versus-lossless
comparison for one book.

```bash
tox -e visualize_lossless -- --book darwin__origin_of_species
```

- `--book` (`darwin__origin_of_species`) — `<author>__<work>` key, `--title` overrides the caption
- `--lossless-dir` (`reports/figures/lossless`), `--sheets-dir` (`reports/figures/a4`),
  `--columns` (`12`)

---

## Document images

### `encode_image` / `decode_image`

Render one document as a printable A4 "colors of meaning" page — 2480×3508 px at 300 DPI with
embedded DPI metadata — using the genuine `Lab → RGB` of each codebook bin.

```bash
tox -e encode_image -- --text "Markets fell sharply today. Investors fled to bonds." --layout score
tox -e encode_image -- --dataset-path data/sample_test.txt --index 0 --layout signature
tox -e decode_image -- --image-path reports/figures/document_a4.png \
  --encoded-documents artifacts/encoded/test_documents.pkl --k 5
```

- `--layout` (`score`) — `score` is one perceptual cell per sentence in reading order and is
  decodable; `signature` is proportional bands; `mosaic` is a frequency-tinted codebook grid
- `--text` takes precedence over `--dataset-path` (`data/sample_test.txt`) with `--index` (`0`)
- `--dpi` (`300`), `--output-path` (`reports/figures/document_a4.png`)
- decode: `--image-path`, `--codebook-name`, `--encoded-documents` (none — omit to print only the
  recovered histogram summary), `--k` (`5`)

Decoding re-samples each cell and re-quantizes it to recover a histogram. Because the codebook
includes out-of-gamut Lab points that `lab_to_rgb` clips, the round-trip is approximate on the
full 4,096-color codebook and exact on a coarse in-gamut one — enough to retrieve the source
document by color distance. The source text is deliberately not recoverable: the 384→3 projection
is lossy by design.

### `encode_lossless` / `decode_lossless`

The lossless counterpart, and a different thing entirely: each cell's color is a data symbol, not
a meaning. `text → UTF-8 → DEFLATE → CRC-framed pages → palette cells → A4 PNG`, with no
embedding, projector, or Lab codebook involved, so no trained artifacts are needed and the exact
bytes come back.

```bash
tox -e encode_lossless -- --text "The exact bytes of this sentence are recoverable."
tox -e encode_lossless -- --input-path reports/austen_pride.txt --output-path reports/figures/austen.png
tox -e decode_lossless -- --input-paths "reports/figures/austen_p*.png" --output-path reports/austen_recovered.txt
```

- `--text` takes precedence over `--input-path`, which is read whole as one document
- `--output-path` (`reports/figures/document_exact.png`) — multi-page runs insert `_p01`, `_p02`
  before the extension
- `--dpi` (`300`), `--cell-size` (`4`) — pixels per module; smaller is denser and less
  print-robust, and must match between encode and decode
- decode: `--input-paths` (`reports/figures/document_exact.png`) takes one path, a comma-separated
  list, or a glob; `--output-path` omitted echoes the text to stdout

A document larger than one page is split across the minimum number of pages. Each page
self-describes its `(page_index, page_count)` and carries a CRC32, so pages decode in any order
and a missing, duplicated, or corrupted page raises rather than returning wrong text. Each page
fills the whole sheet: `--cell-size` fixes the column count and the rows adapt to the content, and
the decoder reads the row count from the page's header row.

---

## Narrative and grounding

### `compass`

Read an ordered story outline — headings or blank lines delimit beats — as a color trajectory.
Each beat's color comes from its whole-text embedding, so lightness tracks sentiment, chroma
tracks concreteness, and hue tracks topic; each beat is also rendered as a histogram to measure
drift between consecutive beats and coherence against the whole-book palette. Beats over the
threshold are flagged. Never calls an external service.

```bash
tox -e compass -- --outline outlines/sample_story.md --config configs/structured.yaml \
  --model-path artifacts/models/structured_projector.pth --codebook-name codebook_4096 \
  --mapper structured --metric sliced --min-beat-chars 200 --drift-threshold 36.5
```

- `--outline` (`outlines/sample_story.md`), `--min-beat-chars` (`200`)
- `--mapper` (`structured`) — or `plain`; `--metric` (`sliced`) — `sliced`, `jensen_shannon`,
  `exact`
- `--drift-threshold` (`25.0`) — metric- and corpus-dependent; the committed sample uses `36.5`
- `--output-path` (`reports/story_compass.md`), `--figure-path`
  (`reports/figures/story_compass.png`)

### `grounding`

Audit a retrieval-augmented answer for drift from its retrieved context. The answer and each
passage are encoded as color histograms and the answer's perceptual distance from the uniform
mixture of the passage histograms is measured. The context palette is a mixture of distributions,
never a mean of Lab colors, which would cancel chroma. Never calls an external service.

```bash
tox -e grounding -- --sample samples/rag_grounding.yaml --config configs/base.yaml \
  --model-path artifacts/models/projector.pth --codebook-name codebook_4096 \
  --mapper plain --metric sliced --threshold 25
```

- `--sample` (`samples/rag_grounding.yaml`) — question, retrieved passages, and candidate answers
- `--threshold` (`25.0`), `--mapper` (`plain`), `--metric` (`sliced`)
- `--output-path` (`reports/grounding_audit.md`)

A large divergence means the answer is topically unsupported by the retrieved passages. A small
one does not certify factual correctness — only that the answer's color distribution lies within
the context's.

### `generate`

Write a book with a language model and audit it with the compass. From a one-paragraph summary
the use case asks the generator for a structured outline, writes each chapter with the preceding
ones in context, and runs the compass over the beats so far after each one. A chapter that drifts
past the threshold is regenerated with a corrective note within the retry budget, and recorded as
a flagged beat if it still drifts.

**Needs a live `ANTHROPIC_API_KEY` and is not run in CI.** The unit suite mocks the SDK client
entirely, and a consumer-driven contract test pins the request shape.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
tox -e generate -- --summary "A lighthouse keeper on a fog-bound coast befriends the ghost of a drowned sailor." \
  --num-beats 6 --words-per-beat 500 --tone "sombre, tender" \
  --config configs/structured.yaml --model-path artifacts/models/structured_projector.pth
```

- `--summary`, or `--summary-file` to read it from disk
- `--num-beats` (`8`), `--words-per-beat` (`400`), `--tone` (empty)
- `--retry-budget` (`2`), `--drift-threshold` (`25.0`)
- `--model` (`claude-opus-4-8`), `--effort` (`high`)
- `--mapper` (`structured`), `--metric` (`sliced`)
- `--output-dir` (`reports/book`), `--figure-path` (`reports/figures/story_book.png`)

Chapters are written to `reports/book/NN_<title>.md` and the compass audit to
`reports/book/arc.md`. The generator is reached through a provider-agnostic `TextGenerator` port,
so the domain and application layers stay provider-agnostic. The adapter plans the outline with
structured outputs and streams each chapter, caching the summary-and-outline context so per-beat
calls pay only for the delta. Refusals and typed API errors surface as a domain-level
`GenerationError`.

---

## Utilities

### `cli`

The `colors-of-meaning-cli` entry point declared in `pyproject.toml`. It is a scaffold placeholder
that prints its `--message` argument and is not part of the pipeline.

---

## Not commands

`tox` with no environment runs the quality gates and the test suite. `tox -e format` formats with
ruff, `tox -e watch` re-runs affected tests on change, and `clean`, `build`, and `publish` handle
packaging. `./bin/generate` regenerates every artifact under `reports/` by invoking the
environments above with each report's own committed Reproduce command; `./bin/run-local -c` runs
the API.
