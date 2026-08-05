# Authorship by colour: training, evaluating and testing against `documents/`

This report trains the semantic-colour projector on a real authored-document corpus and
measures how much authorship signal survives compressing each paragraph's meaning into a
distribution over a 4,096-colour palette (12 bits/token). It is generated from the local,
git-ignored `./documents/` corpus and is **not** CI-reproducible; regenerate it locally with
the command at the end. The computed tables below are machine-written; only the prose is fixed.

Library versions: numpy 2.4.6, scikit-learn 1.9.0.

## Corpus

`documents/<author>/<work>.txt` — 22 authors, 133 works. The author is the class label; works are stripped of Gutenberg boilerplate, split into paragraphs,
globally de-duplicated, and partitioned by a work-level three-way holdout.

| author | works |
|---|---|
| austen | 6 |
| carroll | 6 |
| conrad | 6 |
| darwin | 6 |
| dickens | 7 |
| dostoevsky | 6 |
| doyle | 7 |
| eliot | 6 |
| hardy | 6 |
| hawthorne | 6 |
| kipling | 6 |
| melville | 6 |
| shelley | 6 |
| smith | 3 |
| stevenson | 6 |
| stoker | 6 |
| thoreau | 6 |
| tolstoy | 6 |
| twain | 7 |
| verne | 6 |
| wells | 7 |
| wilde | 6 |

Per-split paragraph counts at the `paragraphs_per_work` cap of 60:

| split | paragraphs |
|---|---|
| train | 5340 |
| validation | 1258 |
| test | 1320 |

## Held-out test results

Authorship on the held-out test works (chance = 1/22 = 0.0455). The colour method is a supervised author-contrastive projector; TF-IDF keeps every word.

| method | bits/token | test accuracy | macro-F1 |
|---|---|---|---|
| color | 12 | 0.1083 | 0.1061 |
| tfidf | — | 0.3720 | 0.3467 |
| chance | — | 0.0455 | — |

## The texts as colours

Each book's leading paragraphs, encoded to a colour distribution by the trained projector and
projected to 2-D — authors settle into colour neighbourhoods:

![Document colour t-SNE](figures/documents_color_tsne.png)

Per-author colour signatures (the dominant palette colours for each author):

![Document colour signatures](figures/documents_color_signatures.png)

### A4 colour image per book

Each work is rendered as an A4 colours-of-meaning sheet — horizontal bands of the book's palette
colours, sized by how often the projector maps the book's sentences to each colour (the `signature`
layout over the book's leading paragraphs). The contact sheet below tiles all of them, ordered by
author, so an author's works sit together (the per-book sheets are the `figures/a4/`
`<author>__<work>.png` files):

![Per-book A4 colour signatures](figures/documents_a4_gallery.png)

These figures regenerate deterministically (byte-identical across runs) with:

```bash
tox -e visualize_documents -- --config configs/documents.yaml --model-path artifacts/models/projector_documents_valsel.pth --codebook-name codebook_documents_valsel --mapper-type supervised
```

### Lossless A4 colour-barcode representation

The signatures above are a *lossy, semantic* rendering. Separately, the lossless codec
(`encode_lossless` / `decode_lossless`) stores each book's **exact** text as printable A4
colour-barcode page(s) that decode back byte-for-byte. Those barcode images are dense data and
are **not** committed — they are git-ignored, regenerable local artifacts.

## Data scaling: does more training data help?

The `paragraphs_per_work` cap controls how much of each book enters training. Training the
val-selected supervised projector at increasing caps and measuring held-out authorship accuracy
on the same fixed test set (multiple seeds each; mean ± population std):

| paragraphs_per_work | train paragraphs | test accuracy (mean ± std) | × chance |
|---|---|---|---|
| 60 | 5340 | 0.127 ± 0.016 | 2.79× |
| 150 | 13339 | 0.141 ± 0.012 | 3.10× |
| 300 | 25452 | 0.183 ± 0.018 | 4.03× |

## Reproduce (local; requires `./documents/`)

```bash
tox -e authorship -- --config configs/documents.yaml --model-path artifacts/models/projector_documents_valsel.pth --codebook-name codebook_documents_valsel --mapper-type supervised --documents-dir ./documents --paragraphs-per-work 60 --distance jensen_shannon
```
