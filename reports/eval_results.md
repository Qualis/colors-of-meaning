# Scaled, multi-dataset, matched-budget evaluation

Committed evidence for the color method against the TF-IDF and HNSW baselines at a matched sample
budget and split. Every row is produced by the command below; the sliced-Wasserstein proxy is only
trusted once it clears the fidelity gate.

Library versions: numpy 2.4.6, scipy 1.17.1, POT 0.9.6.post1.

## Distance proxy fidelity gate

| proxy | exact | spearman | accuracy_delta (pts) | pairs | threshold | max_delta | faithful |
|-------|-------|----------|----------------------|-------|-----------|-----------|----------|
| sliced_wasserstein | wasserstein | 0.9924 | 0.0000 | 1500 | 0.95 | 1.0 | yes |

## Results

Classification metrics for every method; label-based MRR and recall@k for the retrieval-capable methods (color, hnsw). TF-IDF is classification-only, so its retrieval cells are skipped and shown as n/a.

| dataset | method | distance | budget | accuracy | macro_f1 | bits/token | mrr | recall@k | seconds |
|---|---|---|---|---|---|---|---|---|---|
| ag_news | color | sliced | 4000 | 0.8175 | 0.8178 | 12.00 | 0.8490 | 1:0.7847 5:0.9360 10:0.9688 | 361.4 |
| ag_news | tfidf | sliced | 4000 | 0.8630 | 0.8624 | n/a | n/a | n/a | 4.6 |
| ag_news | hnsw | sliced | 4000 | 0.8890 | 0.8890 | 12288.00 | 0.9046 | 1:0.8575 5:0.9657 10:0.9830 | 30.9 |
| imdb | color | sliced | 600 | 0.5483 | 0.5478 | 12.00 | 0.7206 | 1:0.5500 5:0.9667 10:0.9983 | 102.8 |
| imdb | tfidf | sliced | 600 | 0.7767 | 0.7765 | n/a | n/a | n/a | 0.8 |
| imdb | hnsw | sliced | 600 | 0.6417 | 0.6265 | 12288.00 | 0.7566 | 1:0.6100 5:0.9500 10:0.9967 | 16.7 |
| newsgroups | color | sliced | 600 | 0.1650 | 0.1535 | 12.00 | 0.2763 | 1:0.1600 5:0.4317 10:0.6067 | 104.6 |
| newsgroups | tfidf | sliced | 600 | 0.4517 | 0.4058 | n/a | n/a | n/a | 2.5 |
| newsgroups | hnsw | sliced | 600 | 0.5867 | 0.5637 | 12288.00 | 0.6363 | 1:0.5150 5:0.8050 10:0.8900 | 13.7 |

## Reproduce

```bash
tox -e eval_suite -- --datasets ag_news imdb newsgroups --methods color tfidf hnsw --distance sliced --budgets 4000 600 600 --fidelity-accuracy-delta 0.0 --config configs/agnews_full.yaml --mapper-type unconstrained --task both --k-values 1 5 10
```
