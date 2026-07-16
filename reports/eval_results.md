# Scaled, multi-dataset evaluation of the color method

Committed evidence for the color method beyond the 400-sample AG News budget. Every row is produced
by the command below; the sliced-Wasserstein proxy is only trusted once it clears the fidelity gate.

Library versions: numpy 2.4.6, scipy 1.17.1, POT 0.9.6.post1.

## Distance proxy fidelity gate

| proxy | exact | spearman | accuracy_delta (pts) | pairs | threshold | max_delta | faithful |
|-------|-------|----------|----------------------|-------|-----------|-----------|----------|
| sliced_wasserstein | wasserstein | 0.9924 | 0.0000 | 1500 | 0.95 | 1.0 | yes |

## Results

Classification metrics only. Retrieval quality (label-based MRR and recall@k) is measured separately via `eval --task retrieval` and is not reported as a constant here.

| dataset | method | distance | budget | accuracy | macro_f1 | bits/token | seconds |
|---------|--------|----------|--------|----------|----------|------------|---------|
| ag_news | color | sliced | 4000 | 0.8175 | 0.8178 | 12.00 | 807.3 |
| imdb | color | sliced | 600 | 0.5483 | 0.5478 | 12.00 | 200.4 |
| newsgroups | color | sliced | 600 | 0.1650 | 0.1535 | 12.00 | 199.7 |

## Reproduce

```bash
tox -e eval_suite -- --datasets ag_news imdb newsgroups --distance sliced --budgets 4000 600 600 --fidelity-accuracy-delta 0.0 --config configs/agnews_full.yaml --mapper-type unconstrained
```
