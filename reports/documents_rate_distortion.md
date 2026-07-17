# Rate-distortion frontier for semantic color compression

The ~1024:1 headline is one operating point; this report measures the whole frontier.
Each codec is swept across bit budgets and its native distortion recorded: color-VQ over
grid resolutions (bits = log2(bins)), Product Quantization over subquantizers matched to the
same bits, and gzip as a single data-dependent point. The color codec additionally records a
downstream retrieval accuracy at each budget, so the cost of compression is shown in both
perceptual distortion (ΔE for color-VQ, MSE for gzip/PQ) and task accuracy at matched budgets.

Library versions: numpy 2.4.6, scikit-learn 1.9.0.

## Rate-distortion points

| method | bits/token | distortion | metric | accuracy |
|---|---|---|---|---|
| color_vq | 3.00 | 143.283005 | ΔE | 0.0600 |
| color_vq | 6.00 | 34.451002 | ΔE | 0.0400 |
| color_vq | 9.00 | 14.389268 | ΔE | 0.0533 |
| color_vq | 12.00 | 7.011115 | ΔE | 0.0700 |
| gzip | 11394.27 | 0.000000 | MSE | n/a |
| pq | 3.00 | 0.001943 | MSE | n/a |
| pq | 6.00 | 0.001937 | MSE | n/a |
| pq | 9.00 | 0.001929 | MSE | n/a |
| pq | 12.00 | 0.001898 | MSE | n/a |

## Matched-budget comparison

| bits/token | method | distortion | metric |
|---|---|---|---|
| 3.00 | color_vq | 143.283005 | ΔE |
| 3.00 | pq | 0.001943 | MSE |
| 6.00 | color_vq | 34.451002 | ΔE |
| 6.00 | pq | 0.001937 | MSE |
| 9.00 | color_vq | 14.389268 | ΔE |
| 9.00 | pq | 0.001929 | MSE |
| 12.00 | color_vq | 7.011115 | ΔE |
| 12.00 | pq | 0.001898 | MSE |

## Pareto frontier

The envelope is the geometric lower-left set over (bits, native distortion). Distortion
metrics differ across codecs (ΔE for color-VQ, MSE for gzip/PQ), so cross-codec domination
is not directly comparable; read each codec's own curve in the figure rather than comparing
ΔE against MSE as if they were one number.

| method | bits/token | distortion | metric |
|---|---|---|---|
| color_vq | 3.00 | 143.283005 | ΔE |
| pq | 3.00 | 0.001943 | MSE |
| color_vq | 6.00 | 34.451002 | ΔE |
| pq | 6.00 | 0.001937 | MSE |
| color_vq | 9.00 | 14.389268 | ΔE |
| pq | 9.00 | 0.001929 | MSE |
| color_vq | 12.00 | 7.011115 | ΔE |
| pq | 12.00 | 0.001898 | MSE |
| gzip | 11394.27 | 0.000000 | MSE |

## Reproduce

```bash
tox -e rate_distortion -- --source documents --documents-dir documents --split-strategy work --min-paragraph-chars 200 --paragraphs-per-work 60 --validation-fraction 0.2 --test-fraction 0.2 --budgets 2 4 8 16 --methods color_vq gzip pq --with-accuracy --distance jensen_shannon --max-samples 300 --config configs/documents.yaml
```
