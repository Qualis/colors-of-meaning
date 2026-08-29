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
| color_vq | 3.00 | 117.246167 | ΔE | 0.6450 |
| color_vq | 6.00 | 35.431189 | ΔE | 0.7150 |
| color_vq | 9.00 | 14.128286 | ΔE | 0.7600 |
| color_vq | 12.00 | 6.764355 | ΔE | 0.6300 |
| gzip | 11392.04 | 0.000000 | MSE | n/a |
| pq | 3.00 | 0.002387 | MSE | n/a |
| pq | 6.00 | 0.002372 | MSE | n/a |
| pq | 9.00 | 0.002374 | MSE | n/a |
| pq | 12.00 | 0.002376 | MSE | n/a |

## Matched-budget comparison

| bits/token | method | distortion | metric |
|---|---|---|---|
| 3.00 | color_vq | 117.246167 | ΔE |
| 3.00 | pq | 0.002387 | MSE |
| 6.00 | color_vq | 35.431189 | ΔE |
| 6.00 | pq | 0.002372 | MSE |
| 9.00 | color_vq | 14.128286 | ΔE |
| 9.00 | pq | 0.002374 | MSE |
| 12.00 | color_vq | 6.764355 | ΔE |
| 12.00 | pq | 0.002376 | MSE |

## Pareto frontier

The envelope is the geometric lower-left set over (bits, native distortion). Distortion
metrics differ across codecs (ΔE for color-VQ, MSE for gzip/PQ), so cross-codec domination
is not directly comparable; read each codec's own curve in the figure rather than comparing
ΔE against MSE as if they were one number.

| method | bits/token | distortion | metric |
|---|---|---|---|
| color_vq | 3.00 | 117.246167 | ΔE |
| pq | 3.00 | 0.002387 | MSE |
| color_vq | 6.00 | 35.431189 | ΔE |
| pq | 6.00 | 0.002372 | MSE |
| color_vq | 9.00 | 14.128286 | ΔE |
| color_vq | 12.00 | 6.764355 | ΔE |
| gzip | 11392.04 | 0.000000 | MSE |

## Rate-accuracy diagnosis

The accuracy column above is one distance at one seed. This section re-runs the rate-accuracy axis under
every requested distance and seed at a fixed projector, so a peak that moves with the distance can be told
apart from a peak that belongs to the bit budget. Seeds vary the evaluation sample draw.

| distance | bits/token | mean accuracy | sd | seeds |
|---|---|---|---|---|
| jensen_shannon | 3.00 | 0.6433 | 0.0176 | 3 |
| jensen_shannon | 6.00 | 0.7217 | 0.0161 | 3 |
| jensen_shannon | 9.00 | 0.7600 | 0.0250 | 3 |
| jensen_shannon | 12.00 | 0.6067 | 0.0448 | 3 |
| wasserstein | 3.00 | 0.6483 | 0.0252 | 3 |
| wasserstein | 6.00 | 0.7217 | 0.0126 | 3 |
| wasserstein | 9.00 | 0.7883 | 0.0029 | 3 |
| wasserstein | 12.00 | 0.8050 | 0.0087 | 3 |
| sliced | 3.00 | 0.6450 | 0.0132 | 3 |
| sliced | 6.00 | 0.7150 | 0.0132 | 3 |
| sliced | 9.00 | 0.7833 | 0.0153 | 3 |
| sliced | 12.00 | 0.7883 | 0.0236 | 3 |

Under `jensen_shannon` accuracy peaks at 9.00 bits (0.7600) and reads 0.6067 at 12.00 bits.
Under `wasserstein` accuracy peaks at 12.00 bits (0.8050) and reads 0.8050 at 12.00 bits.
Under `sliced` accuracy peaks at 12.00 bits (0.7883) and reads 0.7883 at 12.00 bits.

The inversion appears only under `jensen_shannon`, so it is a metric artifact rather than a property of the bit budget.

## Reproduce

```bash
tox -e rate_distortion -- --dataset ag_news --budgets 2 4 8 16 --methods color_vq gzip pq --with-accuracy --distance jensen_shannon wasserstein sliced --seeds 42 43 44 --max-samples 200 --config configs/base.yaml --output-path reports/rate_distortion.md --figure-path reports/figures/rate_distortion.png
```
