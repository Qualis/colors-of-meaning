# Narrative color compass

An ordered outline read as a measured colour trajectory. Each beat's colour is read from its
whole-text (document-level) embedding through the structured mapper -- never the mean of its
sentence colours, which would cancel chroma -- so lightness tracks sentiment, chroma tracks
concreteness, and hue tracks topic. Drift is the perceptual distance between consecutive beat
colour histograms; coherence is each beat's distance to the whole-book palette. Beats whose
coherence exceeds the threshold (36.5, metric: sliced) are flagged.

Library versions: numpy 2.4.6.

Beats: 7. Flagged: 1.

## Per-beat arc

| beat | title | lightness | chroma | hue_deg | coherence | drift_to_next | flagged |
|---|---|---|---|---|---|---|---|
| 0 | Chapter 1 — The Quiet Harbour | 56.69 | 72.43 | 197.7 | 35.6059 | 5.2357 | no |
| 1 | Chapter 2 — The Wreck on the Rocks | 57.53 | 72.26 | 196.6 | 35.7368 | 50.9230 | no |
| 2 | Chapter 3 — The Letter's Secret | 53.11 | 73.35 | 185.3 | 34.4946 | 26.7315 | no |
| 3 | Chapter 4 — The Long Road | 55.77 | 73.22 | 193.3 | 27.1120 | 39.8975 | no |
| 4 | Chapter 5 — Quarterly Earnings Review | 45.30 | 69.71 | 16.0 | 37.6811 | 35.8091 | yes |
| 5 | Chapter 6 — The Capital in Flames | 53.12 | 72.38 | 201.0 | 17.6047 | 32.3883 | no |
| 6 | Chapter 7 — A New Light | 56.66 | 72.18 | 198.9 | 27.7136 | n/a | no |

## Reproduce

```bash
tox -e compass -- --outline outlines/sample_story.md --config configs/structured.yaml --model-path artifacts/models/structured_projector.pth --codebook-name codebook_4096 --mapper structured --metric sliced --min-beat-chars 200 --drift-threshold 36.5
```
