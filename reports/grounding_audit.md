# Answer grounding audit

Each candidate answer is encoded to a colour distribution over the 4,096-colour codebook and
measured against the uniform **mixture** of the retrieved passages' colour distributions (never a
chroma-cancelling mean of Lab colours). The grounding divergence is the perceptual distance between
the answer's colour mass and the retrieved evidence (sliced); answers above the threshold
(25) are flagged as off-context.

**This is a distributional off-context signal, not a factuality or citation checker.** A large
divergence is a strong smell that the answer is topically unsupported by the retrieved passages. A
small divergence does **not** prove the answer is factually correct -- only that its colour
distribution lies within the context's. It inherits the 384->3 lossy bottleneck, so it is a coarse
triage instrument, and the threshold is corpus- and metric-dependent, not a universal constant.

Library versions: numpy 2.4.6.

Question: What causes rain in the water cycle?

Retrieved passages: 3. Candidate answers: 2. Flagged off-context: 1.

## Per-answer grounding

| answer | divergence | threshold | verdict |
|---|---|---|---|
| grounded | 12.2354 | 25.0000 | grounded |
| off_context | 49.5271 | 25.0000 | off-context |

## Reproduce

```bash
tox -e grounding -- --sample samples/rag_grounding.yaml --config configs/base.yaml --model-path artifacts/models/projector.pth --codebook-name codebook_4096 --mapper plain --metric sliced --threshold 25.0
```
