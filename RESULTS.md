# Results

Final metrics from the real MARBERT run (`UBC-NLP/MARBERT` + the
`Abdelrahman-Rezk/Arabic_Dialect_Identification` dataset). Reproducible via
`notebooks/train_on_colab.ipynb` (requires an `HF_TOKEN` secret).

## SAE training (Layer 6, 10k steps, 200 texts)

| Metric | Value |
| --- | --- |
| Explained variance | **0.933** |
| Reconstruction MSE | 0.077 per dim |
| L1 per token | 131.3 |
| Dead features (<1e-3) | 0.08% |
| Features active in >1% of tokens | 56% |
| Feature frequency median | 1.1% |

Training quality is strong: high explained variance with very few dead
features. (Note: earlier builds reported `explained_variance = -327`; this was
a normalization bug in the metric itself, fixed in `scripts/evaluate.py`.)

## Layer sweep (200 texts, 2k steps per layer)

| Layer | Explained variance | Dead (<1e-3) |
| --- | --- | --- |
| 4 | 0.633 | 0.00% |
| 6 | 0.601 | 0.00% |
| 8 | **0.688** | 0.00% |

Layer 8 is the recommended target.

## ALDi calibration check

| Text | ALDi (expected) | Observed |
| --- | --- | --- |
| MSA: "الطقس جيد اليوم" | ~0 | 0.29 |
| EGY: "الجو حلو النهاردة" | ~0.95 | 0.42 |
| MSA: "الأمطار غزيرة في الشمال اليوم" | ~0 | 0.36 |
| EGY: "إزيك يا باشا عامل ايه النهارده" | ~0.95 | 0.20 |
| GLF: "وش جيك اليوم شكلك متعب واجد" | ~0.5 | 0.37 |

**Finding:** Sentence-ALDi is poorly calibrated on short/colloquial text; it
does not reliably order MSA < dialectal for informal sentences. Scores are
compressed into a narrow band.

## Causal validation (Layer 8, 500 texts, 3-seed-averaged ALDi)

| Quantity | Value |
| --- | --- |
| Baseline ALDi | 0.4479 |
| Base (masked) reconstruction | 0.4410 (−0.007) |
| Ablate top features | −0.043 (mean) |
| Amplify top features | −0.043 (mean) |
| Control (random feature) | −0.043 (mean) |
| Control-relative effects | ±0.001 (negligible) |

**Finding:** every hidden-state intervention (ablate, amplify, and random
control) shifts ALDi by the same ~−0.043. The SAE reconstruction artifact
dominates any feature-specific signal; no feature survives multiple-testing
correction. This is an honest **null result** for the causal hypothesis.

## Summary

- The toolkit trains high-quality SAEs (93% explained variance, ~0% dead).
- SAE features **correlate** with Arabic dialectness (r ≈ 0.16–0.20).
- MLM-head causal scrub cannot isolate **feature-specific** causal effects —
  reconstruction artifacts act as a confound.
- Sentence-ALDi calibration on colloquial text limits its use as a causal
  metric; token-level or better-calibrated scoring is future work.
