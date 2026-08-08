# Workshop Paper Outline

Target: an interpretability / Arabic NLP workshop (e.g. *Workshop on Arabic NLP
(WANLP)*, *MechInterp workshop*). 4-6 pages + references.

---

## Title

**Sparse Autoencoders for Arabic Dialectness: A Toolkit and an Honest Causal
Evaluation**

## Abstract (draft)

> We present **sae-arabic**, an open-source toolkit for training and causally
> validating Sparse Autoencoder (SAE) features in Arabic language models, using
> the Arabic Level of Dialectness (ALDi) as external ground truth. On MARBERT,
> our SAE reaches 93% explained variance with 0.08% dead features. Features
> correlate with sentence dialectness (r ≈ 0.2). However, a causal MLM-head
> scrub shows that ablating, amplifying, or randomly ablating features shifts
> ALDi by the same amount (~−0.04), i.e. reconstruction artifacts dominate and
> no feature-specific causal effect is detectable. We argue this null result is
> informative: it highlights both the limits of Sentence-ALDi calibration on
> colloquial text and the confound introduced by hidden-state replacement in
> causal probing. The toolkit, results, and reproducible Colab pipeline are
> publicly released.

## 1. Introduction
- Motivation: Arabic NLP interpretability is under-served; dialect is a salient,
  linguistically meaningful property to probe.
- Contribution summary (3 bullets):
  1. Open toolkit (train, evaluate, layer-sweep, causal scrub) + 29 tests.
  2. Strong SAE training results on MARBERT.
  3. A rigorously-executed causal null result and a documented confound.
- Roadmap sentence.

## 2. Background & Related Work
- SAEs for interpretability (Sparse Autoencoders in LLMs; dictionary learning;
  recent SAE libraries).
- Arabic dialect identification & ALDi (continuous dialectness scoring;
  AOC-ALDi; 3 seeds).
- Causal probing via hidden-state replacement; known confounding by
  reconstruction quality.
- Gap: no public SAE-for-Arabic toolkit; little discussion of ALDi calibration
  limits as a causal metric.

## 3. Methods
- **SAE architecture**: linear encoder, ReLU latents, tied/untied decoder, MSE + L1
  (`sae_arabic/sae.py`).
- **Pipeline**: activation extraction from MARBERT (per-layer), Arabic
  normalization/tokenization (`sae_arabic/data.py`, `activations.py`).
- **Training**: 200 texts, layer 6, dict_mult=8, l1=1e-3, 10k steps, AdamW,
  warmup, grad-norm clip.
- **Layer selection**: time-boxed sweep over layers 4/6/8 (2k steps) ranked by
  explained variance.
- **Causal validation**: MLM-head scrub — mask non-special tokens, replace layer
  hidden states with modified SAE reconstructions, re-predict, re-score with
  ALDi. Interventions: ablate (zero feature), amplify (×5), control (zero a
  random other feature). Seed-averaged ALDi (3 seeds). Control-relative effect
  sizes with paired bootstrap 95% CIs.
- Reproducibility: Colab notebook, `HF_TOKEN`, commands in README.

## 4. Experimental Setup
- Model: MARBERT (UBC-NLP/MARBERT), layer 8 (sweep winner) for causal tests.
- Data: Abdelrahman-Rezk/Arabic_Dialect_Identification (200 train / 500 causal).
- Metric: Sentence-ALDi (3-seed average), clamped [0,1].
- Compute: Colab GPU (T4).

## 5. Results
- **Table 1 — SAE training quality**: explained variance 0.933, dead 0.08%,
  56% features active in >1% of tokens, feature-freq median 1.1%.
- **Table 2 — layer sweep**: L4 0.633, L6 0.601, L8 0.688 (recommended).
- **Table 3 — ALDi calibration check** (MSA vs EGY vs GLF sentences): scores
  compressed (0.2–0.4), incorrect EGY < MSA ordering on colloquial inputs.
- **Table 4 — causal scrub (500 texts)**: baseline 0.448, base −0.007,
  ablate/amplify/control all ≈ −0.043; control-relative effects ±0.001, none
  survive multiple-testing correction.
- **Figure 1**: correlation of top-5 features with ALDi (bar).
- **Figure 2**: control-relative effect sizes with 95% CIs (forest plot) —
  clearly showing CIs spanning/near zero.

## 6. Discussion
- The causal null is the headline: hidden-state replacement shifts ALDi
  uniformly, swamping feature-specific signal. Interpretation: (a) reconstruction
  artifact confound; (b) possible absence of localized dialectness features at
  this granularity; (c) ALDi too coarse/compressed for 500 short texts.
- Honest framing: we report the null rather than tuning for a positive result.
- Limitation: Sentence-ALDi calibration on colloquial text (see Table 3).

## 7. Conclusion & Future Work
- Summary: toolkit + strong training + rigorous null.
- Future: token-level dialectness (Token-DI), feature-specific removal without
  full reconstruction, larger corpora, cross-model transfer.

## 8. Availability
- GitHub: `Yousef13133/sae-arabic` · DOI: 10.5281/zenodo.21848168
- License: MIT.

## Figures/Tables to make
1. Feature-ALDi correlation bar chart (from `causal_results.json`).
2. Forest plot of control-relative effects with bootstrap CIs.
3. Layer sweep bar chart.
4. Sample feature-context rows (top activating texts) if any are interpretable.

## Key numbers to quote
- Explained variance **0.933**; dead features **0.08%**.
- Layer 8 recommended (**0.688**).
- Baseline ALDi **0.448**; any intervention ≈ **−0.043**.
- Control-relative effects **±0.001**, non-significant after correction.
