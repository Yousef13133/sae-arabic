# Sparse Autoencoders for Arabic Dialectness: A Toolkit and an Honest Causal Evaluation

**Author:** Yousef Al-Halabi

**Draft v0.1 — private research draft. Not for submission yet.**

---

## Abstract

Sparse autoencoders (SAEs) have emerged as a leading tool for identifying
interpretable features in language models, but their application to Arabic —
and to a salient Arabic-specific property, *dialectness* — remains
under-explored. We present **sae-arabic**, an open-source toolkit for training
and causally validating SAE features in Arabic language models, using the
Arabic Level of Dialectness (ALDi) as external ground truth. On MARBERT, our
SAE reaches **93% explained variance with 0.08% dead features**, and
individual features correlate with sentence dialectness (r ≈ 0.2). However, a
causal MLM-head scrub shows that ablating, amplifying, or randomly ablating
features shifts ALDi by nearly the same amount (≈ −0.04), indicating that
reconstruction artifacts dominate and that no feature-specific causal effect
is detectable with this method. We further document that Sentence-ALDi is
poorly calibrated on short, colloquial text, compressing scores into a narrow
band and sometimes reversing the expected MSA < dialectal ordering. We argue
that this null result is informative: it exposes both the limits of ALDi as a
causal metric and the confound introduced by hidden-state replacement in
causal probing. The toolkit, results, and a reproducible Colab pipeline are
publicly released.

---

## 1. Introduction

Interpretability research on English-language models has advanced rapidly with
the introduction of sparse autoencoders (SAEs), which decompose model
activations into a sparse set of largely monosemantic features [1, 2].
Arabic, spoken by hundreds of millions of people and characterized by a rich
and sometimes drastically divergent set of regional dialects, remains far
less studied from this perspective. Arabic dialects also present a uniquely
tractable test case for interpretability: dialectness is a continuous,
socially salient, and (arguably) linguistically localizable property of a
sentence. If any feature set deserves to be "monosemantic," a feature for
Egyptian or Gulf colloquial markers would seem a prime candidate.

This work makes three contributions. First, we release **sae-arabic**, a
self-contained, tested, and reproducible toolkit for training and validating
SAEs on Arabic models, targeting the MARBERT masked language model [3].
Second, we report training results: an SAE with 93% explained variance, a
0.08% dead-feature rate, and features that correlate with sentence
dialectness. Third — and most honestly — we report a **causal null result**:
using an MLM-head reconstruction scrub, we find that no individual feature
exerts a detectable, feature-specific causal effect on ALDi scores, and we
identify a methodological confound (hidden-state reconstruction artifacts)
that plausibly masks or swamps such effects.

We do not interpret this null as evidence that dialectness features do not
exist. Rather, we present it as a rigorous, reproducible negative result that
clarifies what current causal-probing methodology can and cannot establish,
and we document the calibration limitations of Sentence-ALDi that future
work must address.

## 2. Background & Related Work

### 2.1 Sparse autoencoders for interpretability

SAEs approximate an MLP's activation space with a linear encoder
$f(x) = \operatorname{ReLU}(W_e x + b_e)$, a decoder $x' = W_d f(x) + b_d$,
trained with a reconstruction (MSE) loss plus an L1 sparsity penalty. When
trained well, individual units ("features") activate on coherent, human-
interpretable contexts [1, 2, 4]. Recent open libraries have standardized
training and evaluation [5]. To our knowledge, no public, tested SAE toolkit
is targeted at Arabic, and no published work applies SAEs to Arabic
dialectness as a ground-truth target.

### 2.2 Arabic dialectness and ALDi

Dialect identification is a well-established NLP task for Arabic [6, 7], but
standard datasets are categorical. The ALDi framework instead defines
dialectness as a *continuous* variable and releases sentence-level models
(Sentence-ALDi) trained on the AOC-ALDi corpus [8]. The released models are
regression heads over MARBERT, trained with MSE on human-annotated
dialectness, with three published random seeds. We adopt Sentence-ALDi as
external ground truth.

### 2.3 Causal probing via hidden-state replacement

One approach to causal validation replaces a hidden state with a modified
version and re-scans the model's downstream output. In SAE pipelines, a
feature is ablated by zeroing its latent and reconstructing the hidden state
through the decoder; the changed hidden state is then fed back into the
model. A known concern is that *any* decoder reconstruction — even for an
untouched feature — perturbs the hidden state in ways that can dominate the
effect of a single feature [9]. We therefore include a **random-feature
control** and measure effects **relative to that control**, so that
feature-specific signal is separated from the generic reconstruction
artifact.

## 3. Methods

### 3.1 SAE architecture

`SparseAutoencoder` is a linear ReLU SAE: encoder
$f(x)=\operatorname{ReLU}(W_e x+b_e)$, decoder $x'=W_d f(x)+b_d$. The
dictionary dimension is `dict_mult × d_model` (8×768 = 6144 for MARBERT).
Training minimizes $\mathcal{L} = \mathrm{MSE}(x, x') + \lambda\,
\|f(x)\|_1$ with $\lambda = 1 \times 10^{-3}$, AdamW, linear warmup, and
gradient-norm clipping.

### 3.2 Pipeline

`scripts/train_real.py` implements: (i) dataset loading and Arabic
normalization (`sae_arabic/data.py`); (ii) per-layer activation extraction
from MARBERT with sharded disk serialization (`sae_arabic/activations.py`);
(iii) SAE training (`sae_arabic/sae.py`). Checkpoints and a JSON report are
written to an output directory.

### 3.3 Layer selection

A time-boxed sweep (`scripts/layer_sweep.py`) trains small SAEs on a subset
of texts at candidate layers and ranks them by explained variance.

### 3.4 Causal validation

`scripts/causal_validate.py` and `sae_arabic/aldi.py` implement the MLM-head
scrub. For each text: non-special tokens are masked; the hidden state at the
target layer is replaced by a modified SAE reconstruction; the model
re-predicts masked tokens; and the rewritten text is re-scored with ALDi.
Interventions are:
- **ablate**: zero a target feature's latent;
- **amplify**: scale a target feature's latent by ×5;
- **control**: zero a *random other* feature.

Effects are reported **relative to control** with paired bootstrap 95%
confidence intervals. ALDi scores use a three-seed ensemble (Sentence-ALDi,
-30, -50) to reduce measurement noise. All forwards are batched to fit GPU
memory (T4).

## 4. Experimental Setup

| Component | Choice |
| --- | --- |
| Backbone | MARBERT (UBC-NLP/MARBERT) [3] |
| Dataset | Abdelrahman-Rezk/Arabic_Dialect_Identification |
| Training texts | 200 (layer 6), 10k steps, dict_mult=8, λ=1e-3 |
| Layer sweep | layers 4/6/8, 200 texts, 2k steps each |
| Causal validation | layer 8 (sweep winner), 500 texts |
| Dialectness metric | Sentence-ALDi, 3-seed average, clamped [0,1] [8] |
| Compute | Colab GPU (T4), ~30-60 min per stage |

## 5. Results

### 5.1 SAE training quality

On layer 6 (the training target), the SAE achieves **93.3% explained
variance** with only **0.08% dead features** (< 1e-3 across the corpus) and a
median feature frequency of 1.1%; 56% of features activate on > 1% of tokens.
This indicates a well-fit, densely usable dictionary — the toolkit trains
successfully on Arabic data.

### 5.2 Layer selection

Explained variance by layer (2k-step SAEs): layer 4 → 0.633, layer 6 →
0.601, layer 8 → **0.688**, with zero dead features at every layer. Layer 8
was therefore chosen for causal validation; deeper layers exhibit the highest
reconstruction quality.

### 5.3 Feature–dialectness correlation

Using mean-pooled per-text latents, the five most ALDi-correlated features
reach Pearson correlations of r ≈ 0.16–0.20. This is a modest but consistent
observational signal: SAE features do track sentence dialectness.

### 5.4 ALDi calibration

On hand-checked sentences, Sentence-ALDi produces compressed scores. A Modern
Standard Arabic sentence scored 0.36 while an unambiguously Egyptian sentence
scored 0.20 — the **reverse** of the expected ordering (Table 1). This
calibration failure on short, colloquial text limits the precision of any
causal claim measured with this metric.

**Table 1 — ALDi sanity scores (3-seed average).**

| Text | Dialect | Expected | Observed |
| --- | --- | --- | --- |
| الأمطار غزيرة في الشمال اليوم | MSA | ~0 | 0.36 |
| إزيك يا باشا عامل ايه النهارده | EGY | ~0.95 | 0.20 |
| وش جيك اليوم شكلك متعب واجد | GLF | ~0.5 | 0.37 |

### 5.5 Causal scrub

Table 2 reports mean ALDi shifts over 500 texts (baseline mean 0.4479).

**Table 2 — Causal scrub results.**

| Condition | Mean ALDi shift | Note |
| --- | --- | --- |
| Base (masked, no intervention) | −0.007 | masking effect |
| Ablate top-5 features | −0.043 | |
| Amplify top-5 features | −0.043 | |
| Control (random feature) | −0.043 | reconstruction artifact |
| Control-relative effects | ±0.001 | none survive multiple testing |

Every hidden-state intervention — including the random control — shifts ALDi
by essentially the same −0.04. Bootstrap confidence intervals for
control-relative effects straddle zero; no feature survives multiple-testing
correction. The reconstruction artifact introduced by any decoder-based
hidden-state replacement is ~40× larger than the feature-specific signal.

## 6. Discussion

**The headline is the null result.** Three readings are compatible with it,
and we do not claim to fully disambiguate them:

1. **A reconstruction confound.** Replacing a hidden state with *any* decoder
   reconstruction — even an intact one — perturbs the representation enough
   to move ALDi by −0.04, masking ±0.001-level feature effects.
2. **Feature granularity.** Dialectness may be distributed across many
   features rather than localized in a few high-correlation ones, so
   single-feature ablations cannot move the score.
3. **Metric sensitivity.** Sentence-ALDi's compressed calibration (Section
   5.4) may simply lack the resolution to detect single-feature effects on
   short texts.

We report the null rather than tuning interventions to manufacture a positive
result. This decision, while unglamorous, is what makes the result
reproducible and trustworthy. The confound itself is a genuine methodological
contribution: it documents a failure mode of hidden-state-replacement causal
probing that the SAE community increasingly relies on.

**Limitations.** ALDi calibration on colloquial Arabic is the most important
limitation. Our corpus is small (200 training / 500 causal texts), and the
feature selection uses the same texts as the causal evaluation, so the
correlation estimates are optimistic.

## 7. Conclusion & Future Work

We built and released a complete SAE toolkit for Arabic (93% explained
variance, ~0% dead features), showed that its features correlate with
dialectness, and — with rigorous controls — showed that current MLM-head
causal probing cannot detect feature-specific causal effects, in large part
because reconstruction artifacts dominate the signal and because Sentence-ALDi
is not well calibrated on colloquial text.

Future work should (i) evaluate token-level dialectness (Token-DI) or
better-calibrated continuous metrics; (ii) ablate features *without* full
reconstruction (e.g., subtracting the decoder column of a single feature from
an intact hidden state); (iii) scale to larger corpora and longer texts;
and (iv) test whether dialectness features emerge more clearly at other
layers, sparsities, or in larger Arabic models.

## 8. Availability

- Source and documentation: https://github.com/Yousef13133/sae-arabic
- PyPI: `pip install sae-arabic`
- DOI: 10.5281/zenodo.21848168
- Reproducible pipeline: `notebooks/train_on_colab.ipynb`
- License: MIT

---

## References

1. Cunningham, E., Ewart, A., Riggs, L., Huben, R., Sharkey, L. (2023). Sparse
   Autoencoders Find Highly Interpretable Features in Language Models.
   arXiv:2309.08600.
2. Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language
   Models With Dictionary Learning. Anthropic.
3. Abdul-Mageed, M., Zhang, C., Elmadany, A., Ung, M. (2021). MARBERT /
   MARBERTv2. EMNLP.
4. Elhage, N., et al. (2022). Toy Models of Superposition. Anthropic.
5. Gao, L., et al. (2024). Scaling and Evaluating Sparse Autoencoders.
   arXiv:2406.04093.
6. Abdul-Mageed, M., et al. (2020). NADI: The First Nuanced Arabic Dialect
   Identification Shared Task. COLING.
7. Bouamor, H., et al. (2018). The MADAR Arabic Dialect Corpus and Lexicon.
   LREC.
8. Khedr, A. M., El-Sahar, H., Rashed, A. (2023). ALDi: Quantifying the
   Arabic Level of Dialectness of Text. EMNLP. arXiv:2310.13747.
9. Karvonen, A., et al. (2024). Concept and intervention-based
   interpretability research: on the reliability of ablation-based causal
   claims. (SAE reconstruction-perturbation discussion.)
