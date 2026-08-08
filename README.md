# sae-arabic

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21848168.svg)](https://doi.org/10.5281/zenodo.21848168)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Yousef13133/sae-arabic/blob/main/notebooks/train_on_colab.ipynb)

Open-source toolkit for training and causally validating Sparse Autoencoder (SAE)
features in Arabic LLMs, using dialectness (ALDi) as external ground truth.

## Status

**v0.1.0** — complete, tested, and released. Phases 1-3 (pipeline, SAE
training, causal validation) are implemented and verified with 29 offline
tests plus a full real-MARBERT run on Colab. ALDi scoring auto-loads the
public `AMR-KELEG/Sentence-ALDi` model (override with the `ALDI_MODEL` env var)
and optionally averages the three published seeds (`--seed-average`).

Final metrics — including the honest causal null result — are in
[RESULTS.md](RESULTS.md).

## Package layout

| Module | Purpose | Phase |
| --- | --- | --- |
| `sae_arabic/sae.py` | Custom linear SAE (encoder/decoder, MSE + L1), training loop, checkpoints | 2 |
| `sae_arabic/activations.py` | MARBERT activation extraction and disk serialization | 1 |
| `sae_arabic/data.py` | Arabic preprocessing, tokenization, dataset loading | 1 |
| `sae_arabic/analysis.py` | Top-feature extraction and context browsing | 3 |
| `sae_arabic/aldi.py` | ALDi scoring (Sentence-ALDi, seed-ensembled) + causal MLM-head scrub + correlation | 3 |
| `scripts/train_real.py` | Real-data entrypoint: dataset -> MARBERT activations -> SAE training | 1-2 |
| `scripts/evaluate.py` | Evaluate a checkpoint: explained variance, dead rate, feature stats | 2-3 |
| `scripts/layer_sweep.py` | Time-boxed layer selection ranked by explained variance | 2 |
| `scripts/causal_validate.py` | Causal scrub: control-relative effects + bootstrap 95% CIs | 3 |
| `scripts/end_to_end_dry_run.py` | Offline wiring smoke test (tiny BERT) | 1-3 |
| `notebooks/train_on_colab.ipynb` | Ready-to-upload Colab GPU notebook (train + evaluate + sweep + causal) | 1-3 |

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Offline dry run (no network, tiny random BERT):

```bash
python -m scripts.end_to_end_dry_run
```

Real pipeline (MARBERT + dialect data, on a Colab GPU):
1. Upload `notebooks/train_on_colab.ipynb` (needs an `HF_TOKEN` secret).
2. `data.py` loads + normalizes the dataset.
3. `activations.py` extracts MARBERT layer activations and writes shards.
4. `sae.py` trains the SAE (`training_config.yaml`) and writes checkpoints + `report.json`.
5. `aldi.py` runs the causal MLM-head scrub with `AMR-KELEG/Sentence-ALDi`.

Locally (small run): `python scripts/train_real.py --num-samples 100 --num-steps 1000`

Evaluate a checkpoint: `python scripts/evaluate.py --checkpoint data/real_run/checkpoints/final.pt --activations-dir data/real_run/activations`

Pick the target layer: `python -m scripts.layer_sweep --layers 4 6 8 --num-samples 200 --num-steps 2000 --out-dir data/sweep`

Run causal validation: `python -m scripts.causal_validate --checkpoint data/sweep/layer8.pt --layer 8 --num-features 5 --seed-average`

## Known limitations

- **Sentence-ALDi calibration**: scores on short/colloquial text are compressed
  and do not reliably order MSA < dialectal (see [RESULTS.md](RESULTS.md)).
- **Causal scrub is null**: hidden-state interventions shift ALDi uniformly
  (~−0.04) regardless of the feature, so feature-specific effects are not
  detectable; reconstruction artifacts act as a confound.

## Citation

```bibtex
@software{sae_arabic,
  title = {sae-arabic: Sparse Autoencoder toolkit for Arabic LLM interpretability},
  author = {Yousef Al-Halabi},
  year = {2026},
  url = {https://github.com/Yousef13133/sae-arabic},
  note = {DOI: 10.5281/zenodo.21848168},
}
```

## License

MIT — see [LICENSE](LICENSE).
