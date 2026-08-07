# sae-arabic

Open-source toolkit for training and causally validating Sparse Autoencoder (SAE)
features in Arabic LLMs, using dialectness (ALDi) as external ground truth.

## Status

Phases 1-3 pipeline implemented and verified with offline tests (28 passing)
plus an end-to-end dry run (`scripts/end_to_end_dry_run.py`). ALDi scoring
auto-loads the public `AMR-KELEG/Sentence-ALDi` model (override with the
`ALDI_MODEL` env var).

## Package layout

| Module | Purpose | Phase |
| --- | --- | --- |
| `sae_arabic/sae.py` | Custom linear SAE (encoder/decoder, MSE + L1), training loop, checkpoints | 2 |
| `sae_arabic/activations.py` | MARBERT activation extraction and disk serialization | 1 |
| `sae_arabic/data.py` | Arabic preprocessing, tokenization, NADI/MADAR/local loading | 1 |
| `sae_arabic/analysis.py` | Top-feature extraction and context browsing | 3 |
| `sae_arabic/aldi.py` | ALDi scoring (Sentence-ALDi) + causal MLM-head scrub + correlation | 3 |
| `scripts/train_real.py` | Real-data entrypoint: dataset -> MARBERT activations -> SAE training | 1-2 |
| `scripts/evaluate.py` | Evaluate a checkpoint: explained variance, dead rate, feature stats | 2-3 |
| `scripts/layer_sweep.py` | Time-boxed layer selection (e.g., 4/6/8) ranked by explained variance | 2 |
| `scripts/end_to_end_dry_run.py` | Offline wiring smoke test (tiny BERT) | 1-3 |
| `notebooks/train_on_kaggle.ipynb` | Ready-to-upload Kaggle GPU notebook | 1-2 |

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Offline dry run (no network, tiny random BERT):

```bash
python -m scripts.end_to_end_dry_run
```

Real pipeline (MARBERT + dialect data, on a Kaggle/Colab GPU):
1. Upload `notebooks/train_on_colab.ipynb` (needs `HF_TOKEN` secret).
2. `data.py` loads + normalizes the dataset.
3. `activations.py` extracts MARBERT layer activations and writes shards.
4. `sae.py` trains the SAE (`training_config.yaml`) and writes checkpoints + `report.json`.
5. `aldi.py` runs the causal MLM-head scrub with `AMR-KELEG/Sentence-ALDi`.

Locally (small run): `python scripts/train_real.py --num-samples 100 --num-steps 1000`

Evaluate a checkpoint: `python scripts/evaluate.py --checkpoint data/real_run/checkpoints/final.pt --activations-dir data/real_run/activations`

Pick the target layer: `python scripts/layer_sweep.py --layers 4 6 8 --num-samples 200 --num-steps 2000 --out-dir data/sweep`

## License

MIT — see [LICENSE](LICENSE).
