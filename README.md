# sae-arabic

Open-source toolkit for training and causally validating Sparse Autoencoder (SAE)
features in Arabic LLMs, using dialectness (ALDi) as external ground truth.

## Status

Phases 1-3 pipeline implemented and verified with offline tests (25 passing)
plus an end-to-end dry run (`scripts/end_to_end_dry_run.py`). ALDi scoring is
pluggable and requires staged ALDi weights (Phase 0).

## Package layout

| Module | Purpose | Phase |
| --- | --- | --- |
| `sae_arabic/sae.py` | Custom linear SAE (encoder/decoder, MSE + L1), training loop, checkpoints | 2 |
| `sae_arabic/activations.py` | MARBERT activation extraction and disk serialization | 1 |
| `sae_arabic/data.py` | Arabic preprocessing, tokenization, NADI/MADAR/local loading | 1 |
| `sae_arabic/analysis.py` | Top-feature extraction and context browsing | 3 |
| `sae_arabic/aldi.py` | ALDi scoring + causal MLM-head reconstruction validation | 3 |
| `scripts/end_to_end_dry_run.py` | Offline wiring smoke test (tiny BERT) | 1-3 |

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

Offline dry run (no network, tiny random BERT):

```bash
python -m scripts.end_to_end_dry_run
```

Real pipeline (MARBERT + NADI/MADAR, 100-500 sentences) runs on Kaggle:
1. `data.py` loads + normalizes the dataset.
2. `activations.py` extracts MARBERT layer activations and writes shards.
3. `sae.py` trains the SAE (`training_config.yaml`).
4. `aldi.py` runs the causal MLM-head scrub once ALDi is staged.

## License

MIT — see [LICENSE](LICENSE).
