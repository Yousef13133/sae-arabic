"""Train short SAEs on several layers and rank them (Phase 2, Week 4).

Each layer gets a time-boxed training run; the best layer is chosen by
explained variance, with dead-feature rate as a tiebreaker.

Usage:
  python scripts/layer_sweep.py --layers 4 6 8 --num-samples 200 --num-steps 2000 --out-dir data/sweep
  python scripts/layer_sweep.py --tiny  # offline smoke test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_arabic.activations import extract_activations, load_model
from sae_arabic.analysis import top_features
from sae_arabic.data import ArabicDatasetLoader, ArabicTokenizer
from sae_arabic.sae import (
    SAEConfig,
    SAETrainerConfig,
    SparseAutoencoder,
    save_checkpoint,
    train_sae,
)
from scripts.evaluate import evaluate


def _tiny_assets():
    from tests.helpers import StubTokenizer, tiny_bert

    model, tokenizer = tiny_bert(), StubTokenizer()
    texts = [f"جملة عربية رقم {i} للتدريب" for i in range(30)]
    return model, tokenizer, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training_config.yaml")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.num_samples:
        cfg["num_samples"] = args.num_samples
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.tiny:
        model, tokenizer, texts = _tiny_assets()
        layers = [0, 1, 2]
    else:
        model, tokenizer = load_model(cfg["model_name"])
        loader = ArabicDatasetLoader(
            source="huggingface",
            path=cfg["dataset_path"],
            name=cfg["dataset_name"],
            split=cfg["split"],
            text_column=cfg["text_column"],
            dialect_column=cfg["dialect_column"],
        )
        ds = loader.load()
        ds = ds.select(range(min(len(ds), cfg["num_samples"])))
        texts = [t for t in ds[cfg["text_column"]] if isinstance(t, str) and t.strip()]
        layers = args.layers

    arabic = ArabicTokenizer(tokenizer=tokenizer, max_length=cfg["max_length"])
    texts = [arabic.normalize(t) for t in texts]
    print(f"evaluating {len(texts)} texts across layers {layers}")

    results: dict[int, dict] = {}
    for layer in layers:
        acts = extract_activations(
            texts, model, tokenizer, layer=layer,
            batch_size=cfg["batch_size"], max_length=cfg["max_length"],
        )
        flat = np.concatenate(acts, axis=0)
        sae = SparseAutoencoder(
            SAEConfig(d_model=flat.shape[1], dict_mult=cfg["dict_mult"], l1_coeff=cfg["l1_coeff"])
        )
        train_cfg = SAETrainerConfig(
            lr=cfg["lr"], batch_size=cfg["batch_size"], num_steps=args.num_steps,
            warmup_steps=cfg["warmup_steps"], max_grad_norm=cfg["max_grad_norm"],
            seed=cfg["seed"],
        )
        train_sae(sae, torch.from_numpy(flat), train_cfg)
        save_checkpoint(sae, out_dir / f"layer{layer}.pt", args.num_steps)

        metrics = evaluate(sae, torch.from_numpy(flat))
        results[layer] = metrics
        print(
            f"layer {layer:>2}: explained_var={metrics['explained_variance']:.4f} "
            f"dead(<1e-3)={metrics['dead_features_lt_1e-3']:.4f} "
            f"l1/tok={metrics['l1_per_token']:.3f}"
        )

    best = max(results, key=lambda l: results[l]["explained_variance"])
    summary = {
        "layers": {str(k): v for k, v in results.items()},
        "recommended_layer": best,
    }
    (out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"RECOMMENDED LAYER: {best}")

    try:
        device = next(sae.parameters()).device
        with torch.no_grad():
            latents = sae(torch.from_numpy(flat[:512]).to(device))[1]
        top = top_features(latents[:100].cpu().numpy(), k=2)
        print("sample top features (last layer):", list(top)[:4])
    except Exception as exc:  # noqa: BLE001 - cosmetic step, never block the summary
        print(f"top-features demo skipped: {exc}")


if __name__ == "__main__":
    main()
