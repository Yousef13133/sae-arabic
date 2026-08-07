"""Real-data training entrypoint (Phase 1 dry run + Phase 2).

Loads an Arabic dialect dataset, extracts MARBERT layer activations, trains
the SAE, and writes shards + checkpoints + a report. Designed to run on a
Kaggle GPU notebook (set HF_TOKEN / WANDB_API_KEY first) or locally.

Examples:
  python scripts/train_real.py --num-samples 200 --layer 6
  python scripts/train_real.py --tiny  # dev smoke test, no network

Requirements on Kaggle:
  from kaggle_secrets import UserSecretsClient
  s = UserSecretsClient()
  s.set_secret("HF_TOKEN", ...); s.set_secret("WANDB_API_KEY", ...)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_arabic.activations import ActivationsWriter, extract_activations, load_model
from sae_arabic.data import ArabicDatasetLoader, ArabicTokenizer
from sae_arabic.sae import (
    SAEConfig,
    SAETrainerConfig,
    SparseAutoencoder,
    load_checkpoint,
    save_checkpoint,
    train_sae,
)


def _load_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def _tiny_assets():
    from tests.helpers import StubTokenizer, tiny_bert

    model, tokenizer = tiny_bert(), StubTokenizer()
    texts = [f"جملة عربية رقم {i} للتدريب" for i in range(50)]
    return model, tokenizer, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training_config.yaml")
    parser.add_argument("--tiny", action="store_true", help="dev smoke test with tiny BERT")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    _load_env()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("num_samples", "layer", "num_steps", "out_dir"):
        value = getattr(args, key.replace("-", "_"), None)
        if value is not None:
            cfg[key] = value
    cfg["out_dir"] = Path(cfg["out_dir"])
    out_dir = cfg["out_dir"]
    (out_dir / "activations").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    if args.tiny:
        model, tokenizer, texts = _tiny_assets()
        cfg["layer"] = 2
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

    arabic = ArabicTokenizer(tokenizer=tokenizer, max_length=cfg["max_length"])
    texts = [arabic.normalize(t) for t in texts]
    print(f"using {len(texts)} texts on {cfg['layer']=}")

    acts = extract_activations(
        texts, model, tokenizer, layer=cfg["layer"],
        batch_size=cfg["batch_size"], max_length=cfg["max_length"],
    )
    writer = ActivationsWriter(out_dir / "activations", shard_size=256)
    writer.write_batch([f"t{i:06d}" for i in range(len(acts))], acts)
    writer.close()
    flat = np.concatenate(acts, axis=0)
    print(f"extracted {flat.shape[0]} tokens, dim={flat.shape[1]}")

    sae = SparseAutoencoder(
        SAEConfig(d_model=flat.shape[1], dict_mult=cfg["dict_mult"], l1_coeff=cfg["l1_coeff"])
    )
    train_cfg = SAETrainerConfig(
        lr=cfg["lr"], batch_size=cfg["batch_size"], num_steps=cfg["num_steps"],
        warmup_steps=cfg["warmup_steps"], max_grad_norm=cfg["max_grad_norm"],
        checkpoint_every=cfg["checkpoint_every"], seed=cfg["seed"],
    )

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb_run = wandb.init(
            project=cfg["wandb_project"],
            name=f"layer{cfg['layer']}_{cfg['dict_mult']}x",
            config={**cfg, "n_tokens": int(flat.shape[0])},
        )
    train_sae(sae, torch.from_numpy(flat), train_cfg, wandb_run=wandb_run, checkpoint_dir=out_dir / "checkpoints")
    save_checkpoint(sae, out_dir / "checkpoints" / "final.pt", cfg["num_steps"])
    sae = load_checkpoint(out_dir / "checkpoints" / "final.pt")

    with torch.no_grad():
        _, latents = sae(torch.from_numpy(flat[:2048]))
    report = {
        "config": cfg,
        "n_texts": len(texts),
        "n_tokens": int(flat.shape[0]),
        "dead_feature_fraction": float(sae.dead_features(latents)),
        "checkpoint": str(out_dir / "checkpoints" / "final.pt"),
        "activation_shards": len(list((out_dir / "activations").glob("shard_*.npz"))),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({"dead_feature_fraction": report["dead_feature_fraction"]})
        wandb_run.finish()

    print(json.dumps(report, indent=2, default=str))
    print("TRAINING RUN OK")


if __name__ == "__main__":
    main()
