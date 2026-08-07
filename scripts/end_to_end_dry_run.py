"""Offline end-to-end dry run (Phases 1-3 wiring smoke test).

Exercises extraction -> serialization -> SAE training -> analysis -> causal
scrub -> correlation on a tiny random BERT and synthetic Arabic text. The
real dry run (MARBERT + NADI/MADAR, 100-500 sentences) runs on Kaggle.

Usage:  python -m scripts.end_to_end_dry_run [--n-samples N] [--num-steps N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from sae_arabic.activations import ActivationsWriter, extract_activations
from sae_arabic.aldi import causal_scrub, feature_aldicorrelation
from sae_arabic.analysis import contexts_for_features, top_features
from sae_arabic.sae import (
    SAEConfig,
    SAETrainerConfig,
    SparseAutoencoder,
    load_checkpoint,
    save_checkpoint,
    train_sae,
)
from tests.helpers import StubTokenizer, tiny_bert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--out-dir", default="data/dry_run")
    args = parser.parse_args()

    model = tiny_bert()
    tokenizer = StubTokenizer()
    texts = [f"جملة عربية رقم {i} للمعالجة الآلية" for i in range(args.n_samples)]

    t0 = time.time()
    acts = extract_activations(texts, model, tokenizer, layer=2, batch_size=8)
    print(f"[1/6] extracted {len(acts)} sequences in {time.time() - t0:.1f}s")

    writer = ActivationsWriter(args.out_dir, shard_size=16)
    writer.write_batch([f"s{i:04d}" for i in range(len(acts))], acts)
    writer.close()
    flat = np.concatenate(acts, axis=0)
    print(f"[2/6] serialized shards; tokens={flat.shape[0]} dim={flat.shape[1]}")

    sae = SparseAutoencoder(SAEConfig(d_model=flat.shape[1], dict_mult=4))
    train_config = SAETrainerConfig(
        lr=1e-3, num_steps=args.num_steps, batch_size=64, warmup_steps=20
    )
    t0 = time.time()
    train_sae(sae, torch.from_numpy(flat), train_config, checkpoint_dir=args.out_dir)
    save_checkpoint(sae, f"{args.out_dir}/final.pt", args.num_steps)
    sae = load_checkpoint(f"{args.out_dir}/final.pt")
    print(f"[3/6] trained + checkpointed SAE in {time.time() - t0:.1f}s")

    with torch.no_grad():
        _, latents = sae(torch.from_numpy(flat[:512]))
    dead = float(sae.dead_features(latents))
    top = top_features(latents[:100].numpy(), k=3)
    print(f"[4/6] dead-feature fraction={dead:.2f}  top features={list(top)[:4]}")

    first_len = len(texts[0])
    contexts = contexts_for_features(
        [top_id for top_id in list(top)[:2]],
        list(texts[0]),
        latents[:first_len].numpy(),
        window=2,
        k=2,
    )
    print(f"[5/6] sample contexts: { {k: v[:1] for k, v in contexts.items()} }")

    scrub = causal_scrub(
        model,
        tokenizer,
        sae,
        texts[:4],
        feature_ids=[0, 1],
        layer=2,
        scorer=lambda t: [len(x) for x in t],
        modes=("ablate", "control"),
    )
    print("[6/6] causal scrub keys:", sorted(scrub))

    with torch.no_grad():
        pooled = np.stack(
            [
                sae(torch.from_numpy(a))[1].mean(dim=0).numpy()
                for a in acts[:16]
            ]
        )
    corr = feature_aldicorrelation(
        torch.from_numpy(pooled), [len(t) for t in texts[:16]]
    )
    print("      top correlated features:", corr.topk(3).values.tolist())

    print("\nDRY RUN OK")


if __name__ == "__main__":
    main()
