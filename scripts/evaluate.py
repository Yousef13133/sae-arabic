"""Evaluate a trained SAE checkpoint on activation shards (Phase 2-3).

Reports reconstruction quality (explained variance), sparsity (L1), and
feature statistics (dead rate, activation frequency).

Usage:
  python scripts/evaluate.py --checkpoint data/real_run/checkpoints/final.pt \
      --activations-dir data/real_run/activations
  python scripts/evaluate.py --checkpoint x.pt --activations-file shard_000000.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sae_arabic.activations import ActivationsWriter
from sae_arabic.sae import SparseAutoencoder, load_checkpoint


def _coerce(acts: np.ndarray) -> np.ndarray:
    if acts.dtype == object or acts.ndim == 1:
        acts = np.concatenate([np.asarray(a) for a in acts if a is not None], axis=0)
    if acts.ndim == 3:
        acts = acts.reshape(-1, acts.shape[-1])
    return acts.astype(np.float32)


def load_activations(
    activations_dir: str | None = None,
    activations_file: str | None = None,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """Load activation shards into a flat ``(n_tokens, d_model)`` tensor."""
    if activations_dir:
        arrays = []
        for _, acts in ActivationsWriter(Path(activations_dir)).iter_shards():
            arrays.append(acts)
        if not arrays:
            raise FileNotFoundError(f"no shards found in {activations_dir}")
        acts = np.concatenate(arrays, axis=0)
    elif activations_file:
        data = np.load(activations_file, allow_pickle=True)
        acts = data["acts"]
    else:
        raise ValueError("provide --activations-dir or --activations-file")
    acts = _coerce(acts)
    if max_tokens:
        acts = acts[:max_tokens]
    return torch.from_numpy(acts)


def evaluate(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    batch_size: int = 8192,
    device: str | None = None,
    dead_threshold: float = 1e-3,
) -> dict:
    """Compute reconstruction and feature statistics for ``sae`` on ``activations``."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sae = sae.to(device).eval()
    acts = activations.float()
    n, d = acts.shape
    if n == 0:
        raise ValueError("no activations to evaluate")

    sum_x = torch.zeros(d, device=device)
    sum_x2 = torch.zeros(d, device=device)
    sum_mse = torch.zeros(d, device=device)
    sum_l1 = 0.0
    sum_f = torch.zeros(sae.config.d_dict, device=device)
    max_act = torch.full((sae.config.d_dict,), float("-inf"), device=device)
    count_act = torch.zeros(sae.config.d_dict, device=device)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            x = acts[start : start + batch_size].to(device)
            x_hat, f = sae(x)
            sum_x += x.sum(0)
            sum_x2 += (x * x).sum(0)
            sum_mse += ((x - x_hat) ** 2).sum(0)
            sum_l1 += f.abs().sum().item()
            sum_f += f.sum(0)
            max_act = torch.maximum(max_act, f.amax(0))
            count_act += (f > 0).sum(0)

    mean = sum_x / n
    var_total = (sum_x2 / n - mean**2).sum().item()
    mse_total = sum_mse.sum().item()
    freq = count_act / n
    explained = 1.0 - mse_total / var_total if var_total > 0 else float("nan")

    return {
        "n_tokens": int(n),
        "d_dict": sae.config.d_dict,
        "explained_variance": explained,
        "reconstruction_mse": mse_total / n,
        "l1_per_token": sum_l1 / n,
        "dead_features_lt_1e-3": (max_act < dead_threshold).float().mean().item(),
        "dead_features_never_fired": (max_act <= 0).float().mean().item(),
        "feature_freq_median": freq.median().item(),
        "feature_freq_p90": torch.quantile(freq, 0.9).item(),
        "features_active_gt_1pct": (freq > 0.01).float().mean().item(),
        "features_rare_lt_1e-4": (freq < 1e-4).float().mean().item(),
        "mean_activation_median": (sum_f / n).median().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activations-dir")
    parser.add_argument("--activations-file")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--out")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    sae = load_checkpoint(args.checkpoint)
    acts = load_activations(args.activations_dir, args.activations_file, args.max_tokens)
    metrics = evaluate(sae, acts, batch_size=args.batch_size)

    out = Path(args.out) if args.out else Path(args.checkpoint).parent / "evaluation.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
