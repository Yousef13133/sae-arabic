"""Phase 3 causal validation: ablate/amplify SAE features and measure ALDi shifts.

Selects the top ALDi-correlated features, then runs the MLM-head causal scrub
to measure each feature's effect on the Arabic Level of Dialectness.

The reported metric is the effect *relative to a control* (ablation of a random
other feature), with a paired bootstrap 95% CI. This isolates feature-specific
causal signal from the generic reconstruction artifact introduced by any hidden
state replacement.

Usage:
  python -m scripts.causal_validate --checkpoint /content/sweep/layer8.pt --layer 8
  python -m scripts.causal_validate --checkpoint data/real_run/checkpoints/final.pt --layer 6 --num-texts 500 --num-features 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from sae_arabic.activations import extract_activations, load_model
from sae_arabic.aldi import aldi_score, causal_scrub, feature_aldicorrelation
from sae_arabic.data import ArabicDatasetLoader, ArabicTokenizer
from sae_arabic.sae import load_checkpoint


def bootstrap_relative_effect(
    treated: list[float],
    control: list[float],
    rng: np.random.Generator,
    n_iter: int = 1000,
) -> tuple[float, float, float, float]:
    """Paired bootstrap of ``treated - control`` per-text deltas.

    Returns ``(mean_effect, ci_low, ci_high, p)`` where ``p`` is the fraction
    of bootstrap means <= 0 (small p => effect is robustly positive).
    """
    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    diffs = treated - control
    n = len(diffs)
    means = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = diffs[idx].mean()
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    p = float(np.mean(means <= 0))
    return float(diffs.mean()), float(ci_low), float(ci_high), p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="training_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--num-texts", type=int, default=500)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--seed-average", action="store_true", default=True)
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(args.out_dir or str(Path(args.checkpoint).parent))
    out_dir.mkdir(parents=True, exist_ok=True)

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
    ds = ds.select(range(min(len(ds), args.num_texts)))
    texts = [t for t in ds[cfg["text_column"]] if isinstance(t, str) and t.strip()]
    arabic = ArabicTokenizer(tokenizer=tokenizer, max_length=cfg["max_length"])
    texts = [arabic.normalize(t) for t in texts]
    print(f"loaded {len(texts)} texts")

    aldi_scores = aldi_score(texts, seed_average=args.seed_average)
    print(f"baseline ALDi mean={float(np.mean(aldi_scores)):.4f}")

    acts = extract_activations(
        texts, model, tokenizer, layer=args.layer,
        batch_size=cfg["batch_size"], max_length=cfg["max_length"],
    )
    flat = np.concatenate(acts, axis=0)
    sae = load_checkpoint(args.checkpoint)
    device = next(sae.parameters()).device
    with torch.no_grad():
        all_latents = sae(torch.from_numpy(flat).to(device))[1].float().cpu()

    per_text = []
    start = 0
    for a in acts:
        end = start + len(a)
        per_text.append(all_latents[start:end].mean(0))
        start = end
    per_text_latents = torch.stack(per_text)

    corr = feature_aldicorrelation(per_text_latents, aldi_scores)
    top_feat_ids = corr.argsort(descending=True)[: args.num_features].tolist()
    print(f"top features by ALDi correlation: {top_feat_ids} (corr={[f'{corr[f]:.3f}' for f in top_feat_ids]})")

    scrub = causal_scrub(
        model,
        tokenizer,
        sae,
        texts,
        feature_ids=top_feat_ids,
        layer=args.layer,
        modes=("ablate", "amplify", "control"),
        seed_average=args.seed_average,
    )

    baseline_mean = float(np.mean(scrub["original"]))
    base_mean = float(np.mean(scrub["base"]))
    feature_effects: dict[str, dict] = {}
    for mode in ("ablate", "amplify", "control"):
        for fid, scores in scrub[mode].items():
            mean_score = float(np.mean(scores))
            feature_effects[f"{mode}:{fid}"] = {
                "mean_aldi": mean_score,
                "delta": mean_score - baseline_mean,
            }

    rng = np.random.default_rng(0)
    relative_effects: dict[str, dict] = {}
    for mode in ("ablate", "amplify"):
        for fid in top_feat_ids:
            mean_effect, ci_low, ci_high, p = bootstrap_relative_effect(
                scrub[mode][fid], scrub["control"][fid], rng
            )
            relative_effects[f"{mode}:{fid}"] = {
                "mean_relative_to_control": mean_effect,
                "ci95": [ci_low, ci_high],
                "p_effect_positive": 1.0 - p,
            }

    summary = {
        "layer": args.layer,
        "num_texts": len(texts),
        "seed_average": args.seed_average,
        "selected_features": top_feat_ids,
        "feature_correlations": {f: float(corr[f]) for f in top_feat_ids},
        "baseline_aldi_mean": baseline_mean,
        "base_reconstruction_aldi_mean": base_mean,
        "feature_effects": feature_effects,
        "relative_effects_vs_control": relative_effects,
    }
    (out_dir / "causal_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    print(f"\nbaseline={baseline_mean:.4f}  base_recon={base_mean:.4f}")
    print(f"\n{'feature':>8} {'ablate−ctrl Δ':>18} {'95% CI':>24} {'amplify−ctrl Δ':>18}")
    for fid in top_feat_ids:
        abl = relative_effects.get(f"ablate:{fid}", {})
        amp = relative_effects.get(f"amplify:{fid}", {})
        print(
            f"{fid:>8} {abl.get('mean_relative_to_control', 0):>+18.4f} "
            f"[{abl.get('ci95', [0, 0])[0]:+.4f}, {abl.get('ci95', [0, 0])[1]:+.4f}] "
            f"{amp.get('mean_relative_to_control', 0):>+18.4f}"
        )


if __name__ == "__main__":
    main()
