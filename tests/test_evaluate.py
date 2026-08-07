"""Offline tests for SAE checkpoint evaluation."""

import numpy as np
import torch

from sae_arabic.activations import ActivationsWriter
from sae_arabic.sae import SAEConfig, SparseAutoencoder
from scripts.evaluate import evaluate, load_activations


class _HalfSAE(SparseAutoencoder):
    """Deterministic stand-in returning ``0.5 * x`` as the reconstruction."""

    def forward(self, x):
        return 0.5 * x, torch.zeros(x.shape[0], self.config.d_dict)


def _tiny_acts(n=2000, d=16, seed=0):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))


def test_evaluate_random_sae_metrics_in_range():
    sae = SparseAutoencoder(SAEConfig(d_model=16, dict_mult=2))
    m = evaluate(sae, _tiny_acts())
    assert m["d_dict"] == 32
    assert m["n_tokens"] == 2000
    assert m["explained_variance"] <= 1.0 + 1e-6
    assert 0.0 <= m["dead_features_never_fired"] <= 1.0
    assert 0.0 <= m["dead_features_lt_1e-3"] <= 1.0
    assert 0.0 <= m["features_active_gt_1pct"] <= 1.0
    assert m["reconstruction_mse"] >= 0.0


def test_explained_variance_formula():
    x = torch.randn(2000, 8)
    sae = _HalfSAE(SAEConfig(d_model=8, dict_mult=4))
    m = evaluate(sae, x)
    assert abs(m["explained_variance"] - 0.75) < 0.02


def test_evaluate_requires_activations():
    sae = SparseAutoencoder(SAEConfig(d_model=16, dict_mult=2))
    try:
        evaluate(sae, torch.zeros(0, 16))
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty activations")


def test_load_activations_from_dir_and_file(tmp_path):
    writer = ActivationsWriter(tmp_path, shard_size=3)
    writer.write_batch(
        ["a", "b", "c"],
        [np.ones((2, 4)), np.ones((3, 4)), np.ones((1, 4))],
    )
    writer.close()

    from_dir = load_activations(activations_dir=str(tmp_path))
    assert from_dir.shape == (6, 4)

    from_file = load_activations(activations_file=str(tmp_path / "shard_000000.npz"))
    assert from_file.shape == (6, 4)
    assert from_file.dtype == torch.float32
