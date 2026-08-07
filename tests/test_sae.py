"""Baseline unit tests for the SAE module (Phase 4, Week 10)."""

import torch

from sae_arabic.sae import (
    SAEConfig,
    SAETrainerConfig,
    SparseAutoencoder,
    load_checkpoint,
    save_checkpoint,
    train_sae,
)


def test_forward_shapes():
    config = SAEConfig(d_model=8, dict_mult=4)
    sae = SparseAutoencoder(config)

    x = torch.randn(2, 16, 8)
    x_hat, f = sae(x)
    assert x_hat.shape == x.shape
    assert f.shape == (2, 16, 32)


def test_l1_penalty_nonzero():
    config = SAEConfig(d_model=8, dict_mult=4)
    sae = SparseAutoencoder(config)

    x = torch.randn(2, 16, 8)
    x_hat, f = sae(x)
    loss = sae.loss(x, x_hat, f)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_train_reduces_loss():
    torch.manual_seed(0)
    config = SAEConfig(d_model=16, dict_mult=2)
    sae = SparseAutoencoder(config)
    activations = torch.randn(500, 16)

    x = activations[:64]
    x_hat, f = sae(x)
    loss_before = sae.loss(x, x_hat, f).item()

    train_config = SAETrainerConfig(lr=1e-2, num_steps=60, batch_size=64, warmup_steps=5, seed=0)
    train_sae(sae, activations, train_config)

    x_hat, f = sae(x)
    loss_after = sae.loss(x, x_hat, f).item()
    assert loss_after < loss_before


def test_checkpoint_roundtrip(tmp_path):
    config = SAEConfig(d_model=8, dict_mult=2)
    sae = SparseAutoencoder(config)
    path = tmp_path / "sae.pt"
    save_checkpoint(sae, path, step=10)

    loaded = load_checkpoint(path)
    x = torch.randn(3, 8)
    with torch.no_grad():
        a1, _ = sae(x)
        a2, _ = loaded(x)
    assert torch.allclose(a1, a2)
    assert loaded.config == config
