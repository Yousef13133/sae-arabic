"""Minimal custom Sparse Autoencoder (SAE) compatible with encoder-only models.

Implements a linear encoder/decoder pair with an L1 sparsity penalty (Phase 2).
Designed to be architecture-agnostic so it works with MARBERT's hidden states
without requiring standard SAE libraries built for decoder-only stacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


@dataclass
class SAEConfig:
    """Configuration for a SparseAutoencoder instance.

    Attributes:
        d_model: Hidden dimension of the input activations.
        dict_mult: Dictionary expansion factor (4-8x hidden dimension).
        l1_coeff: Weight of the L1 sparsity penalty on encoder activations.
        use_bias: Whether encoder/decoder carry bias terms.
    """

    d_model: int = 768
    dict_mult: int = 8
    l1_coeff: float = 1e-3
    use_bias: bool = True

    @property
    def d_dict(self) -> int:
        return self.d_model * self.dict_mult


class SparseAutoencoder(nn.Module):
    """Linear encoder/decoder SAE with MSE reconstruction + L1 sparsity loss."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Linear(config.d_model, config.d_dict, bias=config.use_bias)
        self.decoder = nn.Linear(config.d_dict, config.d_model, bias=config.use_bias)
        nn.init.kaiming_normal_(self.encoder.weight)
        nn.init.zeros_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reconstruction, latent activations) for input ``x``."""
        f = torch.relu(self.encoder(x))
        x_hat = self.decoder(f)
        return x_hat, f

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        mse = nn.functional.mse_loss(x_hat, x)
        l1 = f.abs().sum(dim=-1).mean()
        return mse + self.config.l1_coeff * l1

    def dead_features(self, f: torch.Tensor, threshold: float = 1e-3) -> torch.Tensor:
        """Fraction of dictionary features never exceeding ``threshold`` activation."""
        max_act = f.max(dim=0).values
        return (max_act < threshold).float().mean()


@dataclass
class SAETrainerConfig:
    """Hyperparameters for the SAE training loop (Phase 2, Week 3-4)."""

    lr: float = 1e-4
    warmup_steps: int = 100
    batch_size: int = 32
    num_steps: int = 10_000
    checkpoint_every: int = 1_000
    log_every: int = 50
    max_grad_norm: float = 1.0
    seed: int = 42


def _linear_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    """Warmup-then-linear-decay learning-rate multiplier in [0, 1]."""
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))


def train_sae(
    model: SparseAutoencoder,
    activations: torch.Tensor,
    config: SAETrainerConfig,
    wandb_run=None,
    checkpoint_dir: str | None = None,
) -> SparseAutoencoder:
    """Train ``model`` on ``activations`` (Phase 2 primary training loop).

    Token-level training: ``activations`` of shape ``(N, d_model)`` or
    ``(N, T, d_model)`` are flattened and random token vectors are sampled
    per step. Uses AdamW with linear warmup/decay, gradient clipping, and
    periodic checkpointing.
    """
    if activations.ndim == 3:
        activations = activations.reshape(-1, activations.shape[-1])
    tokens = activations.detach().float()
    if config.seed is not None:
        torch.manual_seed(config.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    model.train()

    def lr_for(step: int) -> float:
        return config.lr * _linear_warmup(step, config.warmup_steps, config.num_steps)

    for step in range(config.num_steps):
        idx = torch.randint(0, len(tokens), (config.batch_size,))
        x = tokens[idx]
        x_hat, f = model(x)
        loss = model.loss(x, x_hat, f)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        for group in optimizer.param_groups:
            group["lr"] = lr_for(step)
        optimizer.step()

        if step % config.log_every == 0 and wandb_run is not None:
            wandb_run.log(
                {
                    "loss": loss.item(),
                    "reconstruction_mse": nn.functional.mse_loss(x_hat, x).item(),
                    "l1": f.abs().sum(dim=-1).mean().item(),
                    "dead_features": model.dead_features(f).item(),
                    "lr": lr_for(step),
                },
                step=step,
            )

        if checkpoint_dir is not None and step % config.checkpoint_every == 0:
            save_checkpoint(model, Path(checkpoint_dir) / f"checkpoint_{step:08d}.pt", step)

    if checkpoint_dir is not None:
        save_checkpoint(model, Path(checkpoint_dir) / "final.pt", config.num_steps)
    return model


def save_checkpoint(model: SparseAutoencoder, path: str | Path, step: int | None = None) -> None:
    """Serialize an SAE and its configuration to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": model.config.__dict__, "step": step},
        path,
    )


def load_checkpoint(path: str | Path) -> SparseAutoencoder:
    """Restore an SAE from a checkpoint written by ``save_checkpoint``."""
    data = torch.load(path, map_location="cpu")
    model = SparseAutoencoder(SAEConfig(**data["config"]))
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model
