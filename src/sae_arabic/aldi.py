"""ALDi integration and causal validation of SAE features (Phase 3).

The primary validation approach uses MLM-head reconstruction: mask tokens,
run a forward pass with the SAE feature ablated/amplified, substitute the
top-1 predictions, and score the resulting text with ALDi. A correlational
check (feature activation vs. original text ALDi) serves as fallback.

ALDi scoring uses the public ``AMR-KELEG/Sentence-ALDi`` model (a MARBERT
regression head estimating the Arabic "Level of Dialectness" in [0, 1]),
overridable via the ``ALDI_MODEL`` env var or by passing a custom scorer.
"""

from __future__ import annotations

import os
import random
import re

import torch

from sae_arabic.activations import _resolve_layer
from sae_arabic.sae import SparseAutoencoder


def _clamp_score(value: float) -> float:
    return min(max(0.0, float(value)), 1.0)


def _preprocess_aldi(text: str) -> str:
    """Match the official ALDi preprocessing: strip URLs and Latin letters."""
    no_urls = re.sub(
        r"(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b",
        "",
        text,
        flags=re.MULTILINE,
    )
    return re.sub(r"[a-zA-Z]", "", no_urls)


class AldiScorer:
    """Sentence-level ALDi dialectness scorer (AMR-KELEG/Sentence-ALDi).

    0 = Modern Standard Arabic, 1 = highly dialectal. With ``seed_average``
    enabled, the three published seeds (42/30/50) are averaged to reduce
    prediction noise and improve calibration.
    """

    DEFAULT_SEEDS = (
        "AMR-KELEG/Sentence-ALDi",
        "AMR-KELEG/Sentence-ALDi-30",
        "AMR-KELEG/Sentence-ALDi-50",
    )

    def __init__(
        self,
        model_name: str = "AMR-KELEG/Sentence-ALDi",
        device: str | None = None,
        batch_size: int = 32,
        seed_average: bool = False,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        names = self.DEFAULT_SEEDS if seed_average else (model_name,)
        self.tokenizer = AutoTokenizer.from_pretrained(names[0])
        self.models = [
            AutoModelForSequenceClassification.from_pretrained(n) for n in names
        ]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        for m in self.models:
            m.to(self.device).eval()
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> list[float]:
        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                enc = self.tokenizer(
                    [_preprocess_aldi(t) for t in texts[start : start + self.batch_size]],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = torch.stack([m(**enc).logits[:, 0] for m in self.models]).mean(dim=0)
                scores.extend(logits.cpu().tolist())
        return [_clamp_score(s) for s in scores]


def aldi_score(texts: list[str], scorer=None, device: str = "cpu", seed_average: bool = False) -> list[float]:
    """Score dialectness of ``texts`` with ALDi, clamped to [0, 1].

    ``scorer`` must be callable with a list of texts and return a list of
    floats. If omitted, the public Sentence-ALDi model is loaded lazily
    (override the model with the ``ALDI_MODEL`` env var). Set ``seed_average=True``
    to average across the three published seeds (42/30/50).
    """
    if scorer is None:
        model_name = os.environ.get("ALDI_MODEL", "AMR-KELEG/Sentence-ALDi")
        scorer = AldiScorer(model_name=model_name, device=device, seed_average=seed_average)
    return [float(s) for s in scorer(texts)]


class _Capture:
    def __init__(self) -> None:
        self.raw = None
        self.output: torch.Tensor | None = None

    def __call__(self, module, args, output) -> None:
        self.raw = output
        self.output = output[0] if isinstance(output, tuple) else output


def _replace_layer_hook(value: torch.Tensor, reference):
    """Return a hook emitting ``value`` while mirroring the layer output shape."""

    def hook(module, args, output):
        if isinstance(reference, tuple):
            return (value,) + reference[1:]
        return value

    return hook


def causal_scrub(
    model,
    tokenizer,
    sae: SparseAutoencoder,
    texts: list[str],
    feature_ids: list[int],
    layer: int,
    scorer=None,
    modes: tuple[str, ...] = ("ablate", "amplify"),
    factor: float = 5.0,
    seed: int = 0,
    device: str | None = None,
    seed_average: bool = False,
    fwd_batch: int = 64,
    max_length: int = 128,
) -> dict:
    """MLM-head reconstruction causal validation (memory-safe, batched).

    Returns ALDi scores per text for ``original`` texts, ``base`` masked
    reconstruction without intervention, and per-mode/per-feature dicts for
    ``ablate`` (zero the feature), ``amplify`` (scale by ``factor``), and
    ``control`` (ablate a random other feature).

    Forward passes run in chunks of ``fwd_batch`` texts to avoid OOM.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if scorer is None:
        model_name = os.environ.get("ALDI_MODEL", "AMR-KELEG/Sentence-ALDi")
        scorer = AldiScorer(model_name=model_name, device=device, seed_average=seed_average)
    model.to(device)
    model.eval()
    sae.to(device)
    sae.eval()

    encoding = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    layer_module = _resolve_layer(model, layer)
    special_ids = set(tokenizer.all_special_ids)
    is_special = torch.tensor(
        [[t in special_ids for t in row] for row in input_ids.cpu()], device=device
    )
    mask_targets = ~is_special
    mask_token_id = tokenizer.mask_token_id
    masked_ids = torch.where(mask_targets, mask_token_id, input_ids)

    def _run_forward(ids, attn, hook_value=None):
        if hook_value is not None:
            h = layer_module.register_forward_hook(
                _replace_layer_hook(hook_value, ids)
            )
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits
        if hook_value is not None:
            h.remove()
        return logits

    def _capture_hidden() -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_hidden, all_raw = [], []
        for i in range(0, len(texts), fwd_batch):
            handle = layer_module.register_forward_hook(
                lambda m, a, o: None
            )
            hook = _Capture()
            handle = layer_module.register_forward_hook(hook)
            with torch.no_grad():
                model(
                    input_ids=input_ids[i : i + fwd_batch],
                    attention_mask=attention_mask[i : i + fwd_batch],
                )
            handle.remove()
            all_hidden.append(hook.output)
            all_raw.append(hook.raw)
        return torch.cat(all_hidden, dim=0), all_raw

    def forward_with_batched(hook_values_list: list[torch.Tensor]) -> list[str]:
        all_text = []
        for i in range(0, len(texts), fwd_batch):
            hv = hook_values_list[i]
            h = layer_module.register_forward_hook(
                _replace_layer_hook(hv, capture_raw[i])
            )
            with torch.no_grad():
                logits = model(
                    input_ids=masked_ids[i : i + fwd_batch],
                    attention_mask=attention_mask[i : i + fwd_batch],
                ).logits
            h.remove()
            preds = logits.argmax(dim=-1)
            sub = torch.where(mask_targets[i : i + fwd_batch], preds, input_ids[i : i + fwd_batch])
            all_text.extend(
                tokenizer.batch_decode(sub, skip_special_tokens=True)
            )
        return all_text

    hidden_all, capture_raw = _capture_hidden()
    latents_all = torch.relu(sae.encoder(hidden_all))

    results: dict = {"original": aldi_score(texts, scorer), "base": None}
    base_texts = forward_with_batched(
        {i: hidden_all[i : i + 1] for i in range(0, len(texts), 1)}
    )
    results["base"] = aldi_score(base_texts, scorer)

    rng = random.Random(seed)
    for mode in modes:
        results.setdefault(mode, {})
        for feature_id in feature_ids:
            f_mod = latents_all.clone()
            if mode == "ablate":
                f_mod[..., feature_id] = 0.0
            elif mode == "amplify":
                f_mod[..., feature_id] = f_mod[..., feature_id] * factor
            elif mode == "control":
                candidates = [f for f in range(latents_all.shape[-1]) if f != feature_id]
                f_mod[..., rng.choice(candidates)] = 0.0
            else:
                raise ValueError(f"unknown mode: {mode}")
            x_mod = sae.decoder(f_mod)
            mod_list = {i: x_mod[i : i + 1] for i in range(len(texts))}
            result_texts = forward_with_batched(mod_list)
            results[mode][feature_id] = aldi_score(result_texts, scorer)
    return results


def feature_aldicorrelation(
    activations: torch.Tensor, scores: list[float]
) -> torch.Tensor:
    """Correlational fallback: per-feature Pearson r vs. text ALDi.

    ``activations`` has shape ``(n_texts, d)`` or ``(n_texts, n_tokens, d)``
    (mean-pooled over tokens). Returns a tensor of shape ``(d,)``.
    """
    acts = activations.float()
    if acts.ndim == 3:
        acts = acts.mean(dim=1)
    acts = acts - acts.mean(dim=0, keepdim=True)
    acts = acts / (acts.std(dim=0, keepdim=True) + 1e-8)

    s = torch.as_tensor(scores, dtype=acts.dtype)
    s = s - s.mean()
    s = s / (s.std() + 1e-8)
    return (acts.T @ s) / acts.shape[0]
