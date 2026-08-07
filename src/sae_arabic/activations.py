"""Activation extraction pipeline for encoder-only models (Phase 1).

Handles batch processing of text through MARBERT and serializes intermediate
layer activations to disk for downstream SAE training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DEFAULT_MODEL = "UBC-NLP/MARBERT"


def load_model(model_name: str = DEFAULT_MODEL):
    """Load an encoder-only masked LM and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def _resolve_layer(model, layer: int) -> torch.nn.Module:
    """Return the module whose output is the hidden states of ``layer``."""
    base = getattr(model, "base_model", model)
    encoder = getattr(base, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        try:
            return encoder.layer[layer]
        except (IndexError, TypeError):
            pass
    for name, module in model.named_modules():
        if name.endswith(f".layer.{layer}"):
            return module
    raise ValueError(f"could not locate layer {layer} in model {type(model).__name__}")


class _ForwardHook:
    """Captures a module's forward output."""

    def __init__(self) -> None:
        self.output: torch.Tensor | None = None

    def __call__(self, module, args, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self.output = output


def extract_activations(
    texts: list[str],
    model,
    tokenizer,
    layer: int,
    batch_size: int = 32,
    max_length: int = 128,
    device: str | None = None,
    remove_special_tokens: bool = True,
) -> list[np.ndarray]:
    """Extract layer-``layer`` activations for each input text.

    Returns one float32 array per text of shape ``(n_tokens, d_model)``.
    Special tokens (``[CLS]``, ``[SEP]``, ``[PAD]``) are dropped when
    ``remove_special_tokens`` is true, so ``n_tokens`` may vary per text.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    layer_module = _resolve_layer(model, layer)
    hook = _ForwardHook()
    handle = layer_module.register_forward_hook(hook)

    results: list[np.ndarray] = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoding = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = hook.output.detach().cpu().numpy()
            hook.output = None

            for i, row in enumerate(hidden):
                acts = row
                if remove_special_tokens:
                    special_ids = set(tokenizer.all_special_ids)
                    keep = np.fromiter(
                        (t not in special_ids for t in input_ids[i].tolist()),
                        dtype=bool,
                        count=len(input_ids[i]),
                    )
                    acts = row[keep]
                results.append(acts.astype(np.float32))
    finally:
        handle.remove()
    return results


class ActivationsWriter:
    """Serializes batched activations to disk (Phase 1, Days 3-4).

    Activations are appended into shards of up to ``shard_size`` samples and
    flushed as ``shard_XXXXXX.npz`` files, each storing ``ids`` (``str``
    array) and ``acts`` (object array of ``(n_tokens, d_model)`` arrays).
    """

    def __init__(self, out_dir: str | Path, shard_size: int = 512):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self._ids: list[str] = []
        self._acts: list[np.ndarray] = []
        self._shard_index = 0

    def write_batch(self, ids: list[str], acts: list[np.ndarray]) -> None:
        if len(ids) != len(acts):
            raise ValueError("ids and acts must have the same length")
        self._ids.extend(ids)
        self._acts.extend(acts)
        while len(self._ids) >= self.shard_size:
            self._flush(self.shard_size)

    def close(self) -> None:
        if self._ids:
            self._flush(len(self._ids))

    def _flush(self, n: int) -> None:
        path = self.out_dir / f"shard_{self._shard_index:06d}.npz"
        ids = np.asarray(self._ids[:n], dtype=object)
        acts = np.empty(n, dtype=object)
        acts[:] = self._acts[:n]
        np.savez(path, ids=ids, acts=acts)
        self._ids = self._ids[n:]
        self._acts = self._acts[n:]
        self._shard_index += 1

    def iter_shards(self):
        """Yield ``(ids, acts)`` for every shard written so far."""
        for path in sorted(self.out_dir.glob("shard_*.npz")):
            yield self.load_shard(path)

    @staticmethod
    def load_shard(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        return data["ids"], data["acts"]
