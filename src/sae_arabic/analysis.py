"""Feature analysis and browsing tools (Phase 3, Week 6).

Extracts and aggregates top-activating features with their text contexts and
provides a streamlined CLI/Notebook interface. Complex UI/dashboard
components are explicitly out of scope.
"""

from __future__ import annotations

import numpy as np


def top_features(latents: np.ndarray, k: int = 50) -> dict[int, list[tuple[int, float]]]:
    """Return the top-``k`` activating token positions per feature.

    ``latents`` has shape ``(T, D)`` (or ``(n, T, D)``, which is flattened).
    Returns ``{feature_id: [(token_index, activation), ...]}`` sorted by
    descending activation.
    """
    if latents.ndim == 3:
        latents = latents.reshape(-1, latents.shape[-1])
    n_tokens, d_dict = latents.shape
    result: dict[int, list[tuple[int, float]]] = {}
    k = min(k, n_tokens)
    for feature in range(d_dict):
        col = latents[:, feature]
        top_idx = np.argsort(col)[::-1][:k]
        result[feature] = [(int(i), float(col[i])) for i in top_idx]
    return result


def contexts_for_features(
    feature_ids: list[int],
    tokens: list[str],
    latents: np.ndarray,
    window: int = 3,
    k: int = 5,
) -> dict[int, list[tuple[str, float]]]:
    """Return ``(context_string, activation)`` for the top-``k`` activations.

    ``tokens`` and ``latents`` rows must be aligned; each context is the
    ``window``-token neighborhood (``±window``) around the activating token.
    """
    if latents.ndim == 3:
        latents = latents.reshape(-1, latents.shape[-1])
    if len(tokens) != latents.shape[0]:
        raise ValueError("tokens and latents must have the same number of tokens")
    top = top_features(latents, k=k)
    out: dict[int, list[tuple[str, float]]] = {}
    for feature_id in feature_ids:
        if feature_id not in top:
            continue
        contexts = []
        for token_idx, act in top[feature_id]:
            lo, hi = max(0, token_idx - window), min(len(tokens), token_idx + window + 1)
            ctx = " ".join(tokens[lo:hi])
            contexts.append((ctx, act))
        out[feature_id] = contexts
    return out
