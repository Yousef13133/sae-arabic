"""Tests for feature analysis tools."""

import numpy as np
import pytest

from sae_arabic.analysis import contexts_for_features, top_features


def test_top_features_orders_by_activation():
    latents = np.zeros((4, 3))
    latents[2, 1] = 5.0
    latents[0, 1] = 2.0
    top = top_features(latents, k=2)
    assert top[1][0] == (2, 5.0)
    assert top[1][1] == (0, 2.0)
    assert len(top) == 3


def test_top_features_flattens_batches():
    latents = np.zeros((2, 3, 2))
    latents[1, 0, 0] = 9.0
    top = top_features(latents, k=1)
    assert top[0] == [(3, 9.0)]


def test_contexts_for_features_builds_windows():
    tokens = ["a", "b", "c", "d", "e"]
    latents = np.zeros((5, 2))
    latents[2, 0] = 4.0
    contexts = contexts_for_features([0], tokens, latents, window=1, k=1)
    assert contexts[0] == [("b c d", 4.0)]


def test_contexts_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        contexts_for_features([0], ["a", "b"], np.zeros((3, 2)))
