"""Tests for ALDi integration and causal validation."""

import numpy as np
import pytest
import torch

from sae_arabic.aldi import _clamp_score, aldi_score, causal_scrub, feature_aldicorrelation
from sae_arabic.sae import SAEConfig, SparseAutoencoder
from tests.helpers import StubTokenizer, tiny_bert

TEXTS = ["مرحبا بالعالم", "السلام عليكم"]


def _tiny_sae():
    return SparseAutoencoder(SAEConfig(d_model=32, dict_mult=4))


def test_clamp_score():
    assert _clamp_score(-0.5) == 0.0
    assert _clamp_score(1.5) == 1.0
    assert _clamp_score(0.3) == 0.3


def test_aldi_score_with_scorer():
    scores = aldi_score(TEXTS, scorer=lambda texts: [len(t) for t in texts])
    assert scores == [len(TEXTS[0]), len(TEXTS[1])]


def test_causal_scrub_structure():
    model = tiny_bert()
    sae = _tiny_sae()
    scorer = lambda texts: [len(t) for t in texts]
    results = causal_scrub(
        model, StubTokenizer(), sae, TEXTS, feature_ids=[5], layer=2, scorer=scorer,
        modes=("ablate", "amplify", "control"),
    )
    assert set(results) == {"original", "base", "ablate", "amplify", "control"}
    assert len(results["original"]) == len(TEXTS)
    assert len(results["base"]) == len(TEXTS)
    assert list(results["ablate"]) == [5]
    assert len(results["amplify"][5]) == len(TEXTS)
    assert len(results["control"][5]) == len(TEXTS)


def test_causal_scrub_unknown_mode():
    model = tiny_bert()
    sae = _tiny_sae()
    with pytest.raises(ValueError):
        causal_scrub(
            model, StubTokenizer(), sae, TEXTS, feature_ids=[5], layer=2,
            scorer=lambda t: [0.0] * len(t), modes=("nope",),
        )


def test_feature_correlation_shape_and_sign():
    rng = np.random.default_rng(0)
    activations = torch.tensor(rng.normal(size=(50, 8)), dtype=torch.float32)
    scores = [float(activations[i, 0] * 3 + activations[i, 1] * -2) for i in range(50)]
    corr = feature_aldicorrelation(activations, scores)
    assert corr.shape == (8,)
    assert abs(corr[0].item()) > 0.5
    assert abs(corr[1].item()) > 0.5
    assert corr[2].abs().item() < 0.5


def test_feature_correlation_pools_tokens():
    rng = np.random.default_rng(1)
    activations = torch.tensor(rng.normal(size=(20, 5, 3)), dtype=torch.float32)
    pooled = activations.mean(dim=1)
    scores = [float(pooled[i, 2]) for i in range(20)]
    corr = feature_aldicorrelation(activations, scores)
    assert corr.shape == (3,)
    assert corr[2].abs().item() > 0.9
