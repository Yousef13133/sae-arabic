"""Offline tests for the activation extraction pipeline."""

import numpy as np
import pytest

from sae_arabic.activations import ActivationsWriter, extract_activations
from tests.helpers import StubTokenizer, tiny_bert


@pytest.fixture
def tiny_model():
    return tiny_bert()


TEXTS = ["مرحبا بالعالم", "السلام عليكم", "هذا اختبار للنموذج"]


def test_extract_shapes_and_special_tokens_removed(tiny_model):
    results = extract_activations(TEXTS, tiny_model, StubTokenizer(), layer=2, batch_size=2, max_length=64)
    assert len(results) == len(TEXTS)
    for text, acts in zip(TEXTS, results):
        expected_tokens = len(text)
        assert acts.ndim == 2
        assert acts.shape[1] == 32
        assert acts.shape[0] == expected_tokens
        assert acts.dtype == np.float32


def test_extract_keeps_special_tokens_when_requested(tiny_model):
    results = extract_activations(
        TEXTS[:1], tiny_model, StubTokenizer(), layer=0, remove_special_tokens=False
    )
    assert results[0].shape[0] == len(TEXTS[0]) + 2


def test_writer_roundtrip(tmp_path):
    writer = ActivationsWriter(tmp_path, shard_size=2)
    writer.write_batch(["a", "b"], [np.zeros((2, 4)), np.zeros((3, 4))])
    writer.write_batch(["c"], [np.ones((1, 4))])
    writer.close()

    shards = list(tmp_path.glob("shard_*.npz"))
    assert len(shards) == 2

    all_ids, all_acts = [], []
    for ids, acts in writer.iter_shards():
        all_ids.extend(ids.tolist())
        all_acts.extend(acts.tolist())
    assert all_ids == ["a", "b", "c"]
    assert all_acts[2].shape == (1, 4)


def test_writer_rejects_mismatched_lengths(tmp_path):
    writer = ActivationsWriter(tmp_path)
    with pytest.raises(ValueError):
        writer.write_batch(["a"], [np.zeros((2, 4)), np.zeros((3, 4))])
