"""Shared offline test helpers (tiny random BERT + stub tokenizer)."""

from typing import ClassVar

import torch
from transformers import BertConfig, BertForMaskedLM

CLS, PAD, SEP = 1, 0, 2
MASK = 999


def tiny_bert() -> BertForMaskedLM:
    config = BertConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = BertForMaskedLM(config)
    model.eval()
    return model


class StubTokenizer:
    """Char-level stub exposing the subset of the HF API we use."""

    all_special_ids: ClassVar[list[int]] = [CLS, PAD, SEP]
    mask_token_id: ClassVar[int] = MASK
    pad_token_id: ClassVar[int] = PAD

    def __init__(self) -> None:
        self._id_to_char: dict[int, str] = {}

    def _encode(self, text: str) -> list[int]:
        ids = []
        for c in text:
            code = ord(c) % 996 + 3
            self._id_to_char.setdefault(code, c)
            ids.append(code)
        return ids

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors=None):
        rows = [[CLS] + self._encode(t) + [SEP] for t in texts]
        seq_len = max(len(r) for r in rows)
        input_ids, attention_mask = [], []
        for row in rows:
            if truncation and max_length and len(row) > max_length:
                row = row[:max_length]
            row = row[:seq_len] + [PAD] * (seq_len - len(row))
            input_ids.append(row)
            attention_mask.append([1 if t != PAD else 0 for t in row])
        tensors = {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(attention_mask)}
        return type("Encoding", (dict,), {})(tensors)

    def batch_decode(self, ids, skip_special_tokens=True):
        decoded = []
        for row in ids:
            if torch.is_tensor(row):
                row = row.tolist()
            chars = []
            for t in row:
                if skip_special_tokens and t in self.all_special_ids:
                    continue
                chars.append(self._id_to_char.get(t, "?"))
            decoded.append("".join(chars))
        return decoded
