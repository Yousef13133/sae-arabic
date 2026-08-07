"""Arabic data loading and preprocessing (Phase 1, Week 2).

Implements Arabic-specific preprocessing (diacritic handling, orthographic
normalization), tokenization, and the text-to-activation data pipeline.
Supports HuggingFace datasets (e.g., NADI/MADAR) and local files.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from datasets import Dataset, load_dataset

_TRANSLIT = str.maketrans(
    {
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0624": "\u0648",
        "\u0626": "\u064A",
        "\u0649": "\u064A",
    }
)


def strip_diacritics(text: str) -> str:
    """Remove Arabic diacritics and tatweel from ``text``."""
    return "".join(c for c in text if not unicodedata.combining(c) and c != "\u0640")


def normalize_arabic(text: str) -> str:
    """Normalize alef/hamza variants, then strip diacritics."""
    return strip_diacritics(text.translate(_TRANSLIT))


@dataclass
class ArabicTokenizer:
    """Wraps a HuggingFace tokenizer with Arabic-specific preprocessing."""

    tokenizer: object | None = None
    text_column: str = "text"
    max_length: int = 128
    strip_diacritics: bool = True
    normalize_alef: bool = True

    def normalize(self, text: str) -> str:
        """Apply diacritic stripping and orthographic normalization."""
        out = text
        if self.normalize_alef:
            out = out.translate(_TRANSLIT)
        if self.strip_diacritics:
            out = strip_diacritics(out)
        return out

    def __call__(self, batch: dict) -> dict:
        """Map a text batch to normalized (and optionally tokenized) text."""
        texts = [self.normalize(t) for t in batch[self.text_column]]
        if self.tokenizer is None:
            return {**batch, self.text_column: texts}
        encoded = self.tokenizer(
            texts, truncation=True, padding=True, max_length=self.max_length
        )
        return {**batch, **encoded}


@dataclass
class ArabicDatasetLoader:
    """Loads NADI/MADAR subsets into a Dataset.

    ``source`` may be ``"huggingface"`` (``path`` = HF dataset name, optional
    ``name`` = config/subset) or ``"local"`` (``path`` = jsonl/csv/json file).
    """

    source: str = "huggingface"
    path: str = ""
    name: str | None = None
    split: str = "train"
    text_column: str = "text"
    dialect_column: str | None = "dialect"

    def load(self) -> Dataset:
        """Download and return the dataset (validated during Phase 0)."""
        if self.source == "huggingface":
            if not self.path:
                raise ValueError("path must name a HuggingFace dataset")
            kwargs = {"split": self.split}
            if self.name:
                kwargs["name"] = self.name
            return load_dataset(self.path, **kwargs)
        if self.source == "local":
            if not self.path:
                raise ValueError("path must point to a local dataset file")
            fmt = "json" if self.path.endswith(".jsonl") else self.path.rsplit(".", 1)[-1]
            return load_dataset(fmt, data_files=self.path, split=self.split)
        raise ValueError(f"unknown source: {self.source}")
