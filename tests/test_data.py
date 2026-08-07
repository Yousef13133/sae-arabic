"""Tests for Arabic preprocessing and dataset loading."""

import json

from sae_arabic.data import ArabicDatasetLoader, ArabicTokenizer, normalize_arabic, strip_diacritics


def test_strip_diacritics():
    assert strip_diacritics("كِتَابٌ") == "كتاب"


def test_normalize_alef_variants():
    assert normalize_arabic("أحمد إبراهيم") == "احمد ابراهيم"


def test_normalize_hamza():
    assert normalize_arabic("مؤمن لؤلؤة") == "مومن لولوة"


def test_tokenizer_normalize_flags():
    tok = ArabicTokenizer(normalize_alef=False, strip_diacritics=False)
    assert tok.normalize("كِتَابٌ") == "كِتَابٌ"
    assert tok.normalize("أحمد") == "أحمد"
    tok.strip_diacritics = True
    assert tok.normalize("كِتَابٌ") == "كتاب"


def test_tokenizer_call_without_hf():
    tok = ArabicTokenizer()
    out = tok({tok.text_column: ["أحمد", "مُحَمَّد"]})
    assert out["text"] == ["احمد", "محمد"]


def test_local_dataset_loader(tmp_path):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"text": text, "dialect": dialect}, ensure_ascii=False) + "\n" for text, dialect in [("هذا نص", "north"), ("نص آخر", "gulf")])
    loader = ArabicDatasetLoader(source="local", path=str(path))
    ds = loader.load()
    assert len(ds) == 2
    assert ds[0]["text"] == "هذا نص"
    assert loader.dialect_column in ds.column_names


def test_hf_loader_requires_path():
    loader = ArabicDatasetLoader(source="huggingface")
    try:
        loader.load()
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty path")
