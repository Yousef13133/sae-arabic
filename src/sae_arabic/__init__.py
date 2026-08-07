"""Sparse Autoencoder toolkit for Arabic LLMs."""

from sae_arabic.activations import ActivationsWriter, extract_activations
from sae_arabic.aldi import aldi_score, causal_scrub, feature_aldicorrelation
from sae_arabic.analysis import contexts_for_features, top_features
from sae_arabic.data import ArabicDatasetLoader, ArabicTokenizer
from sae_arabic.sae import (
    SparseAutoencoder,
    load_checkpoint,
    save_checkpoint,
    train_sae,
)

__version__ = "0.1.0"

__all__ = [
    "ActivationsWriter",
    "ArabicDatasetLoader",
    "ArabicTokenizer",
    "SparseAutoencoder",
    "aldi_score",
    "causal_scrub",
    "contexts_for_features",
    "extract_activations",
    "feature_aldicorrelation",
    "load_checkpoint",
    "save_checkpoint",
    "top_features",
    "train_sae",
]
