"""
embeddings.py — text embedding abstraction.

Uses sentence-transformers (all-MiniLM-L6-v2):
  - 384 dimensions
  - ~80 MB model, downloaded once to ~/.cache/huggingface/
  - ~5 ms/query on CPU — fast enough for interactive use
  - normalize_embeddings=True → unit-norm vectors
    → dot product == cosine similarity (required by FAISS IndexFlatIP)

The model is a module-level singleton (lazy-loaded on first use)
so import time stays fast.
"""
from __future__ import annotations
import numpy as np
from typing import List

_model = None       # lazy singleton — real sentence-transformers model
_use_mock = False   # set to True if model download fails


def _get_model():
    """
    Try to load sentence-transformers; fall back to a hash-based mock embedder
    if the model cannot be downloaded (e.g. in offline CI environments).

    The mock embedder is deterministic and preserves rough keyword similarity
    well enough for unit tests.  It should NOT be used in production.
    """
    global _model, _use_mock
    if _model is not None or _use_mock:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "sentence-transformers model unavailable — using hash-based mock embedder. "
            "Semantic similarity will be approximate.  Install/download the model for production."
        )
        _use_mock = True
        _model = None
    return _model


def _mock_embed(text: str) -> np.ndarray:
    """
    Deterministic hash-based fallback embedding.
    Uses character-level 2-gram frequency vectors, L2-normalized to unit length.
    Cheap, offline, and good enough for tests.  Not for production.
    """
    import hashlib
    vec = np.zeros(384, dtype=np.float32)
    # Distribute bigram hashes across the 384 dimensions
    for i in range(0, len(text) - 1):
        bigram = text[i : i + 2].encode()
        idx = int(hashlib.md5(bigram).hexdigest(), 16) % 384
        vec[idx] += 1.0
    # Single character hashes as well
    for ch in text:
        idx = int(hashlib.md5(ch.encode()).hexdigest(), 16) % 384
        vec[idx] += 0.5
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 0 else vec


def embed(text: str) -> np.ndarray:
    """
    Return a 384-dim float32 unit-norm embedding for `text`.
    Uses sentence-transformers when available; falls back to hash-based mock.
    """
    model = _get_model()
    if _use_mock or model is None:
        return _mock_embed(text)
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)


def embed_batch(texts: List[str]) -> np.ndarray:
    """
    Embed multiple strings efficiently.
    Returns an (N, 384) float32 array.
    """
    model = _get_model()
    if _use_mock or model is None:
        return np.stack([_mock_embed(t) for t in texts])
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs.astype(np.float32)
