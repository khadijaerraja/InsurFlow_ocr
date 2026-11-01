"""Local sentence-embedding wrapper.

A hosted embedding API would add per-call latency, cost, and an external
dependency to something that runs entirely on synthetic, non-sensitive
documents -- there is no reason to leave the machine. `all-MiniLM-L6-v2` is
small enough (~90MB) to download once and run on CPU in milliseconds per
chunk, which keeps the whole retrieval pipeline offline and free after setup.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

import os

# Any 384-dimension sentence-transformers model works (the database column is vector(384)).
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_model():
    """sentence-transformers if installed, else fastembed (same model through ONNX Runtime,
    no PyTorch -- roughly 2 GB lighter to install)."""
    try:
        from sentence_transformers import SentenceTransformer

        return "st", SentenceTransformer(MODEL_NAME)
    except ImportError:
        from fastembed import TextEmbedding

        return "fastembed", TextEmbedding(MODEL_NAME if "/" in MODEL_NAME else f"sentence-transformers/{MODEL_NAME}")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encode a batch of strings into L2-normalised embedding vectors."""
    backend, model = _get_model()
    if backend == "st":
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    else:
        embeddings = np.asarray(list(model.embed(texts)), dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Encode a single string; a thin convenience wrapper over embed_texts."""
    return embed_texts([text])[0]
