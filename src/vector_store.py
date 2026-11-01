"""In-memory vector store for a ~150-chunk corpus.

At this scale (a few hundred vectors) a dedicated vector database buys
nothing: the entire embedding matrix fits in a few hundred KB and a brute
-force cosine similarity scan over it runs in well under a millisecond on
CPU. Pulling in Chroma/Pinecone/FAISS-as-a-service here would add an
external service dependency to solve a problem numpy already solves in one
matrix multiply -- the kind of premature infrastructure that shows up poorly
in a system-design interview.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore:
    """Holds L2-normalised embeddings in a single numpy array for cosine search."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings must have matching lengths")
        self._chunks.extend(chunks)
        self._embeddings = (
            embeddings if self._embeddings is None else np.vstack([self._embeddings, embeddings])
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[ScoredChunk]:
        """Return the top_k chunks by cosine similarity (embeddings assumed normalised)."""
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        scores = self._embeddings @ query_embedding
        top_k = min(top_k, len(self._chunks))
        top_indices = np.argsort(-scores)[:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(scores[i])) for i in top_indices]
