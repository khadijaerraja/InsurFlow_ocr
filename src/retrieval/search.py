"""In-memory hybrid searcher (BM25 + NumPy cosine + rank fusion).

Used for the SQLite fallback and for unit tests; production search runs inside PostgreSQL (see pg.py).
Both expose the same methods, so the answer logic does not care which one it talks to:
`keyword_search`, `dense_search`, `search`, `coverage`, `known_identifiers`, `has_dense`.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from src.chunking import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import relative_score_fusion
from src.vector_store import ScoredChunk, VectorStore

MODES = ("bm25", "dense", "hybrid")
KEYWORD_WEIGHT, DENSE_WEIGHT = 0.7, 0.3  # hybrid fusion weights: identifiers make keywords the stronger signal
_ID_RE = re.compile(r"\b(INS|SIN|FAC|AVN)[-\s]?(\d{3,6})\b", re.I)


def identifiers_in(text: str) -> set[str]:
    return {f"{p.upper()}-{int(n):03d}" for p, n in _ID_RE.findall(text)}


class Searcher:
    """`embed_query` is injected so tests run without the embedding model.
    With no embedder the searcher is keyword-only and 'dense'/'hybrid' degrade to BM25."""

    def __init__(
        self,
        chunks: list[Chunk],
        embed_texts: Callable[[list[str]], np.ndarray] | None = None,
        embed_query: Callable[[str], np.ndarray] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self.bm25 = BM25Index(chunks)
        self.store: VectorStore | None = None
        self._embed_query = embed_query
        if embed_query is not None and chunks and (embeddings is not None or embed_texts is not None):
            self.store = VectorStore()
            self.store.add(chunks, embeddings if embeddings is not None else embed_texts([c.text for c in chunks]))

    @property
    def has_dense(self) -> bool:
        return self.store is not None

    def keyword_search(self, query: str, top_k: int, flt=None):
        ranked = self.bm25.search(query, len(self.bm25._chunks) if flt else top_k)
        return [sc for sc in ranked if not flt or flt(sc.chunk)][:top_k]

    def dense_search(self, query: str, top_k: int, flt=None):
        if self.store is None or self._embed_query is None:
            return []
        ranked = self.store.search(self._embed_query(query), top_k=len(self.bm25._chunks) if flt else top_k)
        return [sc for sc in ranked if not flt or flt(sc.chunk)][:top_k]

    def coverage(self, query: str, flt=None) -> float:
        return self.bm25.coverage(query, flt or None)

    def known_identifiers(self) -> set[str]:
        return identifiers_in(" ".join(c.text for c in self.bm25._chunks))

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5, flt=None) -> list[ScoredChunk]:
        """`flt` (document, contract, type) is applied before cutting to top_k."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        pool = len(self.bm25._chunks) if flt else max(top_k * 3, 10)
        if mode == "bm25" or not self.has_dense:
            return self.keyword_search(query, top_k, flt)
        if mode == "dense":
            return self.dense_search(query, top_k, flt)
        return relative_score_fusion(
            [self.keyword_search(query, pool, flt), self.dense_search(query, pool, flt)], top_k=top_k,
            weights=[KEYWORD_WEIGHT, DENSE_WEIGHT])
