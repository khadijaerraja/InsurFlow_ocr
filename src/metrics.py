"""Retrieval quality metrics: hit rate and mean reciprocal rank.

Both operate on document-level matches (does the expected doc_id appear
among the retrieved chunks?) rather than exact chunk matches, since the two
chunking strategies under comparison produce different chunk boundaries for
the same source document -- scoring at the chunk level would not be a fair
comparison between them.
"""

from __future__ import annotations

from src.vector_store import ScoredChunk


def reciprocal_rank(retrieved: list[ScoredChunk], expected_doc: str) -> float:
    """1/rank of the first chunk whose doc_id matches, else 0.0."""
    for rank, scored in enumerate(retrieved, start=1):
        if scored.chunk.doc_id == expected_doc:
            return 1.0 / rank
    return 0.0


def is_hit(retrieved: list[ScoredChunk], expected_doc: str) -> bool:
    """Whether expected_doc appears anywhere in the retrieved (top-k) set."""
    return any(scored.chunk.doc_id == expected_doc for scored in retrieved)


def hit_rate(results: list[bool]) -> float:
    if not results:
        return 0.0
    return sum(results) / len(results)


def mean_reciprocal_rank(scores: list[float]) -> float:
    if not scores:
        return 0.0
    return sum(scores) / len(scores)
