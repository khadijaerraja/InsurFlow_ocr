"""Hybrid retrieval: fuse keyword (BM25) and dense rankings with Reciprocal Rank Fusion.

BM25 scores and cosine similarities live on different scales, so adding them
is meaningless. RRF uses only ranks: score = sum over rankers of weight/(k + rank).
It rewards chunks that both rankers like. Weights let the more reliable ranker
count more: on insurance documents full of identifiers (INS-001, SIN-003) keyword
search is the stronger signal, so the Searcher weights it 3:1 (chosen with a sweep over
1:1, 2:1, 3:1 on the evaluation questions -- a small set, so treat it as a starting point).
"""

from __future__ import annotations

from src.vector_store import ScoredChunk

RRF_K = 60


def reciprocal_rank_fusion(rankings: list[list[ScoredChunk]], top_k: int = 5, k: int = RRF_K,
                           weights: list[float] | None = None) -> list[ScoredChunk]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    by_id = {}
    for ranking, weight in zip(rankings, weights):
        for rank, scored in enumerate(ranking, start=1):
            cid = scored.chunk.chunk_id
            by_id[cid] = scored.chunk
            fused[cid] = fused.get(cid, 0.0) + weight / (k + rank)
    ordered = sorted(fused, key=lambda cid: -fused[cid])[:top_k]
    return [ScoredChunk(chunk=by_id[cid], score=fused[cid]) for cid in ordered]


def relative_score_fusion(rankings: list[list[ScoredChunk]], top_k: int = 5,
                          weights: list[float] | None = None) -> list[ScoredChunk]:
    """Weighted sum of min-max normalised scores (a convex combination), the "relative score fusion" used by
    several search engines.

    Rank fusion (RRF) throws away how far ahead a result is: a passage that is the clear keyword winner can lose
    to one that is merely 3rd and 4th in both lists. Normalising each ranker's scores to 0..1 over its own
    candidate list keeps that margin, so an exact identifier match is not outvoted by a vague semantic one.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    by_id = {}
    for ranking, weight in zip(rankings, weights):
        if not ranking:
            continue
        low, high = min(s.score for s in ranking), max(s.score for s in ranking)
        for scored in ranking:
            cid = scored.chunk.chunk_id
            by_id[cid] = scored.chunk
            normalised = (scored.score - low) / (high - low) if high > low else 1.0
            fused[cid] = fused.get(cid, 0.0) + weight * normalised
    ordered = sorted(fused, key=lambda cid: -fused[cid])[:top_k]
    return [ScoredChunk(chunk=by_id[cid], score=fused[cid]) for cid in ordered]
