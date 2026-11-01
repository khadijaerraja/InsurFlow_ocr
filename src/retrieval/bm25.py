"""BM25 keyword search, implemented in ~40 lines so it can be explained line by line.

Dense embeddings capture meaning but blur exact tokens: "CLM-48213" and
"CLM-48231" embed almost identically. Insurance work is full of identifiers
(policy numbers, claim references), which is exactly where keyword scoring is
strong -- hence the hybrid in hybrid.py.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from src.chunking import Chunk
from src.retrieval.text import fold
from src.vector_store import ScoredChunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Question words and glue words carry no evidence; dropping them keeps BM25 scores meaningful
# (otherwise "what is the ..." alone gives every question a score and weakens the refusal gate).
STOPWORDS = frozenset(
    "a an the of in on at to for and or is are was were be been what which who whom how when where why does do "
    "did any under by with from this that these those it its as there their his her has have had can could should "
    "would please tell me about my i you your summarise summarize summary explain list show give describe briefly "
    "short two three lines line details detail much many kind kinds type types sort whether within made "
    # French question and glue words
    "le la les un une des du de et ou est sont quel quelle quels quelles quand comment que qui quoi pour dans sur par "
    "avec au aux ce cette ces mon ma mes son sa ses votre vos donne moi dis resume d l n s c j m t qu".split()
)


# Everyday words for the same insurance object: people say "policy", documents say "contract".
SYNONYMS = {"policy": "contract", "policies": "contracts", "benefits": "guarantees", "benefit": "guarantee",
            "debut": "effet"}  # French contracts say "date d'effet" where people ask for the "date de début"


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens without stopwords; identifiers like INS-001 split on dashes
    consistently for both documents and queries."""
    return [_stem(t) for t in raw_terms(text)]


def raw_terms(text: str) -> list[str]:
    """Words of a text without stopwords, accents folded, synonyms applied, NOT stemmed.
    PostgreSQL full-text search uses these and applies the same prefix rule through `term:*` (see pg.py)."""
    return [SYNONYMS.get(t, t) for t in _TOKEN_RE.findall(fold(text)) if t not in STOPWORDS]


def _stem(token: str) -> str:
    """Crude stemming: keep the first 5 letters of long words so "excluded" matches "exclusions" and
    "claimed" matches "claim". Short tokens and numbers (INS-001 -> ins, 001) are untouched."""
    return token[:5] if len(token) > 5 and token.isalpha() else token


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    norm = 1 - self._b + self._b * self._len[i] / self._avg_len
                    score += self._idf[t] * f * (self._k1 + 1) / (f + self._k1 * norm)
            if score > 0:
                scored.append(ScoredChunk(chunk=self._chunks[i], score=score))
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]

    def coverage(self, query: str, chunk_filter=None) -> float:
        """Largest share of the query's distinct terms found together in a single passage (0 to 1).

        A question can score well on one generic word ("covered") while its real subject
        ("flight delay compensation") appears nowhere; this exposes that. With `chunk_filter`
        only the passages of the selected document or contract are considered."""
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        return max((len(terms & tf.keys()) / len(terms) for c, tf in zip(self._chunks, self._tf)
                    if chunk_filter is None or chunk_filter(c)), default=0.0)
