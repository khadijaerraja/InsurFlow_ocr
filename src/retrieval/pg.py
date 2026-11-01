"""Hybrid search inside PostgreSQL: pgvector (semantic) + full-text index (keywords), fused by relative scores.

* Dense: `embedding <=> query` (cosine distance) served by an HNSW index, with iterative scans so that the
  owner/document filters never starve the result list.
* Keyword: a generated `tsvector` column (English and French share the 'simple' configuration, accents folded,
  hyphens read as spaces) and prefix queries (`expir:*` matches expiry, expire, expiration). The GIN index finds
  the candidates; they are ranked by the sum of the IDF of the query terms they contain (rare terms such as an
  identifier weigh more than "contract"), computed from the owner's own document frequencies.
* Fusion: weighted min-max normalised scores (see hybrid.py), keyword 0.7 / semantic 0.3.

Every query is scoped by `owner_id`: a searcher is created for one user and cannot see another user's passages.
All values are bound parameters; the only text built into SQL is the list of `:t<i>` placeholders.
"""

from __future__ import annotations

import math
import re

import sqlalchemy as sa

from src.chunking import Chunk
from src.retrieval.bm25 import raw_terms, tokenize
from src.retrieval.hybrid import relative_score_fusion
from src.retrieval.search import DENSE_WEIGHT, KEYWORD_WEIGHT, MODES
from src.vector_store import ScoredChunk

_COLUMNS = "c.id, c.document_id, c.page_number, c.text, d.file_name, d.doc_type, d.policy_number"
_FROM = ("FROM chunks c JOIN documents d ON d.id = c.document_id "
         "WHERE c.owner_id = :owner AND d.status IN ('processed', 'needs_review')")
_SAFE_TERM = re.compile(r"^[a-z0-9]+$")


def _terms(query: str) -> list[str]:
    """'when does the contract expire' -> ['contract:*', 'expir:*'] (prefix terms; short ones are exact)."""
    out = []
    for term in dict.fromkeys(raw_terms(query)):
        if _SAFE_TERM.match(term):
            out.append(f"{term[:5]}:*" if len(term) >= 5 else term)
    return out


class PgSearcher:
    def __init__(self, engine: sa.Engine, owner_id: int, embed_query=None) -> None:
        self._engine, self._owner, self._embed_query = engine, owner_id, embed_query

    @property
    def has_dense(self) -> bool:
        return self._embed_query is not None

    # ----------------------------------------------------------------- helpers
    def _scope(self, flt) -> tuple[str, dict]:
        sql, params = "", {"owner": self._owner}
        if flt:
            if flt.document_id is not None:
                sql += " AND c.document_id = :doc"; params["doc"] = flt.document_id
            if flt.contract is not None:
                sql += " AND d.policy_number = :contract"; params["contract"] = flt.contract
            if flt.doc_type is not None:
                sql += " AND d.doc_type = :doc_type"; params["doc_type"] = flt.doc_type
        return sql, params

    @staticmethod
    def _chunk(row) -> Chunk:
        return Chunk(doc_id=row.file_name, chunk_id=f"c{row.id}", section=row.doc_type, text=row.text,
                     page=row.page_number, document_id=row.document_id, contract=row.policy_number)

    def _idf(self, conn, terms: list[str]) -> list[float]:
        """IDF of each query term over this owner's passages (one aggregate query)."""
        counts = ", ".join(f"count(*) FILTER (WHERE c.tsv @@ to_tsquery('simple', :t{i})) AS d{i}"
                           for i in range(len(terms)))
        row = conn.execute(sa.text(f"SELECT count(*) AS n, {counts} {_FROM}"),
                           {"owner": self._owner, **{f"t{i}": t for i, t in enumerate(terms)}}).one()
        return [math.log(1 + (row.n - row[i + 1] + 0.5) / (row[i + 1] + 0.5)) for i in range(len(terms))]

    # ------------------------------------------------------------------ search
    def keyword_search(self, query: str, top_k: int, flt=None) -> list[ScoredChunk]:
        terms = _terms(query)
        if not terms:
            return []
        scope, params = self._scope(flt)
        with self._engine.connect() as conn:
            idf = self._idf(conn, terms)
            bound = {f"t{i}": t for i, t in enumerate(terms)} | {f"w{i}": w for i, w in enumerate(idf)}
            score = " + ".join(f"CASE WHEN c.tsv @@ to_tsquery('simple', :t{i}) THEN :w{i} ELSE 0 END"
                               for i in range(len(terms)))
            sql = (f"SELECT {_COLUMNS}, ({score}) AS score {_FROM} AND c.tsv @@ to_tsquery('simple', :any){scope} "
                   "ORDER BY score DESC, c.id LIMIT :k")
            rows = conn.execute(sa.text(sql), {**params, **bound, "any": " | ".join(terms), "k": top_k}).all()
        return [ScoredChunk(self._chunk(r), float(r.score)) for r in rows]

    def dense_search(self, query: str, top_k: int, flt=None) -> list[ScoredChunk]:
        if self._embed_query is None:
            return []
        vector = "[" + ",".join(f"{x:.6f}" for x in self._embed_query(query)) + "]"
        scope, params = self._scope(flt)
        sql = (f"SELECT {_COLUMNS}, 1 - (c.embedding <=> CAST(:qv AS vector)) AS score {_FROM} "
               f"AND c.embedding IS NOT NULL{scope} ORDER BY c.embedding <=> CAST(:qv AS vector) LIMIT :k")
        with self._engine.begin() as conn:
            conn.execute(sa.text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            conn.execute(sa.text("SET LOCAL hnsw.ef_search = 100"))
            rows = conn.execute(sa.text(sql), {**params, "qv": vector, "k": top_k}).all()
        return [ScoredChunk(self._chunk(r), float(r.score)) for r in rows]

    def search(self, query: str, mode: str = "hybrid", top_k: int = 5, flt=None) -> list[ScoredChunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        pool = max(top_k * 4, 20)
        if mode == "bm25" or not self.has_dense:
            return self.keyword_search(query, top_k, flt)
        if mode == "dense":
            return self.dense_search(query, top_k, flt)
        return relative_score_fusion(
            [self.keyword_search(query, pool, flt), self.dense_search(query, pool, flt)], top_k=top_k,
            weights=[KEYWORD_WEIGHT, DENSE_WEIGHT])

    def coverage(self, query: str, flt=None) -> float:
        """Largest share of the question's terms found together in one of the best keyword matches."""
        terms = set(tokenize(query))
        if not terms:
            return 0.0
        hits = self.keyword_search(query, 25, flt)
        return max((len(terms & set(tokenize(h.chunk.text))) / len(terms) for h in hits), default=0.0)

    def known_identifiers(self) -> set[str]:
        """Contract / claim / invoice / amendment numbers that exist in this user's library."""
        with self._engine.connect() as conn:
            rows = conn.execute(sa.text(
                "SELECT DISTINCT f.value FROM extracted_fields f JOIN documents d ON d.id = f.document_id "
                "WHERE d.owner_id = :owner AND f.name IN ('policy_number', 'reference')"), {"owner": self._owner})
            return {r[0] for r in rows}
