"""Gives each user a searcher over their own passages.

PostgreSQL: a `PgSearcher` per request (the database is the index, nothing to keep in sync).
SQLite fallback: an in-memory searcher cached per user and rebuilt after an import or a deletion.

Dense search needs the embedding model; if it cannot load (not installed, offline on first run) the
searcher degrades to keyword search and `dense_error` says why. `embedder=(embed_texts, embed_query)` injects
another implementation (tests use a small deterministic one).
"""

from __future__ import annotations

import threading

import sqlalchemy as sa

from src.pipeline import backfill_embeddings
from src.retrieval.corpus import db_chunks
from src.retrieval.pg import PgSearcher
from src.retrieval.search import Searcher


class SearchIndex:
    def __init__(self, engine: sa.Engine, use_dense: bool = True, embedder=None) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._cache: dict[int, Searcher] = {}
        self._embed_texts = self._embed_query = None
        self.dense_error: str | None = None
        if embedder is not None:
            self._embed_texts, self._embed_query = embedder
        elif use_dense:
            try:
                from src.embeddings import embed_query, embed_texts

                self._embed_texts, self._embed_query = embed_texts, embed_query
            except Exception as exc:  # embedding library missing
                self.dense_error = f"{type(exc).__name__}: {exc}"[:300]

    @property
    def embed_texts(self):
        return self._embed_texts

    @property
    def embed_query(self):
        return self._embed_query

    @property
    def postgres(self) -> bool:
        return self._engine.dialect.name == "postgresql"

    def refresh(self, owner_id: int | None = None) -> None:
        with self._lock:
            if owner_id is None:
                self._cache.clear()
            else:
                self._cache.pop(owner_id, None)

    def get(self, owner_id: int):
        if self.postgres:
            return PgSearcher(self._engine, owner_id, self._embed_query)
        with self._lock:
            if owner_id not in self._cache:
                chunks, vectors = db_chunks(self._engine, owner_id)
                if self._embed_texts and chunks and vectors is None:
                    try:
                        backfill_embeddings(self._engine, self._embed_texts)
                        chunks, vectors = db_chunks(self._engine, owner_id)
                    except Exception as exc:
                        self.dense_error = f"{type(exc).__name__}: {exc}"[:300]
                dense = self._embed_query is not None and vectors is not None
                self._cache[owner_id] = Searcher(chunks, embed_query=self._embed_query if dense else None,
                                                 embeddings=vectors if dense else None)
            return self._cache[owner_id]
