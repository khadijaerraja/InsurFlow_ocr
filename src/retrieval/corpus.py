"""Load one user's passages from the database (used by the in-memory searcher on the SQLite fallback).

On PostgreSQL the searcher queries the database directly (pg.py), so nothing is loaded into memory there.
"""

from __future__ import annotations

import pathlib

import numpy as np
import sqlalchemy as sa

from src.chunking import Chunk, section_aware_chunks
from src.database import schema as db


def db_chunks(engine: sa.Engine, owner_id: int) -> tuple[list[Chunk], np.ndarray | None]:
    """The owner's searchable passages (processed or needs_review documents) and their vectors, if all exist."""
    query = (
        sa.select(db.chunks.c.id, db.chunks.c.document_id, db.chunks.c.page_number, db.chunks.c.text,
                  db.chunks.c.embedding, db.documents.c.file_name, db.documents.c.doc_type,
                  db.documents.c.policy_number)
        .join(db.documents, db.documents.c.id == db.chunks.c.document_id)
        .where(db.chunks.c.owner_id == owner_id, db.documents.c.status.in_(("processed", "needs_review")))
        .order_by(db.documents.c.id, db.chunks.c.page_number, db.chunks.c.id)
    )
    chunks, vectors = [], []
    with engine.connect() as conn:
        for cid, doc_id, page, text, emb, name, doc_type, contract in conn.execute(query):
            chunks.append(Chunk(doc_id=name, chunk_id=f"c{cid}", section=doc_type, text=text,
                                page=page, document_id=doc_id, contract=contract))
            vectors.append(emb)
    if chunks and all(v is not None for v in vectors):
        return chunks, np.vstack(vectors)
    return chunks, None


def policy_chunks(policy_dir: pathlib.Path) -> list[Chunk]:
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks.extend(section_aware_chunks(path.read_text(encoding="utf-8"), path.name))
    return chunks
