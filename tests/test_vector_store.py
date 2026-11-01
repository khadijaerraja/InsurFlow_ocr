"""VectorStore cosine-similarity search tests using hand-built vectors.

No sentence-transformers model is loaded here: the store's job (rank by
cosine similarity) is independent of where the vectors came from, so it is
tested with small synthetic embeddings for speed and to keep the full test
suite network-free.
"""

import numpy as np

from src.chunking import Chunk
from src.vector_store import VectorStore


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(doc_id="doc.md", chunk_id=chunk_id, section="", text=chunk_id)


def test_search_ranks_by_cosine_similarity():
    store = VectorStore()
    chunks = [_chunk("a"), _chunk("b"), _chunk("c")]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.7071, 0.7071, 0.0],
        ],
        dtype=np.float32,
    )
    store.add(chunks, embeddings)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=3)

    assert [r.chunk.chunk_id for r in results] == ["a", "c", "b"]
    assert results[0].score > results[1].score > results[2].score


def test_search_respects_top_k():
    store = VectorStore()
    chunks = [_chunk(str(i)) for i in range(10)]
    embeddings = np.eye(10, dtype=np.float32)
    store.add(chunks, embeddings)

    query = embeddings[3]
    results = store.search(query, top_k=4)

    assert len(results) == 4
    assert results[0].chunk.chunk_id == "3"


def test_add_rejects_mismatched_lengths():
    import pytest

    store = VectorStore()
    with pytest.raises(ValueError):
        store.add([_chunk("a")], np.zeros((2, 3), dtype=np.float32))


def test_empty_store_search_returns_empty_list():
    store = VectorStore()
    assert store.search(np.array([1.0, 0.0]), top_k=5) == []
