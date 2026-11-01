"""Answer-generation and hallucination-guard tests using ScriptedLLM.

All embeddings here are hand-built unit vectors rather than real
sentence-transformers output, so these tests need no model download, no
network, and no API key -- exactly what the confidence-threshold guard in
src/qa.py is meant to be verifiable without a live LLM.
"""

import numpy as np
import pytest

from src.chunking import Chunk
from src.llm import ScriptedLLM
from src.qa import REFUSAL_TEXT, answer_question
from src.vector_store import VectorStore

COVER_LEVELS_VEC = np.array([1.0, 0.0, 0.0], dtype=np.float32)
EXCLUSIONS_VEC = np.array([0.0, 1.0, 0.0], dtype=np.float32)
UNRELATED_VEC = np.array([0.0, 0.0, 1.0], dtype=np.float32)

EMBEDDING_LOOKUP = {
    "What is the annual claim limit for the Gold Family policy?": COVER_LEVELS_VEC,
    "Are pre-existing conditions excluded?": EXCLUSIONS_VEC,
    "Does any policy cover flight delay compensation?": UNRELATED_VEC,
}


def fake_embed_fn(text: str) -> np.ndarray:
    return EMBEDDING_LOOKUP[text]


@pytest.fixture
def store() -> VectorStore:
    store = VectorStore()
    chunks = [
        Chunk(doc_id="gold_family.md", chunk_id="gold_family.md::cover-levels",
              section="Cover Levels", text="Cover Levels\nAnnual claim limit is USD 500,000."),
        Chunk(doc_id="bronze_individual.md", chunk_id="bronze_individual.md::exclusions",
              section="Exclusions", text="Exclusions\nPre-existing conditions excluded."),
    ]
    embeddings = np.array([COVER_LEVELS_VEC, EXCLUSIONS_VEC], dtype=np.float32)
    store.add(chunks, embeddings)
    return store


def test_grounded_question_returns_cited_answer(store):
    llm = ScriptedLLM(["The annual claim limit is USD 500,000. [gold_family.md]"])
    result = answer_question(
        "What is the annual claim limit for the Gold Family policy?",
        store,
        fake_embed_fn,
        llm,
        top_k=2,
    )
    assert result.refused is False
    assert result.answer == "The annual claim limit is USD 500,000. [gold_family.md]"
    assert result.citations[0].doc_id == "gold_family.md"
    assert result.citations[0].section == "Cover Levels"
    assert len(llm.calls) == 1


def test_unanswerable_question_triggers_refusal_without_calling_llm(store):
    llm = ScriptedLLM([])  # no responses queued: a call here would raise AssertionError
    result = answer_question(
        "Does any policy cover flight delay compensation?",
        store,
        fake_embed_fn,
        llm,
        top_k=2,
        confidence_threshold=0.35,
    )
    assert result.refused is True
    assert result.answer == REFUSAL_TEXT
    assert result.citations == []
    assert llm.calls == []  # the guard short-circuited before any generation call


def test_confidence_threshold_is_the_deciding_factor(store):
    """The same low-relevance retrieval is treated differently by different thresholds."""
    permissive_llm = ScriptedLLM(["An attempted answer despite weak evidence."])
    permissive_result = answer_question(
        "Does any policy cover flight delay compensation?",
        store,
        fake_embed_fn,
        permissive_llm,
        top_k=2,
        confidence_threshold=-1.0,
    )
    assert permissive_result.refused is False
    assert len(permissive_llm.calls) == 1

    strict_llm = ScriptedLLM([])
    strict_result = answer_question(
        "Does any policy cover flight delay compensation?",
        store,
        fake_embed_fn,
        strict_llm,
        top_k=2,
        confidence_threshold=0.35,
    )
    assert strict_result.refused is True
    assert strict_llm.calls == []


def test_empty_store_always_refuses():
    llm = ScriptedLLM([])
    empty_store = VectorStore()
    result = answer_question(
        "Does any policy cover flight delay compensation?",
        empty_store,
        fake_embed_fn,
        llm,
        top_k=2,
    )
    assert result.refused is True
    assert result.top_score == 0.0
