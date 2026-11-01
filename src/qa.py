"""Citation-bearing answer generation with a retrieval-confidence refusal gate.

The harder problem in document QA is not producing an answer -- it is knowing
when not to. An LLM handed irrelevant context will confidently answer anyway
unless told otherwise, so the refusal decision is made in code, before the
LLM is even called: if the best cosine similarity among the retrieved chunks
falls below `confidence_threshold`, the system returns a fixed "insufficient
evidence" response and never invokes the LLM. This makes the guard testable
with a `ScriptedLLM` that has no queued responses -- if the guard failed to
trigger and the code fell through to a live call, the test would fail loudly
on an empty queue rather than silently returning a plausible-looking
hallucination.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from src.vector_store import ScoredChunk, VectorStore

REFUSAL_TEXT = "This is not covered in the provided policy documents."
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class Citation:
    doc_id: str
    chunk_id: str
    section: str
    page: int | None = None


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    refused: bool
    top_score: float
    passages: list = field(default_factory=list)  # ScoredChunk objects the answer was built from


def _build_prompt(question: str, retrieved: list[ScoredChunk]) -> str:
    context = "\n\n".join(
        f"[{sc.chunk.doc_id} | {sc.chunk.chunk_id}]\n{sc.chunk.text}" for sc in retrieved
    )
    return (
        "Answer the question using only the policy excerpts below. "
        "If the excerpts do not contain the answer, say so explicitly.\n\n"
        f"Excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def answer_question(
    question: str,
    store: VectorStore,
    embed_fn: Callable[[str], np.ndarray],
    llm: LLMClient,
    top_k: int = 5,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> AnswerResult:
    """Retrieve, then either refuse on low confidence or generate a cited answer."""
    query_embedding = embed_fn(question)
    retrieved = store.search(query_embedding, top_k=top_k)
    top_score = retrieved[0].score if retrieved else 0.0

    if not retrieved or top_score < confidence_threshold:
        return AnswerResult(answer=REFUSAL_TEXT, citations=[], refused=True, top_score=top_score)

    prompt = _build_prompt(question, retrieved)
    answer_text = llm.generate(prompt)
    citations = [
        Citation(
            doc_id=sc.chunk.doc_id,
            chunk_id=sc.chunk.chunk_id,
            section=sc.chunk.section,
            page=sc.chunk.page,
        )
        for sc in retrieved
    ]
    return AnswerResult(answer=answer_text, citations=citations, refused=False, top_score=top_score)
