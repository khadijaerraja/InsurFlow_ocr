"""Conversation memory, designed after LangChain's documented model.

Short-term memory (LangChain: thread-scoped state kept by a checkpointer; trimmed or summarised before the model call)
  * The whole thread lives in the database (`messages`): that is the persisted state, so a conversation resumes
    at any time.
  * What is *sent to the model* is bounded: the most recent messages within a token budget (trimming), plus a
    rolling summary of everything older. The summary is produced when the thread grows past a trigger
    (`SUMMARY_TRIGGER_TOKENS`) and always keeps the last `KEEP_LAST` messages verbatim, like
    SummarizationMiddleware's `trigger` / `keep` settings.

Long-term memory (LangChain: a store shared across threads, namespaced by user, JSON documents, semantic search)
  * `user_memories` is that store: namespace = user id, one narrow document per fact (a "collection", not one
    ever-growing profile), embedded with the same model as the documents and searched semantically.
  * Memories are written *in the background* after the answer is sent (no added latency) and read at question
    time: the few most relevant ones are injected as context. Semantic memories hold facts and preferences
    ("the user writes in French"); episodic memories hold what was asked ("asked about contract INS-001").
  * The user can switch memory off, list it and delete any item.

Memory is context, never evidence: answers still have to be supported by retrieved document passages.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

import numpy as np
import sqlalchemy as sa

from src.database import schema as db
from src.llm import ExtractiveLLM

KEEP_LAST = 6                 # messages always sent verbatim
CONTEXT_BUDGET_TOKENS = 700   # budget for the verbatim messages (trimming)
SUMMARY_TRIGGER_TOKENS = 900  # summarise older messages once the unsummarised thread exceeds this
RECALL_K = 3                  # memories injected per question (LangChain's example uses 3)
RECALL_MIN_SIMILARITY = 0.35
DUPLICATE_SIMILARITY = 0.92
MAX_MEMORIES_PER_USER = 200

_ID_RE = re.compile(r"\b(INS|SIN|FAC|AVN)[-\s]?(\d{3,6})\b", re.I)
_FRENCH_WORDS = re.compile(r"\b(le|la|les|des|du|est|quel|quelle|quand|mon|ma|mes|contrat|sinistre|facture|garanties|"
                           r"exclusions|plafond|montant|date|combien|comment|pourquoi)\b", re.I)
_PERSONAL_CUE = re.compile(r"\b(i am|i'm|my|i prefer|i work|always|never|please answer|je suis|mon|ma|mes|je prefere|"
                           r"je préfère|toujours|jamais)\b", re.I)


def approx_tokens(text: str) -> int:
    """Rough token count (about four characters per token): enough to budget a prompt without a tokenizer."""
    return len(text) // 4 + 1


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[\d+\]|[*]", "", text)).strip()


# ======================================================================== short-term
def build_context(summary: str, messages: list[dict]) -> str:
    """The text given to the model about the current thread: summary + the latest messages within the budget."""
    recent: list[str] = []
    used = 0
    for m in reversed(messages[-KEEP_LAST:]):  # trimming: newest first until the budget is spent (keep at least one)
        line = f"{'User' if m['role'] == 'user' else 'Assistant'}: {_clean(m['content'])[:300]}"
        cost = approx_tokens(line)
        if recent and used + cost > CONTEXT_BUDGET_TOKENS:
            break
        recent.append(line)
        used += cost
    lines = [f"Summary of earlier discussion: {summary}"] if summary else []
    return "\n".join(lines + list(reversed(recent)))


def _summarise(previous: str, new_messages: list[dict], llm) -> str:
    asked = [_clean(m["content"])[:90] for m in new_messages if m["role"] == "user"]
    plain = " ".join(filter(None, [previous, "The user also asked: " + "; ".join(asked[-4:]) + "."]))[-900:]
    if llm is None or isinstance(llm, ExtractiveLLM):
        return plain  # no language model available: a deterministic summary of the questions
    transcript = "\n".join(f"{m['role']}: {_clean(m['content'])[:300]}" for m in new_messages)
    prompt = ("Summarise this conversation about insurance documents in at most 3 short sentences. Keep contract "
              f"numbers and the facts that were established.\nPrevious summary: {previous or 'none'}\n"
              f"New messages:\n{transcript}\nSummary:")
    try:
        return llm.generate(prompt).strip()[:900] or plain
    except Exception:
        return plain


def maybe_summarise(engine: sa.Engine, conv_id: int, messages: list[dict], summary: str, summary_upto: int, llm) -> None:
    """Fold older messages into the rolling summary once the thread passes the trigger (keeping the last few)."""
    pending = [m for m in messages if m["id"] > summary_upto]
    if sum(approx_tokens(m["content"]) for m in pending) <= SUMMARY_TRIGGER_TOKENS or len(pending) <= KEEP_LAST:
        return
    older = pending[:-KEEP_LAST]
    new_summary = _summarise(summary, older, llm)
    with engine.begin() as conn:
        conn.execute(sa.update(db.conversations).where(db.conversations.c.id == conv_id).values(
            summary=new_summary, summary_upto=older[-1]["id"]))


# ========================================================================= long-term
def _vector_literal(vector) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def memory_enabled(engine: sa.Engine, user_id: int) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(sa.select(db.users.c.memory_enabled).where(db.users.c.id == user_id)).scalar())


def _similar(engine: sa.Engine, user_id: int, vector, limit: int = 3) -> list[tuple[int, str, float]]:
    """The user's memories closest to `vector`: (id, content, cosine similarity), best first."""
    if engine.dialect.name == "postgresql":
        with engine.connect() as conn:
            rows = conn.execute(sa.text(
                "SELECT id, content, 1 - (embedding <=> CAST(:q AS vector)) AS sim FROM user_memories "
                "WHERE user_id = :u AND embedding IS NOT NULL ORDER BY embedding <=> CAST(:q AS vector) LIMIT :k"),
                {"q": _vector_literal(vector), "u": user_id, "k": limit}).all()
        return [(r.id, r.content, float(r.sim)) for r in rows]
    with engine.connect() as conn:
        rows = conn.execute(sa.select(db.user_memories.c.id, db.user_memories.c.content, db.user_memories.c.embedding)
                            .where(db.user_memories.c.user_id == user_id, db.user_memories.c.embedding.is_not(None))).all()
    scored = sorted(((r.id, r.content, float(np.dot(r.embedding, vector))) for r in rows), key=lambda t: -t[2])
    return scored[:limit]


def recall(engine: sa.Engine, user_id: int, query: str, embed_query=None, k: int = RECALL_K) -> list[str]:
    """Memories relevant to this question (semantic search); the most recent ones if embeddings are unavailable."""
    if not memory_enabled(engine, user_id):
        return []
    if embed_query is not None:
        found = [m for m in _similar(engine, user_id, embed_query(query), k) if m[2] >= RECALL_MIN_SIMILARITY]
    else:
        with engine.connect() as conn:
            rows = conn.execute(sa.select(db.user_memories.c.id, db.user_memories.c.content).where(
                db.user_memories.c.user_id == user_id, db.user_memories.c.kind == "semantic").order_by(
                db.user_memories.c.id.desc()).limit(k)).all()
        found = [(r.id, r.content, 1.0) for r in rows]
    if found:
        with engine.begin() as conn:
            conn.execute(sa.update(db.user_memories).where(db.user_memories.c.id.in_([m[0] for m in found])).values(
                last_used_at=datetime.now()))
    return [m[1] for m in found]


def _store(engine: sa.Engine, user_id: int, kind: str, content: str, conv_id: int | None, embed_texts) -> bool:
    """Insert one memory unless a near-identical one exists (then just refresh it). Returns True if inserted."""
    content = content.strip()[:300]
    if not content:
        return False
    vector = embed_texts([content])[0] if embed_texts is not None else None
    if vector is not None:
        close = _similar(engine, user_id, vector, 1)
        if close and close[0][2] >= DUPLICATE_SIMILARITY:
            with engine.begin() as conn:
                conn.execute(sa.update(db.user_memories).where(db.user_memories.c.id == close[0][0]).values(
                    last_used_at=datetime.now()))
            return False
    with engine.begin() as conn:
        count = conn.execute(sa.select(sa.func.count()).select_from(db.user_memories).where(
            db.user_memories.c.user_id == user_id)).scalar()
        if count >= MAX_MEMORIES_PER_USER:  # bounded store: forget the oldest
            oldest = conn.execute(sa.select(db.user_memories.c.id).where(db.user_memories.c.user_id == user_id)
                                  .order_by(db.user_memories.c.id).limit(1)).scalar()
            conn.execute(sa.delete(db.user_memories).where(db.user_memories.c.id == oldest))
        conn.execute(sa.insert(db.user_memories).values(
            user_id=user_id, kind=kind, content=content, embedding=vector, source_conversation_id=conv_id,
            created_at=datetime.now()))
    return True


def _llm_facts(llm, question: str) -> list[str]:
    prompt = ("From the user's message below, extract durable facts or preferences about the USER that would help in "
              "future conversations (language, role, which contracts are theirs, preferred answer format). "
              "Only what the user states explicitly. Never include document contents. "
              'Reply with JSON only: {"facts": ["short statement", ...]} with at most 2 items, or {"facts": []}.\n'
              f"User message: {question}")
    try:
        raw = llm.generate(prompt)
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
        return [str(f)[:200] for f in data.get("facts", [])[:2] if isinstance(f, str)]
    except Exception:
        return []


def remember_turn(engine: sa.Engine, user_id: int, conv_id: int, question: str, contracts: list[str], llm=None,
                  embed_texts=None) -> int:
    """Write memories for one exchange. Meant to run in the background, after the answer was sent."""
    if not memory_enabled(engine, user_id):
        return 0
    written = 0
    for contract in contracts[:2]:
        written += _store(engine, user_id, "episodic", f"Asked about contract {contract}.", conv_id, embed_texts)
    if len(_FRENCH_WORDS.findall(question)) >= 2:
        written += _store(engine, user_id, "semantic", "The user writes in French.", conv_id, embed_texts)
    if llm is not None and not isinstance(llm, ExtractiveLLM) and _PERSONAL_CUE.search(question):
        for fact in _llm_facts(llm, question):
            written += _store(engine, user_id, "semantic", fact, conv_id, embed_texts)
    return written


# ======================================================================== management
def list_memories(engine: sa.Engine, user_id: int) -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(sa.select(db.user_memories.c.id, db.user_memories.c.kind, db.user_memories.c.content,
                                      db.user_memories.c.created_at).where(db.user_memories.c.user_id == user_id)
                            .order_by(db.user_memories.c.id.desc())).all()
    return [{"id": r.id, "kind": r.kind, "content": r.content, "created_at": r.created_at.isoformat(timespec="seconds")}
            for r in rows]


def delete_memory(engine: sa.Engine, user_id: int, memory_id: int | None = None) -> int:
    """Delete one memory, or all of them when `memory_id` is None. Returns how many were removed."""
    cond = [db.user_memories.c.user_id == user_id]
    if memory_id is not None:
        cond.append(db.user_memories.c.id == memory_id)
    with engine.begin() as conn:
        return conn.execute(sa.delete(db.user_memories).where(*cond)).rowcount


def set_memory_enabled(engine: sa.Engine, user_id: int, enabled: bool) -> None:
    with engine.begin() as conn:
        conn.execute(sa.update(db.users).where(db.users.c.id == user_id).values(memory_enabled=bool(enabled)))
