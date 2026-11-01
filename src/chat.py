"""Conversations on top of the document RAG. Every function takes the owner: a user only ever sees, answers in,
renames or deletes their own conversations (anything else behaves as "not found").

Memory lives in `src/memory.py` (short-term: thread state + trimming + summary; long-term: per-user store).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime

import sqlalchemy as sa

from src import memory
from src.database import schema as db
from src.doc_qa import answer_document_question, contract_scope
from src.retrieval.filters import Filter

NOT_FOUND = "I could not find this in your documents."
LIST_TOP_K = 10
_ID_RE = re.compile(r"\b(INS|SIN|FAC|AVN)[-\s]?(\d{3,6})\b", re.I)
_GLOBAL_RE = re.compile(r"\b(contracts|documents|policies|claims|invoices|amendments|all|any|every|each|"
                        r"contrats|documents|tous|toutes|chaque)\b", re.I)


def _ids(text: str) -> list[str]:
    seen: list[str] = []
    for prefix, number in _ID_RE.findall(text):
        ident = f"{prefix.upper()}-{int(number):03d}"
        if ident not in seen:
            seen.append(ident)
    return seen


def is_follow_up(question: str) -> bool:
    """A question that names no contract and does not ask about several ("which contracts...") is taken to
    be about the contract already under discussion, e.g. "and its annual ceiling?"."""
    return not _ids(question) and not _GLOBAL_RE.search(question)


def normalise_citations(text: str) -> str:
    """Some models cite as 【1】 or 【1†file】; the interface and the source filter expect [1]."""
    return re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[\d+\]|[*]", "", text)).strip()


# ------------------------------------------------------------------ storage
def create_conversation(engine: sa.Engine, owner_id: int, document_id: int | None = None,
                        title: str = "New conversation") -> int:
    now = datetime.now()
    with engine.begin() as conn:
        return conn.execute(sa.insert(db.conversations).values(
            owner_id=owner_id, title=title, created_at=now, updated_at=now, document_id=document_id, summary="",
            summary_upto=0, entities="[]")).inserted_primary_key[0]


def _mine(owner_id: int, conv_id: int):
    return sa.and_(db.conversations.c.id == conv_id, db.conversations.c.owner_id == owner_id)


def _message_dict(m) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "sources": json.loads(m.sources),
            "refused": bool(m.refused), "created_at": m.created_at.isoformat(timespec="seconds")}


def get_conversation(engine: sa.Engine, owner_id: int, conv_id: int) -> dict | None:
    with engine.connect() as conn:
        conv = conn.execute(sa.select(db.conversations).where(_mine(owner_id, conv_id))).first()
        if conv is None:
            return None
        rows = conn.execute(sa.select(db.messages).where(db.messages.c.conversation_id == conv_id).order_by(
            db.messages.c.id)).all()
        document = None
        if conv.document_id:
            d = conn.execute(sa.select(db.documents.c.id, db.documents.c.file_name, db.documents.c.status,
                                       db.documents.c.doc_type).where(
                db.documents.c.id == conv.document_id, db.documents.c.owner_id == owner_id)).first()
            document = {"id": d.id, "file_name": d.file_name, "status": d.status, "doc_type": d.doc_type} if d else None
    return {"id": conv.id, "title": conv.title, "pinned": bool(conv.pinned),
            "created_at": conv.created_at.isoformat(timespec="seconds"),
            "updated_at": conv.updated_at.isoformat(timespec="seconds"), "document": document,
            "summary": conv.summary, "summary_upto": conv.summary_upto, "entities": json.loads(conv.entities),
            "messages": [_message_dict(m) for m in rows]}


def list_conversations(engine: sa.Engine, owner_id: int, q: str = "") -> list[dict]:
    with engine.connect() as conn:
        convs = conn.execute(sa.select(db.conversations).where(db.conversations.c.owner_id == owner_id).order_by(
            db.conversations.c.pinned.desc(), db.conversations.c.updated_at.desc())).all()
        ids = [c.id for c in convs]
        counts = dict(conn.execute(sa.select(db.messages.c.conversation_id, sa.func.count()).where(
            db.messages.c.conversation_id.in_(ids)).group_by(db.messages.c.conversation_id)).all()) if ids else {}
        last = {}
        if ids:
            for m in conn.execute(sa.select(db.messages).where(
                    db.messages.c.conversation_id.in_(ids), db.messages.c.role == "assistant").order_by(db.messages.c.id)):
                last[m.conversation_id] = m.content
    items = [{"id": c.id, "title": c.title, "pinned": bool(c.pinned), "updated_at": c.updated_at.isoformat(timespec="seconds"),
              "messages": counts.get(c.id, 0), "preview": _clean(last.get(c.id, ""))[:140]}
             for c in convs if counts.get(c.id, 0)]
    q = q.strip().lower()
    return [i for i in items if q in i["title"].lower() or q in i["preview"].lower()] if q else items


def delete_conversation(engine: sa.Engine, owner_id: int, conv_id: int) -> bool:
    with engine.begin() as conn:
        if conn.execute(sa.select(db.conversations.c.id).where(_mine(owner_id, conv_id))).first() is None:
            return False
        conn.execute(sa.delete(db.messages).where(db.messages.c.conversation_id == conv_id))
        conn.execute(sa.delete(db.conversations).where(_mine(owner_id, conv_id)))
        return True


def rename_conversation(engine: sa.Engine, owner_id: int, conv_id: int, title: str) -> str:
    """New title (kept short, never empty). Does not change the conversation's position in the history."""
    title = re.sub(r"\s+", " ", title).strip()[:120]
    if not title:
        raise ValueError("the title cannot be empty")
    with engine.begin() as conn:
        conn.execute(sa.update(db.conversations).where(_mine(owner_id, conv_id)).values(title=title))
    return title


def set_pinned(engine: sa.Engine, owner_id: int, conv_id: int, pinned: bool) -> None:
    with engine.begin() as conn:
        conn.execute(sa.update(db.conversations).where(_mine(owner_id, conv_id)).values(pinned=bool(pinned)))


def set_document(engine: sa.Engine, owner_id: int, conv_id: int, document_id: int | None) -> None:
    with engine.begin() as conn:
        conn.execute(sa.update(db.conversations).where(_mine(owner_id, conv_id)).values(
            document_id=document_id, updated_at=datetime.now()))


# ---------------------------------------------------------------------- ask
def ask(engine: sa.Engine, searcher, owner_id: int, conv_id: int, question: str, llm, dense_min: float,
        top_k: int = 4, embed_query=None, after_turn: Callable[[dict], None] | None = None) -> dict:
    """Answer one question inside a conversation and store both messages. Returns the assistant message.

    `after_turn` receives {"user_id", "conv_id", "question", "contracts"} once the answer is stored; the web app
    uses it to write long-term memories in the background.
    """
    conv = get_conversation(engine, owner_id, conv_id)
    if conv is None:
        raise KeyError(conv_id)
    scope_doc = conv["document"]["id"] if conv["document"] else None
    contract = None
    if scope_doc is None and is_follow_up(question):  # "and its ceiling?" -> same contract as before
        contract = next((e for e in conv["entities"] if e.startswith("INS-")), None)
    named = contract_scope(question)  # an explicit contract in the question beats the remembered one
    flt = Filter(document_id=scope_doc, contract=named.contract if named else contract)

    if not _ids(question) and _GLOBAL_RE.search(question):
        top_k = max(top_k, LIST_TOP_K)  # "which contracts expire in 2027?" needs more than four passages
    result = answer_document_question(
        question, searcher, llm, top_k, dense_min, flt=flt or None,
        memory=memory.build_context(conv["summary"], conv["messages"]),
        profile="\n".join(f"- {m}" for m in memory.recall(engine, owner_id, question, embed_query)))

    sources = [{"n": i, "document_id": sc.chunk.document_id, "file_name": sc.chunk.doc_id, "doc_type": sc.chunk.section,
                "page": sc.chunk.page, "contract": sc.chunk.contract, "text": sc.chunk.text}
               for i, sc in enumerate(result.passages, 1)]
    content = NOT_FOUND if result.refused else normalise_citations(result.answer)
    entities = list(conv["entities"])
    if not result.refused:
        found = _ids(question) + [s["contract"] for s in sources[:1] if s["contract"]]
        entities = [e for e in dict.fromkeys(found + entities)][:8]  # most recent first

    now = datetime.now()
    with engine.begin() as conn:
        conn.execute(sa.insert(db.messages).values(conversation_id=conv_id, role="user", content=question,
                                                   sources="[]", refused=False, created_at=now))
        msg_id = conn.execute(sa.insert(db.messages).values(
            conversation_id=conv_id, role="assistant", content=content, sources=json.dumps(sources),
            refused=result.refused, created_at=datetime.now()).returning(db.messages.c.id)).scalar()
        values = {"updated_at": datetime.now(), "entities": json.dumps(entities)}
        if not conv["messages"]:
            values["title"] = (_clean(question)[:60] or "New conversation")
        conn.execute(sa.update(db.conversations).where(_mine(owner_id, conv_id)).values(**values))
    fresh = get_conversation(engine, owner_id, conv_id)
    memory.maybe_summarise(engine, conv_id, fresh["messages"], fresh["summary"], fresh["summary_upto"], llm)
    if after_turn is not None and not result.refused:
        after_turn({"user_id": owner_id, "conv_id": conv_id, "question": question, "contracts": entities[:2]})
    return {"id": msg_id, "role": "assistant", "content": content, "sources": sources, "refused": result.refused,
            "created_at": datetime.now().isoformat(timespec="seconds")}
