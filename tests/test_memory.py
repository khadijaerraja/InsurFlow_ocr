"""Short-term and long-term memory (see src/memory.py for the LangChain-based design)."""

import sqlalchemy as sa

from src import auth, chat, memory
from src.database import schema as db

from .conftest import fake_embed_query, fake_embed_texts


def _messages(n):
    return [{"id": i + 1, "role": "user" if i % 2 == 0 else "assistant", "content": f"message number {i} " + "x" * 80}
            for i in range(n)]


# ----------------------------------------------------------------------------- short-term
def test_context_keeps_the_latest_messages_within_the_budget(monkeypatch):
    context = memory.build_context("", _messages(20))
    assert "message number 19" in context and "message number 13" not in context  # only the last KEEP_LAST
    assert len(context.splitlines()) <= memory.KEEP_LAST

    monkeypatch.setattr(memory, "CONTEXT_BUDGET_TOKENS", 40)  # a tiny budget trims older messages first
    trimmed = memory.build_context("", _messages(20))
    assert "message number 19" in trimmed and len(trimmed.splitlines()) < memory.KEEP_LAST


def test_context_always_contains_the_summary_first():
    context = memory.build_context("The user is comparing INS-001 and INS-002.", _messages(2))
    assert context.splitlines()[0].startswith("Summary of earlier discussion: The user is comparing")


def test_summary_is_triggered_by_size_and_keeps_the_last_messages(engine, monkeypatch):
    owner = auth.create_user(engine, "ana@example.com", "Ana", "correct horse battery")
    conv = chat.create_conversation(engine, owner)
    base = _messages(12)
    with engine.begin() as conn:
        for m in base:
            conn.execute(sa.insert(db.messages).values(conversation_id=conv, role=m["role"], content=m["content"],
                                                       sources="[]", refused=False, created_at=sa.func.now()))
    data = chat.get_conversation(engine, owner, conv)
    memory.maybe_summarise(engine, conv, data["messages"], "", 0, None)  # below the trigger: nothing happens
    assert chat.get_conversation(engine, owner, conv)["summary"] == ""

    monkeypatch.setattr(memory, "SUMMARY_TRIGGER_TOKENS", 100)
    memory.maybe_summarise(engine, conv, data["messages"], "", 0, None)
    after = chat.get_conversation(engine, owner, conv)
    assert "The user also asked" in after["summary"]
    assert after["summary_upto"] == data["messages"][-memory.KEEP_LAST - 1]["id"]  # last KEEP_LAST stay verbatim


# ------------------------------------------------------------------------------ long-term
def _user(engine, email="ana@example.com"):
    return auth.create_user(engine, email, "Ana", "correct horse battery")


def test_memories_are_embedded_deduplicated_and_recalled_semantically(engine):
    ana = _user(engine)
    assert memory.remember_turn(engine, ana, 1, "What is the ceiling of INS-001?", ["INS-001"], None, fake_embed_texts) == 1
    assert memory.remember_turn(engine, ana, 2, "ceiling INS-001 again", ["INS-001"], None, fake_embed_texts) == 0  # duplicate
    memory.remember_turn(engine, ana, 3, "Quelle est la date du contrat ?", ["INS-002"], None, fake_embed_texts)
    stored = memory.list_memories(engine, ana)
    assert {m["content"] for m in stored} >= {"Asked about contract INS-001.", "The user writes in French."}
    assert {m["kind"] for m in stored} == {"episodic", "semantic"}

    found = memory.recall(engine, ana, "tell me about contract INS-001", fake_embed_query)
    assert found and found[0] == "Asked about contract INS-001."
    assert memory.recall(engine, ana, "completely unrelated zebra", fake_embed_query) == []


def test_memory_is_private_switchable_and_deletable(engine):
    ana, bob = _user(engine), _user(engine, "bob@example.com")
    memory.remember_turn(engine, ana, 1, "q", ["INS-001"], None, fake_embed_texts)
    assert memory.recall(engine, bob, "contract INS-001", fake_embed_query) == []  # namespaced by user
    assert memory.list_memories(engine, bob) == []

    memory.set_memory_enabled(engine, ana, False)
    assert memory.recall(engine, ana, "contract INS-001", fake_embed_query) == []
    assert memory.remember_turn(engine, ana, 2, "q", ["INS-009"], None, fake_embed_texts) == 0  # nothing written when off
    memory.set_memory_enabled(engine, ana, True)

    item = memory.list_memories(engine, ana)[0]
    assert memory.delete_memory(engine, bob, item["id"]) == 0  # not Bob's to delete
    assert memory.delete_memory(engine, ana, item["id"]) == 1
    assert memory.list_memories(engine, ana) == []


def test_memory_store_is_bounded(engine, monkeypatch):
    monkeypatch.setattr(memory, "MAX_MEMORIES_PER_USER", 3)
    ana = _user(engine)
    for n in range(6):
        memory._store(engine, ana, "semantic", f"fact number {n} about topic{n}", None, fake_embed_texts)
    contents = [m["content"] for m in memory.list_memories(engine, ana)]
    assert len(contents) == 3 and "fact number 5 about topic5" in contents and "fact number 0 about topic0" not in contents


# ----------------------------------------------------------------------------- end to end
def _flush_memory(client):
    client.raw.application.extensions["insurflow"]["workers"][1].submit(lambda: None).result(timeout=20)


def test_asking_questions_builds_long_term_memory_in_the_background(library):
    client = library(count=4)
    conv = client.post("/api/conversations", json={}).get_json()["id"]
    client.post(f"/api/conversations/{conv}/messages", json={"question": "What is the expiry date of contract INS-001?"})
    _flush_memory(client)
    listing = client.get("/api/memories").get_json()
    assert listing["enabled"] and any("INS-001" in m["content"] for m in listing["items"])

    other = client.post("/api/conversations", json={}).get_json()["id"]  # a different conversation, same user
    client.post(f"/api/conversations/{other}/messages", json={"question": "Which contract did I ask about?"})
    detail = client.get(f"/api/conversations/{other}").get_json()
    assert all("Asked about contract" not in s["text"] for m in detail["messages"] for s in m["sources"])  # never evidence


def test_memory_api_controls(library):
    client = library(count=4)
    conv = client.post("/api/conversations", json={}).get_json()["id"]
    client.post(f"/api/conversations/{conv}/messages", json={"question": "What is the expiry date of contract INS-001?"})
    _flush_memory(client)
    item = client.get("/api/memories").get_json()["items"][0]
    assert client.delete(f"/api/memories/{item['id']}").status_code == 200
    assert client.delete(f"/api/memories/{item['id']}").status_code == 404
    assert client.patch("/api/memories", json={"enabled": False}).get_json() == {"enabled": False}
    assert client.patch("/api/memories", json={"enabled": "yes"}).status_code == 400
    client.post(f"/api/conversations/{conv}/messages", json={"question": "And its annual ceiling?"})
    _flush_memory(client)
    assert client.get("/api/memories").get_json()["items"] == []  # nothing written while memory is off
    assert client.delete("/api/memories").get_json() == {"deleted": 0}
