"""Accounts and password handling.

* Passwords are hashed with scrypt (werkzeug), never stored or logged; the minimum length is 10 characters.
* Login does the same amount of work whether or not the email exists, so response time does not reveal accounts.
* Repeated failures for the same address+IP are throttled (sliding window, in memory: per process).
* A deactivated account cannot log in and its existing sessions stop working.
"""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import sqlalchemy as sa
from werkzeug.security import check_password_hash, generate_password_hash

from src.database import schema as db

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}\.[^@\s]{2,}$")
MIN_PASSWORD = 10
_DUMMY_HASH = generate_password_hash("not-a-real-password")


def normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def password_problem(password: str, email: str = "") -> str | None:
    if len(password or "") < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters."
    if len(password) > 200:
        return "That password is too long."
    if password.lower() == email.lower() or password.lower() in {"password123", "1234567890", "qwertyuiop"}:
        return "Choose a less predictable password."
    return None


def create_user(engine: sa.Engine, email: str, name: str, password: str) -> int:
    email, name = normalise_email(email), (name or "").strip()[:80]
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise ValueError("Enter a valid email address.")
    if not name:
        raise ValueError("Enter your name.")
    if problem := password_problem(password, email):
        raise ValueError(problem)
    with engine.begin() as conn:
        if conn.execute(sa.select(db.users.c.id).where(db.users.c.email == email)).first():
            raise ValueError("An account with this email already exists.")
        return conn.execute(sa.insert(db.users).values(
            email=email, name=name, password_hash=generate_password_hash(password), created_at=datetime.now()
        ).returning(db.users.c.id)).scalar()


def authenticate(engine: sa.Engine, email: str, password: str) -> dict | None:
    email = normalise_email(email)
    with engine.connect() as conn:
        row = conn.execute(sa.select(db.users).where(db.users.c.email == email)).first()
    ok = check_password_hash(row.password_hash if row else _DUMMY_HASH, password or "")
    if not (row and ok and row.is_active):
        return None
    with engine.begin() as conn:
        conn.execute(sa.update(db.users).where(db.users.c.id == row.id).values(last_login_at=datetime.now()))
    return public_user(row)


def public_user(row) -> dict:
    return {"id": row.id, "email": row.email, "name": row.name, "memory_enabled": bool(row.memory_enabled)}


def get_user(engine: sa.Engine, user_id: int) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(sa.select(db.users).where(db.users.c.id == user_id, db.users.c.is_active)).first()
    return public_user(row) if row else None


class Throttle:
    """At most `limit` events per `window` seconds for a key (thread-safe, in memory)."""

    def __init__(self, limit: int, window: int) -> None:
        self._limit, self._window = limit, window
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, key: str, now: float) -> deque:
        events = self._events[key]
        while events and now - events[0] > self._window:
            events.popleft()
        return events

    def retry_after(self, key: str) -> int:
        """Seconds to wait if the key is over its limit, else 0."""
        now = time.monotonic()
        with self._lock:
            events = self._trim(key, now)
            return int(self._window - (now - events[0])) + 1 if len(events) >= self._limit else 0

    def hit(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._trim(key, now).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)


def delete_user_data(engine: sa.Engine, user_id: int) -> None:
    """Remove a user and everything they own, explicitly (works on SQLite too, which ignores ON DELETE CASCADE)."""
    owned_docs = sa.select(db.documents.c.id).where(db.documents.c.owner_id == user_id)
    owned_convs = sa.select(db.conversations.c.id).where(db.conversations.c.owner_id == user_id)
    with engine.begin() as conn:
        conn.execute(sa.delete(db.messages).where(db.messages.c.conversation_id.in_(owned_convs)))
        conn.execute(sa.delete(db.conversations).where(db.conversations.c.owner_id == user_id))
        conn.execute(sa.delete(db.user_memories).where(db.user_memories.c.user_id == user_id))
        for table in (db.chunks, db.pages, db.extracted_fields, db.issues):
            key = db.chunks.c.document_id if table is db.chunks else table.c.document_id
            conn.execute(sa.delete(table).where(key.in_(owned_docs)))
        conn.execute(sa.delete(db.documents).where(db.documents.c.owner_id == user_id))
        conn.execute(sa.delete(db.contracts).where(db.contracts.c.owner_id == user_id))
        conn.execute(sa.delete(db.users).where(db.users.c.id == user_id))
