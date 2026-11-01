"""Relational model (SQLAlchemy Core) for PostgreSQL + pgvector, with a SQLite fallback for fast offline tests.

Every piece of user data hangs off `users.id`, so isolation is enforced by the data model itself:

  users 1--* documents 1--* pages
              documents 1--* chunks            (passage text + embedding vector + full-text vector)
              documents 1--* extracted_fields / validation_issues
        1--* contracts                         (a contract number is unique per user, not globally)
        1--* conversations 1--* messages       (short-term memory = recent messages + rolling summary)
        1--* user_memories                     (long-term memory: facts about the user, embedded)
  operations: audit trail (who did what); settings: application-wide thresholds

Conversation memory and the document index are different tables on purpose: memory helps understand a
question, it is never evidence for an answer.
"""

from __future__ import annotations

import numpy as np
import sqlalchemy as sa

EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2 / all-MiniLM-L6-v2

metadata = sa.MetaData()

# Document lifecycle: pending -> processing -> processed | needs_review | error
STATUSES = ("pending", "processing", "processed", "needs_review", "error")


class EmbeddingType(sa.types.TypeDecorator):
    """pgvector `vector(384)` on PostgreSQL, raw float32 bytes on SQLite. Python side: always a NumPy array."""

    impl = sa.LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(EMBEDDING_DIM))
        return dialect.type_descriptor(sa.LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        return array if dialect.name == "postgresql" else array.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32) if dialect.name == "postgresql" else np.frombuffer(value, dtype=np.float32)


users = sa.Table(
    "users", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("email", sa.String(254), nullable=False, unique=True),  # stored lower-case
    sa.Column("name", sa.String(80), nullable=False),
    sa.Column("password_hash", sa.String(256), nullable=False),  # scrypt, never the password
    sa.Column("is_active", sa.Boolean, nullable=False, default=True, server_default=sa.true()),
    sa.Column("memory_enabled", sa.Boolean, nullable=False, default=True, server_default=sa.true()),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("last_login_at", sa.DateTime),
)

processing_runs = sa.Table(
    "processing_runs", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("started_at", sa.DateTime, nullable=False),
    sa.Column("finished_at", sa.DateTime),
    sa.Column("documents_seen", sa.Integer, default=0),
    sa.Column("documents_new", sa.Integer, default=0),
    sa.Column("pages_ocr", sa.Integer, default=0),
    sa.Column("documents_with_issues", sa.Integer, default=0),
)

contracts = sa.Table(
    "contracts", metadata,
    sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    sa.Column("policy_number", sa.String(32), primary_key=True),
    sa.Column("holder_ref", sa.String(32)),  # pseudonym, never the name
    sa.Column("start_date", sa.String(10)),
    sa.Column("end_date", sa.String(10)),
    sa.Column("annual_ceiling", sa.Float),
)

documents = sa.Table(
    "documents", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("file_name", sa.String(255), nullable=False),  # name as uploaded
    sa.Column("stored_path", sa.String(512)),
    sa.Column("sha256", sa.String(64), nullable=False),  # duplicate detection, per owner
    sa.Column("uploaded_at", sa.DateTime),
    sa.Column("doc_type", sa.String(32), nullable=False, default="unknown"),
    sa.Column("policy_number", sa.String(32)),  # the contract this document refers to, if recognised
    sa.Column("holder_ref", sa.String(32)),
    sa.Column("status", sa.String(16), nullable=False),
    sa.Column("kind", sa.String(8), default="pdf"),  # pdf | image
    sa.Column("error_message", sa.Text),
    sa.Column("run_id", sa.Integer, sa.ForeignKey("processing_runs.id")),
    sa.UniqueConstraint("owner_id", "sha256", name="uq_documents_owner_sha"),
    sa.Index("ix_documents_owner_status", "owner_id", "status"),
    sa.Index("ix_documents_owner_policy", "owner_id", "policy_number"),
)

pages = sa.Table(
    "pages", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    sa.Column("page_number", sa.Integer, nullable=False),
    sa.Column("method", sa.String(8), nullable=False),  # native | ocr
    sa.Column("text", sa.Text, nullable=False),  # pseudonymised when the setting is on
    sa.UniqueConstraint("document_id", "page_number"),
)

chunks = sa.Table(
    "chunks", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),  # denormalised: filter without a join
    sa.Column("page_number", sa.Integer, nullable=False),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("embedding", EmbeddingType),  # NULL until computed
    sa.Index("ix_chunks_document", "document_id"),
    sa.Index("ix_chunks_owner", "owner_id"),
    # PostgreSQL also gets: `tsv` (generated full-text column) + GIN index, HNSW index on `embedding` (see init_schema).
)

extracted_fields = sa.Table(
    "extracted_fields", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    sa.Column("name", sa.String(32), nullable=False),
    sa.Column("value", sa.Text, nullable=False),
    sa.Column("source", sa.String(8), nullable=False, default="rule", server_default="rule"),  # rule | llm
    sa.Index("ix_fields_document", "document_id"),
)

issues = sa.Table(
    "validation_issues", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("document_id", sa.Integer, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    sa.Column("kind", sa.String(16), nullable=False, default="extraction"),  # extraction | validation
    sa.Column("message", sa.String(255), nullable=False),
    sa.Index("ix_issues_document", "document_id"),
)

operations = sa.Table(
    "operations", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("at", sa.DateTime, nullable=False),
    sa.Column("user_id", sa.Integer),  # no FK: the audit trail must survive deleting the user
    sa.Column("kind", sa.String(24), nullable=False),  # auth | upload | process | delete | ask | memory | evaluate
    sa.Column("detail", sa.Text),
    sa.Column("document_id", sa.Integer),  # no FK: the history must survive deleting the document
    sa.Column("ok", sa.Boolean, nullable=False, default=True),
    sa.Index("ix_operations_user_at", "user_id", "at"),
)

evaluation_runs = sa.Table(
    "evaluation_runs", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("at", sa.DateTime, nullable=False),
    sa.Column("payload", sa.Text, nullable=False),  # JSON
)

settings = sa.Table(
    "settings", metadata,
    sa.Column("key", sa.String(64), primary_key=True),
    sa.Column("value", sa.Text, nullable=False),
)

conversations = sa.Table(
    "conversations", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("title", sa.String(120), nullable=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("updated_at", sa.DateTime, nullable=False),
    sa.Column("document_id", sa.Integer),  # document the conversation is scoped to (NULL = whole library)
    sa.Column("summary", sa.Text, nullable=False, default=""),  # short-term memory: summary of older messages
    sa.Column("summary_upto", sa.Integer, nullable=False, default=0),  # last message id covered by the summary
    sa.Column("entities", sa.Text, nullable=False, default="[]"),  # JSON list of contracts discussed, most recent first
    sa.Column("pinned", sa.Boolean, nullable=False, default=False, server_default=sa.false()),
    sa.Index("ix_conversations_owner_updated", "owner_id", "updated_at"),
)

messages = sa.Table(
    "messages", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("conversation_id", sa.Integer, sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
    sa.Column("role", sa.String(12), nullable=False),  # user | assistant
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("sources", sa.Text, nullable=False, default="[]"),  # JSON list of cited passages
    sa.Column("refused", sa.Boolean, nullable=False, default=False),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Index("ix_messages_conversation", "conversation_id", "id"),
)

# Long-term memory (cross-conversation), namespaced by user: narrow documents that are embedded and searched.
user_memories = sa.Table(
    "user_memories", metadata,
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    sa.Column("kind", sa.String(12), nullable=False),  # semantic (facts, preferences) | episodic (what was asked)
    sa.Column("content", sa.Text, nullable=False),
    sa.Column("embedding", EmbeddingType),
    sa.Column("source_conversation_id", sa.Integer),
    sa.Column("created_at", sa.DateTime, nullable=False),
    sa.Column("last_used_at", sa.DateTime),
    sa.Index("ix_memories_user", "user_id"),
)

# Backwards-compatible alias: older code refers to `issues` as `validation_issues`.
validation_issues = issues


def init_schema(engine: sa.Engine) -> None:
    """Create tables and, on PostgreSQL, the extension, the full-text column and the vector/GIN indexes. Idempotent."""
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
    metadata.create_all(engine)
    if engine.dialect.name != "postgresql":
        return
    from src.retrieval.text import ACCENTED, PLAIN

    with engine.begin() as conn:
        # 'simple' + accent folding, hyphens read as spaces ('INS-004' -> 'ins', '004') (no stemming): English and French share one index, and queries use
        # prefix matching (`expir:*`) to cover word endings. The literals are constants, not user input.
        conn.execute(sa.text(
            "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS "
            f"(to_tsvector('simple', translate(lower(text), '{ACCENTED}-', '{PLAIN} '))) STORED"))
        conn.execute(sa.text("CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING gin (tsv)"))
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops)"))
        conn.execute(sa.text(
            "CREATE INDEX IF NOT EXISTS ix_memories_embedding ON user_memories USING hnsw (embedding vector_cosine_ops)"))


def make_engine(url: str) -> sa.Engine:
    """Engine + schema. PostgreSQL: postgresql+psycopg2://user:password@host:port/db"""
    if url.startswith("sqlite"):
        engine = sa.create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
    else:
        engine = sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    init_schema(engine)
    return engine
