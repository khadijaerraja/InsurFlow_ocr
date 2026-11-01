"""Shared test fixtures.

* Tests never read the local .env for secrets nor call a language model: answers come from the extractive answerer.
* Database-backed tests run on BOTH backends: SQLite (in-memory search) and the real PostgreSQL + pgvector from
  docker-compose (SQL search). The PostgreSQL variant is skipped, not failed, when the container is not running.
* A small deterministic embedder (hashed bag of words, 384 dimensions) replaces the embedding model.
"""

import hashlib
import os
import re

import numpy as np
import pytest

from src.config import Settings, database_url, load_dotenv

load_dotenv()
_PG_URL = database_url() if os.environ.get("POSTGRES_PASSWORD") else None
TEST_DB = "insurflow_test"


def fake_embed_texts(texts):
    """Hashed bag of words: texts sharing words get similar vectors. Deterministic, no download."""
    out = np.zeros((len(texts), 384), dtype=np.float32)
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 384] += 1.0
        norm = np.linalg.norm(out[i])
        out[i] /= norm if norm else 1.0
    return out


def fake_embed_query(text):
    return fake_embed_texts([text])[0]


FAKE_EMBEDDER = (fake_embed_texts, fake_embed_query)
TEST_SETTINGS = Settings(env="development", secret_key="test-secret-key-" + "x" * 32, allow_registration=True,
                         cookie_secure=False)


@pytest.fixture(autouse=True)
def offline_llm(monkeypatch):
    for name in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _fresh_postgres() -> str:
    import psycopg2

    admin = _PG_URL.rsplit("/", 1)[0] + "/postgres"
    conn = psycopg2.connect(admin.replace("postgresql+psycopg2", "postgresql"))
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {TEST_DB}")
    conn.close()
    url = _PG_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"
    conn = psycopg2.connect(url.replace("postgresql+psycopg2", "postgresql"))
    conn.autocommit = True
    conn.cursor().execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    conn.close()
    return url


@pytest.fixture(params=["sqlite", "postgres"])
def db_url(request, tmp_path):
    if request.param == "sqlite":
        return f"sqlite:///{tmp_path / 'test.db'}"
    if _PG_URL is None:
        pytest.skip("PostgreSQL is not configured (docker compose up -d)")
    try:
        return _fresh_postgres()
    except Exception as exc:  # container stopped
        pytest.skip(f"PostgreSQL is not reachable: {type(exc).__name__}")


@pytest.fixture
def sqlite_url(tmp_path):
    return f"sqlite:///{tmp_path / 'test.db'}"


class ApiClient:
    """Flask test client that signs in and sends the CSRF token on unsafe requests, like the browser script does."""

    def __init__(self, flask_client):
        self.raw, self.csrf = flask_client, None

    def _headers(self, kw):
        headers = dict(kw.pop("headers", {}) or {})
        if self.csrf:
            headers["X-CSRF-Token"] = self.csrf
        return headers

    def sign_up(self, email="ana@example.com", name="Ana", password="correct horse battery"):
        r = self.raw.post("/api/auth/register", json={"email": email, "name": name, "password": password})
        assert r.status_code == 201, r.get_json()
        self.csrf = r.get_json()["csrf"]
        return r.get_json()["user"]

    def sign_in(self, email, password="correct horse battery"):
        r = self.raw.post("/api/auth/login", json={"email": email, "password": password})
        if r.status_code == 200:
            self.csrf = r.get_json()["csrf"]
        return r

    def get(self, *a, **kw): return self.raw.get(*a, headers=self._headers(kw), **kw)
    def post(self, *a, **kw): return self.raw.post(*a, headers=self._headers(kw), **kw)
    def patch(self, *a, **kw): return self.raw.patch(*a, headers=self._headers(kw), **kw)
    def delete(self, *a, **kw): return self.raw.delete(*a, headers=self._headers(kw), **kw)


@pytest.fixture
def make_app(db_url, tmp_path):
    from src.web.app import create_app

    def build(settings=TEST_SETTINGS, **kwargs):
        return create_app(db_url, use_dense=True, upload_dir=tmp_path / "uploads", settings=settings,
                          embedder=kwargs.pop("embedder", FAKE_EMBEDDER), **kwargs)
    return build


@pytest.fixture
def client(make_app):
    return ApiClient(make_app().test_client())


def wait_idle(client, timeout=90):
    import time

    end = time.time() + timeout
    while time.time() < end:
        if client.get("/api/status").get_json()["busy"] == 0:
            return True
        time.sleep(0.2)
    raise AssertionError("documents are still being processed")


@pytest.fixture
def library(client, tmp_path):
    """Signs up a user and imports generated documents (native text: no OCR model needed)."""
    import io

    from src import synthetic_docs

    def build(count=8, seed=3):
        synthetic_docs.generate(count=count, seed=seed, output_dir=tmp_path / "docs")
        client.sign_up()
        for pdf in sorted((tmp_path / "docs" / "clean").glob("*.pdf")):
            r = client.post("/api/documents", data={"files": (io.BytesIO(pdf.read_bytes()), pdf.name)},
                            content_type="multipart/form-data")
            assert r.status_code == 200 and r.get_json()[0]["is_new"]
        wait_idle(client)
        return client
    return build


@pytest.fixture
def engine(db_url):
    from src.database import schema as db

    eng = db.make_engine(db_url)
    yield eng
    eng.dispose()
