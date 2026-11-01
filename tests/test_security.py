"""Authentication, isolation between users, CSRF, throttling and hardening: the properties a reviewer would probe."""

import io

import pytest

from src.config import Settings, load_settings

from .conftest import TEST_SETTINGS, ApiClient, wait_idle


def _second_user(make_app_client):
    other = make_app_client()
    other.sign_up("bob@example.com", "Bob")
    return other


@pytest.fixture
def two_clients(make_app):
    """Two users on the same application (same database)."""
    app = make_app()
    ana, bob = ApiClient(app.test_client()), ApiClient(app.test_client())
    ana.sign_up("ana@example.com", "Ana")
    bob.sign_up("bob@example.com", "Bob")
    return ana, bob, app


def test_every_api_route_needs_a_session(client):
    for method, path in [("get", "/api/documents"), ("get", "/api/conversations"), ("post", "/api/conversations"),
                         ("get", "/api/memories"), ("get", "/api/status"), ("get", "/api/documents/1/file"),
                         ("post", "/api/documents/samples"), ("delete", "/api/memories")]:
        res = getattr(client.raw, method)(path)
        assert res.status_code == 401, (method, path)
    assert client.raw.get("/api/auth/me").get_json()["authenticated"] is False


def test_passwords_are_hashed_and_policy_is_enforced(client, engine=None):
    r = client.raw.post("/api/auth/register", json={"email": "ana@example.com", "name": "Ana", "password": "short"})
    assert r.status_code == 400 and "10 characters" in r.get_json()["error"]
    assert client.raw.post("/api/auth/register", json={"email": "not-an-email", "name": "A", "password": "correct horse battery"}).status_code == 400
    client.sign_up()
    again = client.raw.post("/api/auth/register", json={"email": "ANA@example.com", "name": "Ana", "password": "correct horse battery"})
    assert again.status_code == 400  # emails are case-insensitive and unique


def test_stored_password_is_a_scrypt_hash(make_app):
    from sqlalchemy import select

    from src.database import schema as db

    app = make_app()
    ApiClient(app.test_client()).sign_up()
    engine = app.extensions["insurflow"]["engine"]
    with engine.connect() as conn:
        stored = conn.execute(select(db.users.c.password_hash)).scalar()
    assert stored.startswith("scrypt:") and "correct horse battery" not in stored


def test_login_logout_and_session_cookie_flags(client):
    client.sign_up()
    cookie = next(c for c in client.raw.get("/api/auth/me").headers.getlist("Set-Cookie") + [""] if "insurflow_session" in c) \
        if client.raw.get("/api/auth/me").headers.getlist("Set-Cookie") else ""
    login = client.raw.post("/api/auth/login", json={"email": "ana@example.com", "password": "correct horse battery"})
    flags = login.headers.get("Set-Cookie", "")
    assert "HttpOnly" in flags and "SameSite=Lax" in flags and "insurflow_session" in flags
    client.csrf = login.get_json()["csrf"]
    assert client.get("/api/documents").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/documents").status_code == 401
    assert cookie is not None


def test_wrong_credentials_get_the_same_answer_and_are_throttled(client):
    client.sign_up("ana@example.com", "Ana")
    unknown = client.raw.post("/api/auth/login", json={"email": "ghost@example.com", "password": "whatever-long-enough"})
    wrong = client.raw.post("/api/auth/login", json={"email": "ana@example.com", "password": "wrong-password-here"})
    assert unknown.status_code == wrong.status_code == 401 and unknown.get_json() == wrong.get_json()
    for _ in range(4):
        client.raw.post("/api/auth/login", json={"email": "ana@example.com", "password": "wrong-password-here"})
    locked = client.raw.post("/api/auth/login", json={"email": "ana@example.com", "password": "correct horse battery"})
    assert locked.status_code == 429 and "Try again" in locked.get_json()["error"]  # even the right password waits


def test_csrf_token_is_required_for_unsafe_requests(client):
    client.sign_up()
    assert client.raw.post("/api/conversations", json={}).status_code == 403  # session cookie but no token
    assert client.raw.post("/api/conversations", json={}, headers={"X-CSRF-Token": "forged"}).status_code == 403
    assert client.post("/api/conversations", json={}).status_code == 201
    cross = client.raw.post("/api/conversations", json={}, headers={"X-CSRF-Token": client.csrf, "Origin": "http://evil.example"})
    assert cross.status_code == 403  # cross-origin request refused even with a valid token


def test_users_cannot_see_or_touch_each_others_data(two_clients, tmp_path):
    from src import synthetic_docs

    ana, bob, _app = two_clients
    synthetic_docs.generate(count=4, seed=3, output_dir=tmp_path / "docs")
    for pdf in sorted((tmp_path / "docs" / "clean").glob("*.pdf")):
        ana.post("/api/documents", data={"files": (io.BytesIO(pdf.read_bytes()), pdf.name)}, content_type="multipart/form-data")
    wait_idle(ana)
    doc_id = ana.get("/api/documents").get_json()[0]["id"]
    conv = ana.post("/api/conversations", json={}).get_json()["id"]
    ana.post(f"/api/conversations/{conv}/messages", json={"question": "What is the expiry date of contract INS-001?"})

    assert bob.get("/api/documents").get_json() == []
    assert bob.get(f"/api/documents/{doc_id}").status_code == 404
    assert bob.get(f"/api/documents/{doc_id}/file").status_code == 404
    assert bob.delete(f"/api/documents/{doc_id}").status_code == 404
    assert bob.get("/api/conversations").get_json() == []
    assert bob.get(f"/api/conversations/{conv}").status_code == 404
    assert bob.patch(f"/api/conversations/{conv}", json={"title": "hijacked"}).status_code == 404
    assert bob.delete(f"/api/conversations/{conv}").status_code == 404
    assert bob.post(f"/api/conversations/{conv}/messages", json={"question": "anything"}).status_code == 404
    assert bob.post("/api/conversations", json={"document_id": doc_id}).status_code == 404  # cannot scope to Ana's file
    own = bob.post("/api/conversations", json={}).get_json()["id"]
    answer = bob.post(f"/api/conversations/{own}/messages", json={"question": "What is the expiry date of contract INS-001?"})
    assert answer.get_json()["message"]["refused"]  # Bob's library is empty: Ana's contract is not evidence for him
    assert ana.get(f"/api/documents/{doc_id}").status_code == 200  # and nothing happened to Ana's data
    assert len(ana.get("/api/conversations").get_json()) == 1


def test_security_headers_and_no_caching_of_api_responses(client):
    client.sign_up()
    res = client.get("/api/documents")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["Cache-Control"] == "no-store"
    csp = res.headers["Content-Security-Policy"]
    assert "default-src 'self'" in csp and "object-src 'self'" in csp and "frame-ancestors 'self'" in csp
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp.split("script-src")[1].split(";")[0]
    assert client.raw.get("/").headers["X-Frame-Options"] == "SAMEORIGIN"


def test_registration_can_be_closed(make_app):
    closed = Settings(env="development", secret_key=TEST_SETTINGS.secret_key, allow_registration=False, cookie_secure=False)
    c = ApiClient(make_app(settings=closed).test_client())
    assert c.raw.post("/api/auth/register", json={"email": "a@example.com", "name": "A", "password": "correct horse battery"}).status_code == 403
    assert c.raw.get("/api/auth/me").get_json()["allow_registration"] is False


def test_production_refuses_a_weak_secret(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "short")
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        load_settings()
    monkeypatch.setenv("SECRET_KEY", "k" * 40)
    settings = load_settings()
    assert settings.production and settings.cookie_secure and not settings.allow_registration


def test_account_deletion_removes_everything_the_user_owns(library, tmp_path):
    client = library(count=3)
    conv = client.post("/api/conversations", json={}).get_json()["id"]
    client.post(f"/api/conversations/{conv}/messages", json={"question": "What is the expiry date of contract INS-001?"})
    wrong = client.delete("/api/auth/me", json={"password": "not my password"})
    assert wrong.status_code == 403
    assert client.delete("/api/auth/me", json={"password": "correct horse battery"}).status_code == 200
    assert client.get("/api/documents").status_code == 401
    assert not list((tmp_path / "uploads").glob("*/*"))  # the stored files are gone as well
    assert client.raw.post("/api/auth/login", json={"email": "ana@example.com", "password": "correct horse battery"}).status_code == 401
