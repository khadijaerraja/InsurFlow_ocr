"""InsurFlow web application: a Flask JSON API plus a static single-page interface.

Security model (summarised in docs/SECURITY.md):
  * Authentication: email + password (scrypt), signed HttpOnly SameSite=Lax session cookie (Secure in production).
  * Authorisation: every query is scoped by the logged-in user's id; other users' objects behave as "not found".
  * CSRF: unsafe requests need the per-session token in `X-CSRF-Token` and, when present, a same-origin `Origin`.
  * Brute force: failed logins are throttled per address+IP; registration is rate-limited and can be switched off.
  * Hardening: CSP and other security headers, upload checks (signature, size), no raw errors sent to the client.
Processing (OCR, embeddings) runs on one background worker so requests stay fast.
"""

from __future__ import annotations

import hmac
import pathlib
import secrets
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from urllib.parse import urlparse

import sqlalchemy as sa
from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory, session

from src import auth, chat, memory, synthetic_docs
from src.config import Settings, database_url, load_settings
from src.database import schema as db
from src.llm import ExtractiveLLM, make_llm
from src.pipeline import (UPLOAD_DIR, delete_document, get_fields, get_setting, log_operation, process_document,
                          register_document)
from src.processing.llm_extract import make_fallback
from src.retrieval.index import SearchIndex

STATIC = pathlib.Path(__file__).resolve().parent / "static"
DOC_TYPES = ("contract", "amendment", "claim_form", "invoice")
META_FIELDS = ("policy_number", "reference", "start_date", "end_date", "document_date", "amount", "annual_ceiling",
               "provider")
BUSY = ("pending", "processing")
UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}
PUBLIC = {"home", "static_files", "auth_login", "auth_register", "auth_me"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "frame-src 'self'; object-src 'self'; connect-src 'self'; frame-ancestors 'self'; base-uri 'none'; "
       "form-action 'self'")


def create_app(db_url: str | None = None, use_dense: bool = True, upload_dir: pathlib.Path = UPLOAD_DIR,
               settings: Settings | None = None, embedder=None) -> Flask:
    settings = settings or load_settings()
    app = Flask(__name__, static_folder=None)
    app.secret_key = settings.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=60 * 1024 * 1024, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.cookie_secure, SESSION_COOKIE_NAME="insurflow_session",
        PERMANENT_SESSION_LIFETIME=timedelta(days=settings.session_days), JSON_SORT_KEYS=False)
    engine = db.make_engine(db_url or database_url())
    index = SearchIndex(engine, use_dense=use_dense, embedder=embedder)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="insurflow-docs")
    memory_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="insurflow-memory")
    login_throttle, register_throttle = auth.Throttle(5, 15 * 60), auth.Throttle(10, 3600)
    app.extensions["insurflow"] = {"engine": engine, "index": index, "workers": (worker, memory_worker)}

    # ------------------------------------------------------------------ guards
    def same_origin() -> bool:
        origin = request.headers.get("Origin")
        return origin is None or urlparse(origin).netloc == request.host

    @app.before_request
    def guard():
        if not request.path.startswith("/api/"):
            return None
        if request.method in UNSAFE and not same_origin():
            return jsonify({"error": "Cross-origin request refused."}), 403
        user = auth.get_user(engine, session["uid"]) if "uid" in session else None
        g.user = user
        if request.endpoint in PUBLIC:
            return None
        if user is None:
            session.clear()
            return jsonify({"error": "Please sign in."}), 401
        if request.method in UNSAFE:
            sent = request.headers.get("X-CSRF-Token", "")
            if not hmac.compare_digest(sent, session.get("csrf", "")):
                return jsonify({"error": "Your session expired. Reload the page."}), 403
        return None

    @app.after_request
    def secure_headers(response):
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        if settings.cookie_secure:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.path.startswith("/api/") and "/file" not in request.path:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(Exception)
    def unexpected(exc):
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description}), exc.code
        app.logger.exception("unhandled error")  # details stay in the server log, not in the response
        return jsonify({"error": "Something went wrong. Please try again."}), 500

    uid = lambda: g.user["id"]  # noqa: E731

    def audit(kind: str, detail: str, ok: bool = True, user_id: int | None = None, document_id: int | None = None):
        log_operation(engine, kind, detail, document_id, ok, user_id if user_id is not None else
                      (g.user["id"] if getattr(g, "user", None) else None))

    # ---------------------------------------------------------------- auth API
    def start_session(user: dict) -> dict:
        session.clear()  # new session id on every login (no fixation)
        session.permanent = True
        session["uid"], session["csrf"] = user["id"], secrets.token_urlsafe(32)
        return {"authenticated": True, "user": user, "csrf": session["csrf"]}

    @app.get("/api/auth/me")
    def auth_me():
        if g.user is None:
            return jsonify({"authenticated": False, "allow_registration": settings.allow_registration})
        session.setdefault("csrf", secrets.token_urlsafe(32))
        return jsonify({"authenticated": True, "user": g.user, "csrf": session["csrf"],
                        "allow_registration": settings.allow_registration})

    @app.post("/api/auth/register")
    def auth_register():
        if not settings.allow_registration:
            return jsonify({"error": "Registration is closed. Ask the administrator for an account."}), 403
        wait = register_throttle.retry_after(request.remote_addr or "?")
        if wait:
            return jsonify({"error": f"Too many sign-ups. Try again in {wait} s."}), 429
        register_throttle.hit(request.remote_addr or "?")
        body = request.get_json(silent=True) or {}
        try:
            user_id = auth.create_user(engine, body.get("email", ""), body.get("name", ""), body.get("password", ""))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        audit("auth", "account created", user_id=user_id)
        return jsonify(start_session(auth.get_user(engine, user_id))), 201

    @app.post("/api/auth/login")
    def auth_login():
        body = request.get_json(silent=True) or {}
        email = auth.normalise_email(body.get("email", ""))
        key = f"{request.remote_addr}|{email}"
        wait = login_throttle.retry_after(key)
        if wait:
            return jsonify({"error": f"Too many attempts. Try again in {wait} s."}), 429
        user = auth.authenticate(engine, email, body.get("password", ""))
        if user is None:
            login_throttle.hit(key)
            audit("auth", "failed sign-in", ok=False)
            return jsonify({"error": "Wrong email or password."}), 401  # same answer for unknown email and bad password
        login_throttle.reset(key)
        audit("auth", "signed in", user_id=user["id"])
        return jsonify(start_session(user))

    @app.post("/api/auth/logout")
    def auth_logout():
        audit("auth", "signed out")
        session.clear()
        return jsonify({"authenticated": False})

    @app.delete("/api/auth/me")
    def auth_delete_account():
        """Delete the account and everything it owns (documents, files, conversations, memories)."""
        body = request.get_json(silent=True) or {}
        if auth.authenticate(engine, g.user["email"], body.get("password", "")) is None:
            return jsonify({"error": "The password is not correct."}), 403
        user_id = uid()
        auth.delete_user_data(engine, user_id)
        shutil.rmtree(upload_dir / str(user_id), ignore_errors=True)
        audit("auth", "account deleted", user_id=user_id)
        session.clear()
        return jsonify({"deleted": True})

    # ------------------------------------------------------------ background work
    def _process_and_reindex(doc_id: int, owner_id: int) -> None:
        process_document(engine, doc_id, embed_texts=index.embed_texts, fallback=make_fallback())
        index.refresh(owner_id)

    def _queue_file(owner_id: int, data: bytes, name: str) -> dict:
        try:
            doc_id, is_new = register_document(engine, data, name, upload_dir, owner_id)
        except ValueError as exc:
            log_operation(engine, "upload", f"{name}: rejected ({exc})", ok=False, user_id=owner_id)
            return {"file_name": name, "error": str(exc)}
        if is_new:
            worker.submit(_process_and_reindex, doc_id, owner_id)
        return {"file_name": name, "id": doc_id, "is_new": is_new}

    def _load_samples(owner_id: int) -> int:
        inbox = synthetic_docs.OUTPUT_DIR / "inbox"
        if not any(inbox.glob("*.pdf")):
            synthetic_docs.generate()
        return sum(bool(_queue_file(owner_id, p.read_bytes(), p.name).get("is_new")) for p in sorted(inbox.glob("*.pdf")))

    def _doc_status(owner_id: int, doc_id: int) -> str | None:
        with engine.connect() as conn:
            return conn.execute(sa.select(db.documents.c.status).where(
                db.documents.c.id == doc_id, db.documents.c.owner_id == owner_id)).scalar()

    # --------------------------------------------------------------------- helpers
    def summaries(conn, owner_id: int, where=None) -> list[dict]:
        q = sa.select(db.documents).where(db.documents.c.owner_id == owner_id).order_by(
            db.documents.c.uploaded_at.desc(), db.documents.c.id.desc())
        if where is not None:
            q = q.where(where)
        rows = conn.execute(q).all()
        ids = [r.id for r in rows]
        fields: dict[int, dict] = {}
        scanned: set[int] = set()
        issue_counts: dict[int, int] = {}
        if ids:
            for doc_id, name, value, source in conn.execute(sa.select(
                    db.extracted_fields.c.document_id, db.extracted_fields.c.name, db.extracted_fields.c.value,
                    db.extracted_fields.c.source).where(db.extracted_fields.c.document_id.in_(ids),
                                                        db.extracted_fields.c.name.in_(META_FIELDS))):
                fields.setdefault(doc_id, {})[name] = value
            issue_counts = dict(conn.execute(sa.select(db.issues.c.document_id, sa.func.count()).where(
                db.issues.c.document_id.in_(ids)).group_by(db.issues.c.document_id)).all())
            scanned = {r[0] for r in conn.execute(sa.select(db.pages.c.document_id).where(
                db.pages.c.document_id.in_(ids), db.pages.c.method == "ocr").distinct())}
        return [{"id": r.id, "file_name": r.file_name, "kind": r.kind or "pdf", "doc_type": r.doc_type,
                 "status": r.status, "owner": g.user["name"], "contract": r.policy_number,
                 "uploaded_at": r.uploaded_at.isoformat(timespec="seconds") if r.uploaded_at else None,
                 "scanned": r.id in scanned or r.kind == "image", "issues": issue_counts.get(r.id, 0),
                 "error": r.error_message, "meta": fields.get(r.id, {})} for r in rows]

    def file_mimetype(path: pathlib.Path) -> str:
        head = path.read_bytes()[:8]
        if head.startswith(b"%PDF"):
            return "application/pdf"
        return "image/png" if head.startswith(b"\x89PNG") else "image/jpeg"

    # ------------------------------------------------------------------ static pages
    @app.get("/")
    def home():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC, name)

    # ------------------------------------------------------------------- documents
    @app.get("/api/documents")
    def list_documents():
        q = (request.args.get("q") or "").strip()
        doc_type, status = request.args.get("type"), request.args.get("status")
        where = []
        if doc_type in DOC_TYPES + ("unknown",):
            where.append(db.documents.c.doc_type == doc_type)
        if status in db.STATUSES:
            where.append(db.documents.c.status == status)
        if q:  # name, contract number, reference or anything written in the document
            in_text = sa.exists().where(db.pages.c.document_id == db.documents.c.id,
                                        db.pages.c.text.icontains(q, autoescape=True))
            in_fields = sa.exists().where(db.extracted_fields.c.document_id == db.documents.c.id,
                                          db.extracted_fields.c.value.icontains(q, autoescape=True))
            where.append(sa.or_(db.documents.c.file_name.icontains(q, autoescape=True), in_fields, in_text))
        with engine.connect() as conn:
            return jsonify(summaries(conn, uid(), sa.and_(*where) if where else None))

    @app.post("/api/documents")
    def upload_documents():
        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "no file received"}), 400
        if len(files) > 20:
            return jsonify({"error": "at most 20 files per upload"}), 400
        results = [_queue_file(uid(), f.read(), f.filename or "document.pdf") for f in files]
        for r in results:
            if "id" in r:
                r["status"] = _doc_status(uid(), r["id"])
        return jsonify(results)

    @app.get("/api/documents/<int:doc_id>")
    def document_detail(doc_id: int):
        with engine.connect() as conn:
            found = summaries(conn, uid(), db.documents.c.id == doc_id)
            if not found:
                abort(404, "Document not found.")
            doc = found[0]
            fields = get_fields(conn, doc_id)
            sources = dict(conn.execute(sa.select(db.extracted_fields.c.name, db.extracted_fields.c.source).where(
                db.extracted_fields.c.document_id == doc_id)).all())
            issue_rows = conn.execute(sa.select(db.issues).where(db.issues.c.document_id == doc_id)).all()
            page_count = conn.execute(sa.select(sa.func.count()).select_from(db.pages).where(
                db.pages.c.document_id == doc_id)).scalar()
            related = [d for d in summaries(conn, uid(), db.documents.c.policy_number == doc["contract"])
                       if d["id"] != doc_id] if doc["contract"] else []
        return jsonify({**doc, "fields": fields, "pages": page_count, "related": related,
                        "ai_fields": sorted(n for n, s in sources.items() if s == "llm"),
                        "alerts": [i.message for i in issue_rows]})

    @app.get("/api/documents/<int:doc_id>/file")
    def document_file(doc_id: int):
        with engine.connect() as conn:
            path = conn.execute(sa.select(db.documents.c.stored_path).where(
                db.documents.c.id == doc_id, db.documents.c.owner_id == uid())).scalar()
        if not path or not pathlib.Path(path).exists():
            abort(404, "Document not found.")
        return send_file(path, mimetype=file_mimetype(pathlib.Path(path)), conditional=True)

    @app.delete("/api/documents/<int:doc_id>")
    def remove_document(doc_id: int):
        if not delete_document(engine, doc_id, uid(), upload_dir):
            abort(404, "Document not found.")
        with engine.begin() as conn:  # conversations scoped to this document go back to the whole library
            conn.execute(sa.update(db.conversations).where(
                db.conversations.c.owner_id == uid(), db.conversations.c.document_id == doc_id).values(document_id=None))
        index.refresh(uid())
        return jsonify({"deleted": doc_id})

    @app.post("/api/documents/samples")
    def load_samples():
        return jsonify({"queued": _load_samples(uid())})

    @app.get("/api/status")
    def status():
        with engine.connect() as conn:
            busy = conn.execute(sa.select(sa.func.count()).select_from(db.documents).where(
                db.documents.c.owner_id == uid(), db.documents.c.status.in_(BUSY))).scalar()
        return jsonify({"busy": busy})

    # ---------------------------------------------------------------- conversations
    @app.get("/api/conversations")
    def conversations():
        return jsonify(chat.list_conversations(engine, uid(), request.args.get("q") or ""))

    @app.post("/api/conversations")
    def new_conversation():
        doc_id = (request.get_json(silent=True) or {}).get("document_id")
        if doc_id is not None and _doc_status(uid(), int(doc_id)) is None:
            abort(404, "Document not found.")
        return jsonify({"id": chat.create_conversation(engine, uid(), doc_id)}), 201

    @app.get("/api/conversations/<int:conv_id>")
    def conversation(conv_id: int):
        conv = chat.get_conversation(engine, uid(), conv_id)
        if conv is None:
            abort(404, "Conversation not found.")
        return jsonify(conv)

    @app.patch("/api/conversations/<int:conv_id>")
    def update_conversation(conv_id: int):
        if chat.get_conversation(engine, uid(), conv_id) is None:
            abort(404, "Conversation not found.")
        body = request.get_json(silent=True) or {}
        if "document_id" in body:  # only the fields that are sent are changed
            doc_id = body["document_id"]
            if doc_id is not None and _doc_status(uid(), int(doc_id)) is None:
                abort(404, "Document not found.")
            chat.set_document(engine, uid(), conv_id, doc_id)
        if "title" in body:
            try:
                chat.rename_conversation(engine, uid(), conv_id, str(body["title"]))
            except ValueError as exc:
                return jsonify({"error": f"Could not rename: {exc}."}), 400
        if "pinned" in body:
            chat.set_pinned(engine, uid(), conv_id, bool(body["pinned"]))
        return jsonify(chat.get_conversation(engine, uid(), conv_id))

    @app.delete("/api/conversations/<int:conv_id>")
    def remove_conversation(conv_id: int):
        if not chat.delete_conversation(engine, uid(), conv_id):
            abort(404, "Conversation not found.")
        return jsonify({"deleted": conv_id})

    @app.post("/api/conversations/<int:conv_id>/messages")
    def send_message(conv_id: int):
        conv = chat.get_conversation(engine, uid(), conv_id)
        if conv is None:
            abort(404, "Conversation not found.")
        question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
        if not question:
            return jsonify({"error": "question is required"}), 400
        if len(question) > 1000:
            return jsonify({"error": "question is too long"}), 400
        if conv["document"] and conv["document"]["status"] in BUSY:
            return jsonify({"error": "This document is still being read. Try again in a moment."}), 409
        searcher = index.get(uid())
        dense_min = float(get_setting(engine, "dense_min"))
        user_id, warning = uid(), None

        def after_turn(turn: dict) -> None:  # long-term memory is written in the background
            memory_worker.submit(_remember, turn, llm_for_memory[0])

        llm_for_memory: list = [None]

        def _remember(turn: dict, llm) -> None:
            try:
                memory.remember_turn(engine, turn["user_id"], turn["conv_id"], turn["question"], turn["contracts"],
                                     llm, index.embed_texts)
            except Exception:
                app.logger.exception("memory write failed")

        try:
            llm, _used = make_llm(get_setting(engine, "assistant"))
            llm_for_memory[0] = llm
            message = chat.ask(engine, searcher, user_id, conv_id, question, llm, dense_min,
                               embed_query=index.embed_query, after_turn=after_turn)
        except KeyError:
            abort(404, "Conversation not found.")
        except Exception as exc:  # missing key, network error, quota: fall back instead of failing the chat
            warning = "The language model was unavailable, so this answer quotes the documents directly."
            log_operation(engine, "ask", f"{type(exc).__name__}: {exc}"[:200], ok=False, user_id=user_id)
            llm_for_memory[0] = None
            message = chat.ask(engine, searcher, user_id, conv_id, question, ExtractiveLLM(), dense_min,
                               embed_query=index.embed_query, after_turn=after_turn)
        audit("ask", f"{'not found' if message['refused'] else 'answered'}")
        updated = chat.get_conversation(engine, user_id, conv_id)
        return jsonify({"message": message, "title": updated["title"], "warning": warning})

    # ----------------------------------------------------------------- memory API
    @app.get("/api/memories")
    def list_memories():
        return jsonify({"enabled": g.user["memory_enabled"], "items": memory.list_memories(engine, uid())})

    @app.patch("/api/memories")
    def memory_settings():
        enabled = (request.get_json(silent=True) or {}).get("enabled")
        if not isinstance(enabled, bool):
            return jsonify({"error": "enabled must be true or false"}), 400
        memory.set_memory_enabled(engine, uid(), enabled)
        audit("memory", f"memory {'on' if enabled else 'off'}")
        return jsonify({"enabled": enabled})

    @app.delete("/api/memories")
    def clear_memories():
        removed = memory.delete_memory(engine, uid())
        audit("memory", f"all memories cleared ({removed})")
        return jsonify({"deleted": removed})

    @app.delete("/api/memories/<int:memory_id>")
    def delete_memory(memory_id: int):
        if not memory.delete_memory(engine, uid(), memory_id):
            abort(404, "Memory not found.")
        return jsonify({"deleted": memory_id})

    return app
