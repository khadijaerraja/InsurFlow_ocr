"""Document pipeline: register -> extract text (native/OCR) -> extract fields -> pseudonymise -> store -> embed -> validate.

Design choices worth being able to explain:

* Every row belongs to a user (`owner_id`); every function that touches documents takes the owner, so one user
  can never read, validate against or delete another user's data. Duplicates are detected per owner (SHA-256).
* Each document moves through explicit statuses (pending, processing, processed, needs_review, error). Nothing
  is silently dropped: failures keep their error message, suspicious documents keep their list of issues.
* Validation that needs other documents (does this claim's contract exist? is the care date inside the contract
  period?) runs separately and is re-run when a contract arrives later, so import order does not matter.
* Passage embeddings are computed here, right after the text is stored, and written to pgvector. If the embedding
  model is unavailable the document is still usable with keyword search and the vectors are back-filled later.
* Every step is written to the `operations` audit table.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from datetime import datetime

import sqlalchemy as sa

from src.chunking import split_passages
from src.database import schema as db
from src.ingestion.pdf_text import extract_pages
from src.processing.anonymize import anonymize_text, pseudonym
from src.processing.extraction import extract, normalize_labels

ROOT = pathlib.Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "uploads"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
LIST_FIELDS = {"guarantees", "exclusions", "dates"}
SETTING_DEFAULTS = {"anonymize": "true", "search_mode": "hybrid", "top_k": "5", "assistant": "auto",
                    "bm25_min": "2.3", "dense_min": "0.6"}


# ----------------------------------------------------------------- small helpers
def log_operation(engine: sa.Engine, kind: str, detail: str, document_id: int | None = None, ok: bool = True,
                  user_id: int | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(sa.insert(db.operations).values(
            at=datetime.now(), user_id=user_id, kind=kind, detail=detail[:1000], document_id=document_id, ok=ok))


def get_setting(engine: sa.Engine, key: str) -> str:
    with engine.connect() as conn:
        row = conn.execute(sa.select(db.settings.c.value).where(db.settings.c.key == key)).first()
    return row[0] if row else SETTING_DEFAULTS[key]


def set_setting(engine: sa.Engine, key: str, value: str) -> None:
    if key not in SETTING_DEFAULTS:
        raise KeyError(key)
    with engine.begin() as conn:
        conn.execute(sa.delete(db.settings).where(db.settings.c.key == key))
        conn.execute(sa.insert(db.settings).values(key=key, value=str(value)))


def get_fields(conn: sa.Connection, document_id: int) -> dict:
    rows = conn.execute(sa.select(db.extracted_fields.c.name, db.extracted_fields.c.value)
                        .where(db.extracted_fields.c.document_id == document_id)).all()
    out: dict = {}
    for name, value in rows:
        out[name] = [v for v in re.split(r"[|,]", value) if v] if name in LIST_FIELDS else value
    return out


def file_kind(data: bytes) -> str | None:
    """'pdf' or 'image' judged by the file's own signature, not its name; None if unsupported."""
    if data.startswith(b"%PDF"):
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff"):
        return "image"
    return None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", pathlib.Path(name).name)[:120] or "document.pdf"


# --------------------------------------------------------------------- register
def register_document(engine: sa.Engine, data: bytes, original_name: str, upload_dir: pathlib.Path, owner_id: int,
                      run_id: int | None = None) -> tuple[int, bool]:
    """Store the original file under the owner's folder and create a `pending` row. Returns (document_id, is_new)."""
    kind = file_kind(data)
    if kind is None:
        raise ValueError("unsupported file: import a PDF, PNG or JPEG")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("file is larger than 20 MB")
    digest = hashlib.sha256(data).hexdigest()
    with engine.begin() as conn:
        existing = conn.execute(sa.select(db.documents.c.id).where(
            db.documents.c.owner_id == owner_id, db.documents.c.sha256 == digest)).first()
        if existing:
            return existing[0], False
        folder = upload_dir / str(owner_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{digest[:12]}_{_safe_name(original_name)}"
        path.write_bytes(data)
        doc_id = conn.execute(sa.insert(db.documents).values(
            owner_id=owner_id, file_name=_safe_name(original_name), stored_path=str(path), sha256=digest,
            uploaded_at=datetime.now(), doc_type="unknown", status="pending", kind=kind,
            run_id=run_id)).inserted_primary_key[0]
    log_operation(engine, "upload", f"{original_name} ({len(data) // 1024} KB)", doc_id, user_id=owner_id)
    return doc_id, True


# -------------------------------------------------------------------- embeddings
def embed_document_chunks(engine: sa.Engine, doc_id: int, embed_texts) -> int:
    """Compute and store the vectors of a document's passages that have none. Returns how many were written."""
    with engine.connect() as conn:
        todo = conn.execute(sa.select(db.chunks.c.id, db.chunks.c.text).where(
            db.chunks.c.document_id == doc_id, db.chunks.c.embedding.is_(None))).all()
    if not todo:
        return 0
    vectors = embed_texts([text for _, text in todo])
    with engine.begin() as conn:
        for (chunk_id, _), vector in zip(todo, vectors):
            conn.execute(sa.update(db.chunks).where(db.chunks.c.id == chunk_id).values(embedding=vector))
    return len(todo)


def backfill_embeddings(engine: sa.Engine, embed_texts, batch: int = 64) -> int:
    """Embed every passage that still has no vector (after a model outage or a model change)."""
    done = 0
    while True:
        with engine.connect() as conn:
            todo = conn.execute(sa.select(db.chunks.c.id, db.chunks.c.text).where(
                db.chunks.c.embedding.is_(None)).limit(batch)).all()
        if not todo:
            return done
        vectors = embed_texts([text for _, text in todo])
        with engine.begin() as conn:
            for (chunk_id, _), vector in zip(todo, vectors):
                conn.execute(sa.update(db.chunks).where(db.chunks.c.id == chunk_id).values(embedding=vector))
        done += len(todo)


# ---------------------------------------------------------------------- process
def process_document(engine: sa.Engine, doc_id: int, ocr=None, embed_texts=None, fallback=None) -> str:
    """Extract, pseudonymise, store, embed and validate one document. Returns its final status.

    `fallback(text, partial_fields)` may fill fields the rules could not find (see processing/llm_extract.py).
    """
    with engine.begin() as conn:
        row = conn.execute(sa.select(db.documents).where(db.documents.c.id == doc_id)).one()
        conn.execute(sa.update(db.documents).where(db.documents.c.id == doc_id).values(
            status="processing", error_message=None))
    owner = row.owner_id
    try:
        pages = extract_pages(pathlib.Path(row.stored_path), **({"ocr": ocr} if ocr else {}))
        raw = "\n".join(p.text for p in pages)
        if not raw.strip():
            raise ValueError("no text could be extracted (empty or unreadable scan)")
        fields = extract(raw)
        holder_ref = pseudonym(fields.holder) if fields.holder else None
        names = [fields.holder] if fields.holder else []
        redact = get_setting(engine, "anonymize") == "true"
        readable = [normalize_labels(p.text) if p.method == "ocr" else p.text for p in pages]
        stored_texts = [anonymize_text(t, names) if redact else t for t in readable]
        llm_filled: list[str] = []
        if fallback is not None:  # never sees names: it works on the pseudonymised text
            llm_filled = fallback("\n".join(stored_texts), fields)

        with engine.begin() as conn:
            for table in (db.pages, db.chunks, db.extracted_fields, db.issues):
                conn.execute(sa.delete(table).where(table.c.document_id == doc_id))
            for p, text in zip(pages, stored_texts):
                conn.execute(sa.insert(db.pages).values(
                    document_id=doc_id, page_number=p.page_number, method=p.method, text=text))
                # Each passage starts with the contract number so it stays attributable once cut from its page.
                prefix = f"{fields.policy_number}\n" if fields.policy_number else ""
                for passage in split_passages(text):
                    conn.execute(sa.insert(db.chunks).values(
                        document_id=doc_id, owner_id=owner, page_number=p.page_number, text=prefix + passage))
            for name, value in fields.as_fields().items():
                if value not in (None, []):
                    stored = "|".join(value) if name in LIST_FIELDS and name != "dates" else (
                        ",".join(value) if isinstance(value, list) else str(value))
                    conn.execute(sa.insert(db.extracted_fields).values(
                        document_id=doc_id, name=name, value=stored, source="llm" if name in llm_filled else "rule"))
            for message in fields.issues:
                conn.execute(sa.insert(db.issues).values(document_id=doc_id, kind="extraction", message=message))
            if llm_filled:
                conn.execute(sa.insert(db.issues).values(
                    document_id=doc_id, kind="extraction",
                    message=f"{len(llm_filled)} value(s) were read by the language model: please check them"))
            conn.execute(sa.update(db.documents).where(db.documents.c.id == doc_id).values(
                doc_type=fields.doc_type, policy_number=fields.policy_number, holder_ref=holder_ref,
                status="processed"))
            if fields.doc_type == "contract" and fields.policy_number:
                values = {"holder_ref": holder_ref, "start_date": fields.start_date, "end_date": fields.end_date,
                          "annual_ceiling": fields.annual_ceiling}
                same = sa.and_(db.contracts.c.owner_id == owner, db.contracts.c.policy_number == fields.policy_number)
                if conn.execute(sa.select(db.contracts.c.policy_number).where(same)).first():
                    conn.execute(sa.update(db.contracts).where(same).values(**values))
                else:
                    conn.execute(sa.insert(db.contracts).values(
                        owner_id=owner, policy_number=fields.policy_number, **values))
            status = validate_document(conn, doc_id)
            if fields.doc_type == "contract" and fields.policy_number:
                for (other,) in conn.execute(sa.select(db.documents.c.id).where(
                        db.documents.c.owner_id == owner, db.documents.c.policy_number == fields.policy_number,
                        db.documents.c.id != doc_id, db.documents.c.status.in_(("processed", "needs_review")))).all():
                    validate_document(conn, other)  # a late contract can clear "contract not found"
        if embed_texts is not None:
            try:
                embed_document_chunks(engine, doc_id, embed_texts)
            except Exception as exc:  # keyword search still works; vectors are back-filled later
                log_operation(engine, "process", f"{row.file_name}: embeddings postponed ({type(exc).__name__})",
                              doc_id, ok=False, user_id=owner)
        ocr_pages = sum(p.method == "ocr" for p in pages)
        log_operation(engine, "process", f"{row.file_name}: {fields.doc_type}, {status}"
                      f"{f', {ocr_pages} page(s) read with OCR' if ocr_pages else ''}", doc_id, user_id=owner)
        return status
    except Exception as exc:  # keep the failure visible instead of losing the document
        with engine.begin() as conn:
            conn.execute(sa.update(db.documents).where(db.documents.c.id == doc_id).values(
                status="error", error_message=f"{type(exc).__name__}: {exc}"[:500]))
        log_operation(engine, "process", f"{row.file_name}: {exc}", doc_id, ok=False, user_id=owner)
        return "error"


def validate_document(conn: sa.Connection, doc_id: int) -> str:
    """Cross-document checks within one owner's library. Idempotent: replaces earlier validation issues."""
    conn.execute(sa.delete(db.issues).where(db.issues.c.document_id == doc_id, db.issues.c.kind == "validation"))
    doc = conn.execute(sa.select(db.documents).where(db.documents.c.id == doc_id)).one()
    fields = get_fields(conn, doc_id)
    problems: list[str] = []
    if doc.doc_type in ("amendment", "claim_form", "invoice") and doc.policy_number:
        contract = conn.execute(sa.select(db.contracts).where(
            db.contracts.c.owner_id == doc.owner_id, db.contracts.c.policy_number == doc.policy_number)).first()
        if contract is None:
            problems.append(f"contract {doc.policy_number} not found among imported documents")
        else:
            when = fields.get("start_date") if doc.doc_type == "amendment" else fields.get("document_date")
            if when and contract.start_date and contract.end_date and not contract.start_date <= when <= contract.end_date:
                problems.append(f"date {when} is outside the contract period "
                                f"{contract.start_date} to {contract.end_date}")
    for message in problems:
        conn.execute(sa.insert(db.issues).values(document_id=doc_id, kind="validation", message=message))
    has_issues = conn.execute(sa.select(sa.func.count()).select_from(db.issues).where(
        db.issues.c.document_id == doc_id)).scalar() > 0
    status = "needs_review" if has_issues else "processed"
    conn.execute(sa.update(db.documents).where(db.documents.c.id == doc_id).values(status=status))
    return status


# ----------------------------------------------------------------------- delete
def delete_document(engine: sa.Engine, doc_id: int, owner_id: int, upload_dir: pathlib.Path = UPLOAD_DIR) -> bool:
    """Delete one of the owner's documents (False if it does not exist or belongs to someone else)."""
    with engine.begin() as conn:
        doc = conn.execute(sa.select(db.documents).where(
            db.documents.c.id == doc_id, db.documents.c.owner_id == owner_id)).first()
        if doc is None:
            return False
        for table in (db.pages, db.chunks, db.extracted_fields, db.issues):
            conn.execute(sa.delete(table).where(table.c.document_id == doc_id))
        conn.execute(sa.delete(db.documents).where(db.documents.c.id == doc_id))
        if doc.doc_type == "contract" and doc.policy_number:
            conn.execute(sa.delete(db.contracts).where(
                db.contracts.c.owner_id == owner_id, db.contracts.c.policy_number == doc.policy_number))
            for (other,) in conn.execute(sa.select(db.documents.c.id).where(
                    db.documents.c.owner_id == owner_id, db.documents.c.policy_number == doc.policy_number,
                    db.documents.c.status.in_(("processed", "needs_review")))).all():
                validate_document(conn, other)
    path = pathlib.Path(doc.stored_path) if doc.stored_path else None
    if path and path.exists() and upload_dir.resolve() in path.resolve().parents:
        path.unlink()  # only ever delete files inside the upload folder
    log_operation(engine, "delete", doc.file_name, doc_id, user_id=owner_id)
    return True


# ----------------------------------------------------------------- folder ingest
def ingest_folder(engine: sa.Engine, folder: pathlib.Path, owner_id: int, ocr=None, embed_texts=None,
                  upload_dir: pathlib.Path = UPLOAD_DIR) -> dict:
    """Register and process every PDF in a folder for one owner. Safe to re-run: known files are skipped."""
    stats = {"seen": 0, "new": 0, "ocr_pages": 0, "with_issues": 0}
    new_ids: list[int] = []
    with engine.begin() as conn:
        run_id = conn.execute(sa.insert(db.processing_runs).values(started_at=datetime.now())).inserted_primary_key[0]
    for path in sorted(folder.glob("*.pdf")):
        stats["seen"] += 1
        doc_id, is_new = register_document(engine, path.read_bytes(), path.name, upload_dir, owner_id, run_id)
        if not is_new:
            continue
        stats["new"] += 1
        new_ids.append(doc_id)
        process_document(engine, doc_id, ocr, embed_texts)
    with engine.connect() as conn:  # final statuses: a contract imported later can clear earlier warnings
        if new_ids:
            stats["with_issues"] = conn.execute(sa.select(sa.func.count()).select_from(db.documents).where(
                db.documents.c.id.in_(new_ids), db.documents.c.status.in_(("needs_review", "error")))).scalar()
            stats["ocr_pages"] = conn.execute(sa.select(sa.func.count()).select_from(db.pages).where(
                db.pages.c.document_id.in_(new_ids), db.pages.c.method == "ocr")).scalar()
    with engine.begin() as conn:
        conn.execute(sa.update(db.processing_runs).where(db.processing_runs.c.id == run_id).values(
            finished_at=datetime.now(), documents_seen=stats["seen"], documents_new=stats["new"],
            pages_ocr=stats["ocr_pages"], documents_with_issues=stats["with_issues"]))
    return stats


def settings_snapshot(engine: sa.Engine) -> dict:
    return {key: get_setting(engine, key) for key in SETTING_DEFAULTS}


__all__ = ["register_document", "process_document", "validate_document", "delete_document", "ingest_folder",
           "get_setting", "set_setting", "log_operation", "get_fields", "settings_snapshot",
           "embed_document_chunks", "backfill_embeddings"]
