import pathlib

import fitz
import sqlalchemy as sa

from src import auth, synthetic_docs
from src.database import schema as db
from src.pipeline import delete_document, ingest_folder, process_document, register_document

from .conftest import fake_embed_texts


def _user(engine, email="ana@example.com"):
    return auth.create_user(engine, email, "Ana", "correct horse battery")


def _fake_ocr_factory(truths):
    """Stands in for the OCR engine: returns the known text of each scanned document, in order."""
    texts = iter([t for t in truths if t["scanned"]])

    def fake_ocr(_png: bytes) -> str:
        return "\n".join(synthetic_docs._lines(next(texts)))

    return fake_ocr


def _statuses(engine, owner):
    with engine.connect() as conn:
        return dict(conn.execute(sa.select(db.documents.c.status, sa.func.count()).where(
            db.documents.c.owner_id == owner).group_by(db.documents.c.status)).all())


def test_pipeline_ingests_embeds_validates_and_is_idempotent(engine, tmp_path):
    owner = _user(engine)
    truths = synthetic_docs.generate(count=8, seed=3, output_dir=tmp_path)  # contracts first within each holder
    first = ingest_folder(engine, tmp_path / "inbox", owner, ocr=_fake_ocr_factory(truths),
                          embed_texts=fake_embed_texts, upload_dir=tmp_path / "uploads")
    assert first["seen"] == first["new"] == 8
    assert first["ocr_pages"] == sum(t["scanned"] for t in truths)
    assert _statuses(engine, owner) == {"processed": 8}
    with engine.connect() as conn:
        assert conn.execute(sa.select(sa.func.count()).select_from(db.contracts)).scalar() == 3
        assert conn.execute(sa.select(sa.func.count()).select_from(db.chunks)).scalar() >= 8
        assert conn.execute(sa.select(sa.func.count()).select_from(db.chunks).where(
            db.chunks.c.embedding.is_(None))).scalar() == 0  # every passage has its vector
        vector = conn.execute(sa.select(db.chunks.c.embedding).limit(1)).scalar()
        assert vector.shape == (db.EMBEDDING_DIM,)

    again = ingest_folder(engine, tmp_path / "inbox", owner, ocr=lambda _: "", upload_dir=tmp_path / "uploads")
    assert again == {"seen": 8, "new": 0, "ocr_pages": 0, "with_issues": 0}


def test_unknown_contract_is_flagged_and_cleared_when_the_contract_arrives(engine, tmp_path):
    owner = _user(engine)
    synthetic_docs.generate(count=3, seed=3, output_dir=tmp_path)  # contract, amendment, claim of holder 1
    uploads = tmp_path / "uploads"
    files = sorted((tmp_path / "clean").glob("*.pdf"))
    claim = next(f for f in files if f.name.startswith("claim_form"))
    contract = next(f for f in files if f.name.startswith("contract"))

    claim_id, _ = register_document(engine, claim.read_bytes(), claim.name, uploads, owner)
    assert process_document(engine, claim_id) == "needs_review"
    contract_id, _ = register_document(engine, contract.read_bytes(), contract.name, uploads, owner)
    process_document(engine, contract_id)
    assert _statuses(engine, owner) == {"processed": 2}  # the late contract resolved the claim

    assert delete_document(engine, contract_id, owner, uploads)
    assert _statuses(engine, owner) == {"needs_review": 1}  # and deleting it brings the warning back
    assert not list((uploads / str(owner)).glob(f"*{contract.name}"))  # the stored copy is removed too


def test_each_user_has_their_own_library_and_cannot_delete_others(engine, tmp_path):
    ana, bob = _user(engine, "ana@example.com"), _user(engine, "bob@example.com")
    synthetic_docs.generate(count=2, seed=3, output_dir=tmp_path)
    pdf = sorted((tmp_path / "clean").glob("*.pdf"))[0]
    a_id, a_new = register_document(engine, pdf.read_bytes(), pdf.name, tmp_path / "up", ana)
    b_id, b_new = register_document(engine, pdf.read_bytes(), pdf.name, tmp_path / "up", bob)  # same bytes, other user
    assert a_new and b_new and a_id != b_id
    assert register_document(engine, pdf.read_bytes(), "again.pdf", tmp_path / "up", ana) == (a_id, False)
    assert delete_document(engine, a_id, bob, tmp_path / "up") is False  # not Bob's
    assert _statuses(engine, ana) == {"pending": 1}


def test_unreadable_scan_ends_in_error_with_a_message(engine, tmp_path):
    owner = _user(engine)
    synthetic_docs.generate(count=2, seed=1, output_dir=tmp_path)
    stats = ingest_folder(engine, tmp_path / "inbox", owner, ocr=lambda _: "", upload_dir=tmp_path / "up")
    assert stats["new"] == 2 and stats["with_issues"] == 1
    with engine.connect() as conn:
        row = conn.execute(sa.select(db.documents).where(db.documents.c.status == "error")).one()
        assert "no text" in row.error_message


def test_upload_rejects_unsupported_files(engine, tmp_path):
    owner = _user(engine)
    try:
        register_document(engine, b"hello", "x.pdf", tmp_path, owner)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "unsupported" in str(exc)


def test_images_are_read_with_ocr(engine, tmp_path):
    owner = _user(engine)
    truths = synthetic_docs.generate(count=1, seed=3, output_dir=tmp_path)  # a contract
    with fitz.open(tmp_path / "clean" / truths[0]["file"]) as pdf:
        png = pdf[0].get_pixmap(dpi=100).tobytes("png")
    doc_id, is_new = register_document(engine, png, "photo.png", tmp_path / "up", owner)
    assert is_new
    status = process_document(engine, doc_id, ocr=lambda _png: "\n".join(synthetic_docs._lines(truths[0])))
    assert status == "processed"
    with engine.connect() as conn:
        row = conn.execute(sa.select(db.documents.c.kind, db.documents.c.doc_type)).one()
        assert (row.kind, row.doc_type) == ("image", "contract")
        assert conn.execute(sa.select(db.pages.c.method)).scalar() == "ocr"
    assert pathlib.Path(tmp_path / "up" / str(owner)).exists()


def test_model_filled_fields_are_flagged_and_sent_for_review(engine, tmp_path):
    import json

    from src.llm import ScriptedLLM
    from src.processing.llm_extract import make_fallback

    owner = _user(engine)
    page = ("HEALTH CONTRACT\nPolicy reference INS-020\nCover starts on 15 March 2026 and ends on 14 March 2027.\n"
            "Yearly fee 4,200.00 MAD. Maximum reimbursement per year 250 000 MAD.\nCovered: hospitalisation and pharmacy.\n"
            "Not covered: cosmetic surgery.")
    pdf = fitz.open()
    pdf.new_page().insert_text((50, 100), page)
    doc_id, _ = register_document(engine, pdf.tobytes(), "other_insurer.pdf", tmp_path / "up", owner)
    answer = {"doc_type": "contract", "start_date": "2026-03-15", "end_date": "2027-03-14", "annual_ceiling": 250000,
              "amount": 4200.0, "policy_number": "INS-020", "provider": "Made Up Clinic"}
    status = process_document(engine, doc_id, fallback=make_fallback(ScriptedLLM([json.dumps(answer)])))
    assert status == "needs_review"  # a person confirms what the model read
    with engine.connect() as conn:
        sources = dict(conn.execute(sa.select(db.extracted_fields.c.name, db.extracted_fields.c.source)).all())
        issues = [r[0] for r in conn.execute(sa.select(db.issues.c.message))]
    assert sources["end_date"] == "llm" and sources["annual_ceiling"] == "llm" and "provider" not in sources
    assert any("read by the language model" in m for m in issues)


def test_files_with_too_many_pages_are_refused_not_processed(engine, tmp_path, monkeypatch):
    from src.ingestion import pdf_text

    monkeypatch.setattr(pdf_text, "MAX_PAGES", 2)
    owner = _user(engine)
    pdf = fitz.open()
    for _ in range(3):
        pdf.new_page().insert_text((50, 100), "A page with enough text to be read natively.")
    doc_id, _ = register_document(engine, pdf.tobytes(), "long.pdf", tmp_path / "up", owner)
    assert process_document(engine, doc_id) == "error"
    with engine.connect() as conn:
        assert "limit is 2" in conn.execute(sa.select(db.documents.c.error_message)).scalar()
