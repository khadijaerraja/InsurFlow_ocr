"""Search behaviour, checked on SQLite (in-memory) and on PostgreSQL + pgvector (SQL) through the same interface."""

import fitz
import numpy as np
import sqlalchemy as sa

from src import auth, synthetic_docs
from src.database import schema as db
from src.doc_qa import check_evidence
from src.pipeline import ingest_folder, process_document, register_document
from src.retrieval.filters import Filter
from src.retrieval.index import SearchIndex

from .conftest import FAKE_EMBEDDER, fake_embed_texts


def _library(engine, tmp_path, email="ana@example.com", count=8):
    owner = auth.create_user(engine, email, "Ana", "correct horse battery")
    synthetic_docs.generate(count=count, seed=3, output_dir=tmp_path)
    ingest_folder(engine, tmp_path / "clean", owner, embed_texts=fake_embed_texts, upload_dir=tmp_path / "up")
    return owner, SearchIndex(engine, embedder=FAKE_EMBEDDER)


def test_vectors_live_in_the_database_with_the_right_shape(engine, tmp_path):
    _library(engine, tmp_path)
    with engine.connect() as conn:
        assert conn.execute(sa.select(db.chunks.c.embedding).limit(1)).scalar().shape == (384,)
        if engine.dialect.name == "postgresql":  # a real pgvector column with an HNSW index
            kind = conn.execute(sa.text("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                                        "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'")).scalar()
            assert kind == "vector(384)"
            indexes = {r[0] for r in conn.execute(sa.text("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'"))}
            assert {"ix_chunks_embedding", "ix_chunks_tsv"} <= indexes


def test_keyword_search_matches_identifiers_and_word_endings(engine, tmp_path):
    owner, index = _library(engine, tmp_path)
    searcher = index.get(owner)
    top = searcher.keyword_search("INS-001 expiry date", 3)
    assert top and top[0].chunk.contract == "INS-001" and "expir" in top[0].chunk.text.lower()
    # "expire" and "expiry" share a stem: the question matches the document even though the word differs
    assert any("xpiry" in h.chunk.text for h in searcher.keyword_search("when does the contract expire", 5))


def test_dense_and_hybrid_search_return_cosine_ranked_passages(engine, tmp_path):
    owner, index = _library(engine, tmp_path)
    searcher = index.get(owner)
    assert searcher.has_dense
    dense = searcher.dense_search("annual ceiling of the contract", 3)
    assert dense and dense[0].score >= dense[-1].score and -1.0 <= dense[0].score <= 1.0
    hybrid = searcher.search("What is the expiry date of contract INS-001?", "hybrid", 3)
    assert hybrid and hybrid[0].chunk.contract == "INS-001"


def test_filters_by_document_contract_and_type(engine, tmp_path):
    owner, index = _library(engine, tmp_path)
    searcher = index.get(owner)
    by_contract = searcher.search("amount", "hybrid", 10, Filter(contract="INS-002"))
    assert by_contract and {h.chunk.contract for h in by_contract} == {"INS-002"}
    claims = searcher.search("amount", "bm25", 10, Filter(doc_type="claim_form"))
    assert claims and {h.chunk.section for h in claims} == {"claim_form"}
    one = searcher.search("amount", "hybrid", 10, Filter(document_id=claims[0].chunk.document_id))
    assert {h.chunk.document_id for h in one} == {claims[0].chunk.document_id}


def test_a_searcher_only_sees_its_owner(engine, tmp_path):
    ana, index = _library(engine, tmp_path / "a", "ana@example.com")
    bob = auth.create_user(engine, "bob@example.com", "Bob", "correct horse battery")
    assert index.get(bob).search("contract", "hybrid", 5) == []
    assert index.get(bob).known_identifiers() == set()
    assert index.get(ana).known_identifiers() >= {"INS-001", "INS-002"}


def test_french_text_is_found_without_accents(engine, tmp_path):
    owner = auth.create_user(engine, "ana@example.com", "Ana", "correct horse battery")
    pdf = fitz.open()
    pdf.new_page().insert_text((72, 100), "Date d'échéance du contrat : 2027-01-04\nPlafond annuel : 500 000 MAD")
    doc_id, _ = register_document(engine, pdf.tobytes(), "contrat_fr.pdf", tmp_path / "up", owner)
    process_document(engine, doc_id, embed_texts=fake_embed_texts)
    searcher = SearchIndex(engine, embedder=FAKE_EMBEDDER).get(owner)
    assert searcher.keyword_search("echeance du contrat", 3), "an unaccented query must match an accented document"
    assert searcher.keyword_search("plafond", 3)


def test_evidence_check_refuses_unknown_identifiers_and_unrelated_questions(engine, tmp_path):
    owner, index = _library(engine, tmp_path)
    searcher = index.get(owner)
    assert check_evidence("What is the expiry date of contract INS-001?", searcher)[0]
    assert not check_evidence("What is the expiry date of contract INS-777?", searcher)[0]  # does not exist
    assert not check_evidence("Is flight delay compensation covered?", searcher)[0]
    assert check_evidence("When does it expire?", searcher, flt=Filter(contract="INS-001"))[0]


def test_hnsw_filtering_does_not_starve_results(engine, tmp_path):
    """With many other users' vectors in the index, a user still gets their k nearest passages (iterative scan)."""
    if engine.dialect.name != "postgresql":
        return
    owner, index = _library(engine, tmp_path / "a", "ana@example.com", count=4)
    other = auth.create_user(engine, "other@example.com", "Other", "correct horse battery")
    rng = np.random.default_rng(0)
    with engine.begin() as conn:  # 300 unrelated passages for another user, all closer to random noise
        doc = conn.execute(sa.insert(db.documents).values(
            owner_id=other, file_name="noise.pdf", sha256="0" * 64, status="processed", doc_type="contract",
            uploaded_at=sa.func.now()).returning(db.documents.c.id)).scalar()
        for i in range(300):
            v = rng.normal(size=384).astype(np.float32)
            conn.execute(sa.insert(db.chunks).values(
                document_id=doc, owner_id=other, page_number=1, text=f"noise {i}", embedding=v / np.linalg.norm(v)))
    assert len(index.get(owner).dense_search("expiry date", 3)) == 3
