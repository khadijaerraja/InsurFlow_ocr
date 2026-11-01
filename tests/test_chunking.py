"""Chunking behaviour tests. No embedding model or network required."""

from src.chunking import fixed_size_chunks, section_aware_chunks
from src.generate_documents import generate_document


def test_fixed_size_chunks_cover_full_text_with_overlap():
    text = "A" * 1000
    chunks = fixed_size_chunks(text, "doc.md", size=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.doc_id == "doc.md" for c in chunks)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_size_chunks_rejects_overlap_ge_size():
    import pytest

    with pytest.raises(ValueError):
        fixed_size_chunks("hello", "doc.md", size=100, overlap=100)


def test_section_aware_chunks_splits_on_headings():
    text = generate_document("Gold", "Family")
    chunks = section_aware_chunks(text, "gold_family.md")
    sections = {c.section for c in chunks}
    assert "Cover Levels" in sections
    assert "Exclusions" in sections
    assert "Waiting Periods" in sections
    assert "Claims Procedure" in sections
    assert "Geographic Scope" in sections
    for c in chunks:
        assert c.doc_id == "gold_family.md"
        assert c.text.strip()


def test_section_aware_chunk_ids_are_unique_per_document():
    text = generate_document("Bronze", "Student")
    chunks = section_aware_chunks(text, "bronze_student.md")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
