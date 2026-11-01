"""Internal quality gate of the whole system (not shown in the interface).

It runs on the synthetic documents (native-text versions, so no OCR model is needed) with a small deterministic
embedder, on both backends, and fails if extraction, validation, retrieval or the refusal rule regress.
`python -m src.evaluation --user EMAIL` runs the same measurements on real stored documents, including scans and
the real embedding model.
"""

import pytest

from src import auth, evaluation, synthetic_docs
from src.pipeline import ingest_folder
from src.retrieval.index import SearchIndex

from .conftest import FAKE_EMBEDDER, fake_embed_texts


@pytest.fixture
def report(engine, tmp_path):
    truths = synthetic_docs.generate(output_dir=tmp_path / "docs")
    owner = auth.create_user(engine, "ana@example.com", "Ana", "correct horse battery")
    ingest_folder(engine, tmp_path / "docs" / "clean", owner, embed_texts=fake_embed_texts, upload_dir=tmp_path / "up")
    searcher = SearchIndex(engine, embedder=FAKE_EMBEDDER).get(owner)
    return evaluation.evaluate(engine, searcher, owner, truths), truths


def test_extraction_is_exact_on_every_document(report):
    result, truths = report
    fields = result["extraction"]["fields"]
    assert all(score == 1.0 for score in fields.values() if score is not None), fields
    assert result["extraction"]["documents_fully_correct"] == len(truths)


def test_validation_flags_exactly_the_planted_problems(report):
    result, _ = report
    expected = result["validation"]["expected_anomalies"]
    assert expected >= 3 and result["validation"] == {"expected_anomalies": expected, "detected": expected, "false_alarms": 0}


def test_retrieval_finds_the_right_document(report):
    result, _ = report
    modes = result["retrieval"]["modes"]
    assert modes["bm25"]["fact"]["hit_rate"] == 1.0
    assert modes["hybrid"]["all"]["hit_rate"] >= 0.95
    assert modes["hybrid"]["all"]["mrr"] >= 0.7
    assert result["evidence_recall"] >= 0.95


def test_assistant_refuses_unanswerable_and_answers_the_rest(report):
    result, _ = report
    refusal = result["refusal"]
    assert refusal["correctly_refused"] == refusal["unanswerable"]
    assert refusal["wrongly_refused"] == 0, refusal["wrongly_refused_questions"]
