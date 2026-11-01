"""Real evaluation of the system against the generated ground truth.

Nothing here is a fixed number: every metric is computed on the documents that
are currently in the database, compared with data/documents/ground_truth.json.

* Extraction: per-field accuracy and the share of documents processed without issues.
* Validation: are the two deliberately flawed documents flagged, and nothing else?
* Retrieval: hit rate and MRR at top-k for BM25, dense and hybrid, plus whether
  the retrieved passages actually contain the expected value ("evidence recall").
* Refusal: unanswerable questions refused vs answerable questions wrongly refused.

Run it from the interface (Evaluation page) or with `python -m src.evaluation`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from datetime import datetime

import sqlalchemy as sa

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.database import schema as db  # noqa: E402
from src.doc_qa import check_evidence, contract_scope  # noqa: E402
from src.metrics import hit_rate, mean_reciprocal_rank  # noqa: E402
from src.pipeline import get_fields, get_setting  # noqa: E402
from src.retrieval.search import MODES  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GROUND_TRUTH = ROOT / "data" / "documents" / "ground_truth.json"
TOP_K = 5

UNANSWERABLE = [
    "What is the status of claim SIN-999?",
    "What is the annual premium of contract INS-777?",
    "Is flight delay compensation covered?",
    "Does any contract cover funeral expenses?",
    "Which contract covers third-party legal liability?",
    "Quel est le statut du sinistre SIN-999 ?",
    "Le retard de vol est-il indemnisé ?",
]


def load_truths(path: pathlib.Path = GROUND_TRUTH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def _squash(text: str) -> str:
    return re.sub(r"\s+", "", text.lower())


def build_questions(truths: list[dict]) -> list[dict]:
    """Questions with the document(s) that really contain the answer, derived from the ground truth."""
    qs = []

    def add(kind, question, expected_files, value=None):
        qs.append({"kind": kind, "question": question, "expected": sorted(expected_files), "value": value})

    contracts = [t for t in truths if t["doc_type"] == "contract" and t.get("locale") != "fr"]
    for t in truths:
        pn, ref, f = t["policy_number"], t["reference"], t["file"]
        if t.get("locale") == "fr":  # French documents are asked about in French (values as printed: day-first, comma)
            comma = lambda v: f"{v:.2f}".replace(".", ",")  # noqa: E731
            y, m, d = (t.get("end_date") or "0-0-0").split("-")
            if t["doc_type"] == "contract":
                add("fact", f"Quelle est la date d'échéance du contrat {pn} ?", [f], f"{d}/{m}/{y}")
                add("fact", f"Quel est le plafond annuel du contrat {pn} ?", [f], comma(t["annual_ceiling"]))
                add("semantic", f"Quelles exclusions s'appliquent au contrat {pn} ?", [f], "chirurgie esthétique")
            elif t["doc_type"] == "claim_form":
                add("fact", f"Quel est le montant déclaré pour le sinistre {ref} ?", [f], comma(t["amount"]))
            else:
                add("fact", f"Quel est le total à payer de la facture {ref} ?", [f], comma(t["amount"]))
        elif t["doc_type"] == "contract":
            add("fact", f"What is the expiry date of contract {pn}?", [f], t["end_date"])
            add("fact", f"What is the annual ceiling of contract {pn}?", [f], f"{t['annual_ceiling']:.2f}")
            add("semantic", f"Which treatments are excluded under contract {pn}?", [f], t["exclusions"][0].lower())
            add("semantic", f"Which guarantees does contract {pn} cover?", [f], t["guarantees"][0].lower())
        elif t["doc_type"] == "amendment":
            add("fact", f"What is the new annual ceiling in amendment {ref}?", [f], f"{t['annual_ceiling']:.2f}")
        elif t["doc_type"] == "claim_form":
            add("fact", f"What amount is claimed in claim {ref}?", [f], f"{t['amount']:.2f}")
        else:
            add("fact", f"What is the total due on invoice {ref}?", [f], f"{t['amount']:.2f}")
    for guarantee in ("Optical care", "Maternity", "Dental care"):
        files = [c["file"] for c in contracts if guarantee in c["guarantees"]]
        if files:
            add("semantic", f"Which contracts cover {guarantee.lower()}?", files, guarantee.lower())
    for exclusion in ("Cosmetic surgery", "War and civil unrest"):
        files = [c["file"] for c in contracts if exclusion in c["exclusions"]]
        if files:
            add("semantic", f"Which contracts exclude {exclusion.lower()}?", files, exclusion.lower())
    return qs


def _rank(results, expected: list[str]) -> int | None:
    for rank, scored in enumerate(results, start=1):
        if scored.chunk.doc_id in expected:
            return rank
    return None


def evaluate_extraction(engine: sa.Engine, truths: list[dict], owner_id: int) -> dict:
    checks = {"doc_type": 0, "policy_number": 0, "reference": 0, "amount": 0, "dates": 0, "ceiling": 0,
              "guarantees": 0, "exclusions": 0}
    totals = dict.fromkeys(checks, 0)
    rows = []
    with engine.connect() as conn:
        for t in truths:
            doc = conn.execute(sa.select(db.documents).where(
                db.documents.c.owner_id == owner_id, db.documents.c.file_name == t["file"])).first()
            if doc is None:
                continue
            f = get_fields(conn, doc.id) if doc.status != "error" else {}

            def score(name, ok):
                totals[name] += 1
                checks[name] += bool(ok)
                return bool(ok)

            ok = [score("doc_type", doc.doc_type == t["doc_type"]),
                  score("policy_number", f.get("policy_number") == t["policy_number"]),
                  score("reference", f.get("reference") == t["reference"]),
                  score("amount", abs(float(f.get("amount", -1)) - t["amount"]) < 0.01)]
            if t["doc_type"] == "contract":
                ok += [score("dates", f.get("start_date") == t["start_date"] and f.get("end_date") == t["end_date"]),
                       score("ceiling", abs(float(f.get("annual_ceiling", -1)) - t["annual_ceiling"]) < 0.01),
                       score("guarantees", set(f.get("guarantees", [])) == set(t["guarantees"])),
                       score("exclusions", set(f.get("exclusions", [])) == set(t["exclusions"]))]
            rows.append({"file": t["file"], "ok": all(ok), "status": doc.status})
    return {"fields": {k: (checks[k] / totals[k] if totals[k] else None) for k in checks},
            "documents_fully_correct": sum(r["ok"] for r in rows), "documents_checked": len(rows), "rows": rows}


def evaluate_validation(engine: sa.Engine, truths: list[dict], owner_id: int) -> dict:
    flagged_expected = detected = false_alarms = 0
    with engine.connect() as conn:
        for t in truths:
            doc = conn.execute(sa.select(db.documents).where(
                db.documents.c.owner_id == owner_id, db.documents.c.file_name == t["file"])).first()
            if doc is None:
                continue
            is_flagged = doc.status == "needs_review"
            if t.get("anomaly"):
                flagged_expected += 1
                detected += is_flagged
            else:
                false_alarms += is_flagged
    return {"expected_anomalies": flagged_expected, "detected": detected, "false_alarms": false_alarms}


def evaluate(engine: sa.Engine, searcher, owner_id: int, truths: list[dict] | None = None, top_k: int = TOP_K) -> dict:
    truths = load_truths() if truths is None else truths
    with engine.connect() as conn:
        present = {r[0] for r in conn.execute(sa.select(db.documents.c.file_name).where(db.documents.c.owner_id == owner_id))}
        counts = dict(conn.execute(sa.select(db.documents.c.status, sa.func.count()).where(
            db.documents.c.owner_id == owner_id).group_by(
            db.documents.c.status)).all())
        errors = [{"file": r.file_name, "message": r.error_message} for r in conn.execute(
            sa.select(db.documents.c.file_name, db.documents.c.error_message).where(
                db.documents.c.owner_id == owner_id, db.documents.c.status == "error"))]
    truths = [t for t in truths if t["file"] in present]
    total = sum(counts.values())
    payload: dict = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "documents": {"total": total, "by_status": counts, "with_ground_truth": len(truths)},
        "processing_success_rate": (counts.get("processed", 0) / total) if total else None,
        "errors": errors,
        "extraction": evaluate_extraction(engine, truths, owner_id),
        "validation": evaluate_validation(engine, truths, owner_id),
    }

    questions = build_questions(truths)
    modes = MODES if searcher.has_dense else ("bm25",)
    per_mode, detail = {}, [{"question": q["question"], "kind": q["kind"], "expected": q["expected"], "ranks": {},
                             "evidence": None} for q in questions]
    for mode in modes:
        by_kind: dict[str, list] = {}
        for q, row in zip(questions, detail):
            results = searcher.search(q["question"], mode, top_k, contract_scope(q["question"]))
            rank = _rank(results, q["expected"])
            row["ranks"][mode] = rank
            if mode == "hybrid" or (mode == "bm25" and not searcher.has_dense):
                row["evidence"] = any(_squash(q["value"]) in _squash(r.chunk.text) for r in results)
            by_kind.setdefault(q["kind"], []).append(rank)
            by_kind.setdefault("all", []).append(rank)
        per_mode[mode] = {kind: {"hit_rate": hit_rate([r is not None for r in ranks]),
                                 "mrr": mean_reciprocal_rank([1 / r if r else 0.0 for r in ranks]),
                                 "questions": len(ranks)} for kind, ranks in by_kind.items()}
    payload["retrieval"] = {"top_k": top_k, "dense_available": searcher.has_dense, "modes": per_mode}
    payload["questions"] = detail
    ev = [r["evidence"] for r in detail if r["evidence"] is not None]
    payload["evidence_recall"] = hit_rate(ev) if ev else None

    dense_min = float(get_setting(engine, "dense_min"))

    def has_evidence(question: str) -> bool:
        return check_evidence(question, searcher, dense_min)[0]

    wrongly_refused = [q["question"] for q in questions if not has_evidence(q["question"])]
    refused = [q for q in UNANSWERABLE if not has_evidence(q)]
    payload["refusal"] = {"unanswerable": len(UNANSWERABLE), "correctly_refused": len(refused),
                          "answerable": len(questions), "wrongly_refused": len(wrongly_refused),
                          "wrongly_refused_questions": wrongly_refused,
                          "dense_min": dense_min}
    return payload


def run_and_store(engine: sa.Engine, searcher, owner_id: int) -> dict:
    payload = evaluate(engine, searcher, owner_id)
    with engine.begin() as conn:
        run_id = conn.execute(sa.insert(db.evaluation_runs).values(
            at=datetime.now(), payload=json.dumps(payload))).inserted_primary_key[0]
    payload["id"] = run_id
    return payload


def latest(engine: sa.Engine) -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(sa.select(db.evaluation_runs).order_by(db.evaluation_runs.c.id.desc()).limit(1)).first()
    return {**json.loads(row.payload), "id": row.id} if row else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction, search and refusals on one user's documents")
    parser.add_argument("--user", required=True, help="email of the account whose documents are evaluated")
    parser.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()
    from src.config import database_url, load_dotenv
    from src.retrieval.index import SearchIndex

    load_dotenv()
    engine = db.make_engine(database_url())
    with engine.connect() as conn:
        owner = conn.execute(sa.select(db.users.c.id).where(db.users.c.email == args.user.strip().lower())).scalar()
    if owner is None:
        raise SystemExit(f"no account for {args.user}")
    index = SearchIndex(engine, use_dense=not args.no_dense)
    result = run_and_store(engine, index.get(owner), owner)
    print(json.dumps({k: v for k, v in result.items() if k not in ("questions",)}, indent=2))
    if index.dense_error:
        print("dense search unavailable:", index.dense_error)


if __name__ == "__main__":
    main()
