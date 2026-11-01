# InsurFlow — interview guide

## The problem in one sentence
Insurance information lives in scanned or PDF documents; InsurFlow turns them into
validated, traceable database rows and makes them searchable with cited answers.

## Pipeline
```
PDF (text or scan) -> page text (native, else OCR) -> classify + extract fields
  -> validate -> database (contract / document / page / field / issue / run)
  -> BM25 + embeddings -> reciprocal rank fusion -> refusal gate -> answer + page citations
```

## Choices and how to defend them
| Choice | Why | Honest limit |
|---|---|---|
| Native text first, OCR only if a page has no text layer | Exact and cheap when possible; OCR can misread | Threshold of 20 characters is a heuristic |
| RapidOCR (pip, CPU) | Free, no system install | Tuned for clean synthetic scans, not handwriting or stamps |
| Regex extraction | Identifiers have a fixed shape, so rules are exact and explainable | Brittle to new layouts; an ML extractor would be the next step with more data |
| SQLAlchemy schema on SQLite | Foreign keys make traceability enforceable; one file, no service to run | PostgreSQL is possible through the same layer but untested |
| SHA-256 per file | Re-running the pipeline never duplicates | Same content with a different name is skipped by design |
| Statuses ready / needs review / failed | Nothing is silently dropped; inconsistent documents are flagged | Resolving a flag is manual |
| BM25 written by hand, crude stemming | Can explain every line; strong on identifiers | English only |
| Reciprocal rank fusion, keyword weighted 3:1 | Rank-based, so BM25 and cosine scales do not clash | Weight chosen on a small question set |
| Conversation memory in tables (recent messages + summary + contracts discussed) | Follow-ups work and conversations can resume; memory never counts as evidence | The summary is a simple one without an LLM key |
| Refusal decided in code before the LLM | Testable: an empty `ScriptedLLM` fails loudly if the gate leaks | Thresholds are tuned on a small set |

## Evaluation (internal: `python -m src.evaluation` and `tests/test_quality.py`)
- **Extraction:** fraction of documents with the correct type, policy number, reference and amount, against generated ground truth.
- **Retrieval:** hit rate and MRR at top-5 for BM25, dense and hybrid, separately on identifier questions and on policy-wording questions.
- **Refusal:** unanswerable questions correctly refused vs answerable ones wrongly refused.

Numbers come from synthetic, generated documents, so they show the method works, not that it
would hold on real insurer data.

## Questions to expect
1. *Why hybrid search?* Embeddings blur identifiers like `CLM-48213` vs `CLM-48231`; BM25 handles them, embeddings handle paraphrase.
2. *How do you know an answer is grounded?* Citations name document and page of the retrieved chunks; the gate refuses without evidence. Faithfulness of the generated prose is **not** measured yet.
3. *What about confidentiality?* Everything runs locally; data is synthetic. Access control and anonymisation are not implemented.
4. *What would you do with more time?* Real document layouts, field-level confidence, review UI, answer-faithfulness evaluation, access control, an API.
