# Architecture

InsurFlow is a single Flask process (served by waitress) with two background workers, one PostgreSQL database with
the pgvector extension, and an optional free language-model API. There is no separate vector database: dense and
keyword search both run inside PostgreSQL, so a query is one transaction and isolation is one `WHERE owner_id = ...`.

| Part | Technology | Role |
|---|---|---|
| Web | Flask, waitress, plain HTML/CSS/JS | Sign-in, documents, chat, memory drawer. No build step. |
| Ingestion | PyMuPDF, RapidOCR (ONNX), regex extraction | Text from PDFs, OCR for scans and photos, classification, fields, validation. |
| Retrieval | PostgreSQL `tsvector` + IDF, pgvector HNSW, fastembed `all-MiniLM-L6-v2` | Hybrid search, relative score fusion 0.7 keyword / 0.3 dense. |
| RAG | `doc_qa.py`, `chat.py` | Evidence gate in code, prompt with untrusted excerpts, cited answer. |
| Memory | `memory.py` | Short-term (trim + rolling summary) and long-term (per-user semantic store). |
| LLM | Groq `openai/gpt-oss-20b` or Gemini (free tiers) | Optional. Without a key answers quote the documents. |
| Storage | PostgreSQL 15 + pgvector (Docker), `uploads/<user>/` | Everything relational and vectors; original files on disk. |

SQLite is supported as a fallback (tests, quick start without Docker): the same `Searcher` interface is implemented by
an in-memory BM25 + NumPy cosine index. PostgreSQL is the reference backend.

## Design choices

- **Gate before generation.** Whether the documents can answer is decided in code (identifier known? term coverage? dense
  similarity?), not by asking the model. This keeps refusals deterministic and testable.
- **Score fusion over RRF.** Reciprocal-rank fusion ignores score margins; on identifier-heavy questions it let weak
  dense matches outrank exact keyword hits. Min-max normalised scores weighted 0.7 / 0.3 fixed this (evidence recall 0.79 → 1.0).
- **Extraction by rules first.** Layouts are known and bilingual (EN/FR); regex is auditable. A model is only a
  fallback, may only return text that is literally in the page, and sends the document to review.
- **Multi-user by design.** `owner_id` everywhere; contract numbers are unique per user.
- **Local and free.** ONNX models (OCR, embeddings), Docker Postgres, free LLM tiers.

See [security.md](security.md) for the controls and their limits. The schema is in `src/database/schema.py`.

## Repository layout

```
src/
  web/            Flask app, static frontend, waitress entry point
  database/       SQLAlchemy schema, pgvector types, init_schema
  ingestion/      PDF text, page limits
  processing/     classification, regex extraction, grounded LLM fallback, anonymisation
  retrieval/      Postgres searcher, in-memory BM25 + dense, fusion, filters
  pipeline.py     register -> process -> validate -> embed
  doc_qa.py       evidence gate and prompt
  chat.py         conversations
  memory.py       short- and long-term memory
  auth.py         users, throttling, erasure
  evaluation.py   retrieval / refusal metrics on the synthetic set
tests/            131 tests, run on SQLite and PostgreSQL
data/documents/   21 synthetic documents (EN + FR, clean / scanned) with ground truth
docs/             this documentation and the diagram generator
```
