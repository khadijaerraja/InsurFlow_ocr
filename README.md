# InsurFlow

![Chat](docs/screenshots/03-chat.png)

| Documents | Sign in |
|---|---|
| ![Documents](docs/screenshots/02-documents.png) | ![Sign in](docs/screenshots/01-sign-in.png) |

## Purpose

InsurFlow is a small assistant for insurance documents. You import PDFs, scans or photos (contracts, amendments, claims,
invoices); it reads them with OCR, extracts the key values (policy number, dates, amounts, guarantees), checks them against
each other, and lets you ask questions in English or French. Answers come **only from your documents**, with the source
passage and page; when the documents do not contain the answer, it says so.

It is a portfolio prototype built on synthetic documents, with free and local tools.

## Technologies

| Layer | Tools |
|---|---|
| Language / web | Python 3.11+, Flask, waitress, plain HTML/CSS/JS |
| Database | PostgreSQL 15 + pgvector (Docker), SQLAlchemy, SQLite fallback |
| OCR / PDF | PyMuPDF (digital PDFs), RapidOCR (scans, images) |
| Extraction | Rule-based (regex), English and French layouts; optional grounded LLM fallback |
| Search | PostgreSQL full-text + pgvector HNSW, fastembed `all-MiniLM-L6-v2` embeddings, score fusion |
| LLM (optional, free tiers) | Groq `openai/gpt-oss-20b` or Gemini; without a key, answers quote the documents |
| Memory | Short-term (trimming + rolling summary) and long-term (per-user semantic store) |
| Tests | pytest (131 tests, SQLite and PostgreSQL), ruff |

## Run the project

Requirements: Python 3.11+, Git and Docker.

```bash
git clone <repository-url>
cd insurance-document-rag
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`: set `POSTGRES_PASSWORD` and `SECRET_KEY` (generate one with
`python -c "import secrets; print(secrets.token_hex(32))"`). If port 5432 is taken, change `POSTGRES_PORT`.
Optionally add `GROQ_API_KEY` or `GEMINI_API_KEY`.

```bash
docker compose up -d
python -m src.cli init-db
python -m src.cli create-user you@example.com --name "Your Name"
python -m src.web
```

Open http://127.0.0.1:8000, sign in, go to **Documents** and click **Load sample documents** (21 synthetic files,
English and French, 7 scanned; about 40 seconds). Then try:

- *What is the expiry date of contract INS-001?* followed by *And its annual ceiling?*
- *Quelle est la date d'échéance du contrat INS-006 ?*
- *What is the capital of France?* (answer: not found in your documents)

Without Docker, leave `POSTGRES_PASSWORD` empty: the app falls back to a local SQLite file (same features, no pgvector).

Other commands:

```bash
python -m src.cli load-samples you@example.com          # load the sample documents from the terminal
python -m src.cli ask you@example.com "What is covered under dental care?"
python -m src.evaluation --user you@example.com         # retrieval and refusal metrics
python -m pytest -q                                      # run the tests
```

## How it works

1. **Import**: file checks, duplicate detection, text or OCR per page, classification, extraction, validation, embedding.
2. **Ask**: hybrid search (keywords + vectors) limited to your documents; a check in code decides whether the evidence is
   sufficient before the language model is called; the answer cites its sources.
3. **Remember**: the conversation keeps its recent turns and a summary; a per-user store remembers lasting facts. You can
   list, delete or disable it from the account menu. Memory is context, never evidence.

Details: [architecture](docs/architecture.md) and [security](docs/security.md).

Measured on the synthetic set (PostgreSQL, real embedding model): hybrid retrieval hit rate 100 %, MRR 0.98, evidence
recall 1.0, 7 of 7 unanswerable questions refused. The set is small; these numbers show the system works, not how it
would perform on real documents.

## Current limits

- Synthetic documents only; extraction rules are written for these layouts.
- The default embedding model is English: a question in one language about a document in the other works when it names a
  contract number, but may be refused otherwise (set `EMBEDDING_MODEL` to a multilingual 384-dimension model).
- No migration tool yet, one ingestion worker, login throttling kept in memory per process.
- Prompt-injection protection is partial. With an API key, excerpts are sent to the provider: do not import real personal data.
- When a contract and its amendment disagree (for example the annual ceiling), the answer does not yet say which one is in force.

## Perspectives

- **Roles and permissions (RBAC)**: admin / reviewer / reader, shared libraries per team or organisation.
- **Stronger security**: password reset and e-mail verification, two-factor sign-in, shared rate limiting (Redis), audit
  export, encryption of stored files, dependency scanning in CI.
- **Multi-tenant organisations** with per-tenant isolation and quotas.
- **Multilingual retrieval** (multilingual embeddings, cross-language evaluation) and Arabic support.
- **Amendment-aware answers**: resolve which clause is currently in force.
- **Review workflow**: a screen to correct flagged fields and re-run validation.
- **More document layouts and real-data evaluation**; layout-aware extraction for tables.
- **Deployment**: Alembic migrations, job queue for ingestion, Docker image for the app, CI pipeline, monitoring.

License: see [LICENSE](LICENSE).
