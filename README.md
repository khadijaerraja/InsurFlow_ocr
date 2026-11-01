# InsurFlow — Intelligent Insurance Document Assistant

<p align="center">
  <img src="docs/screenshots/03-chat.png" alt="InsurFlow Chat Interface" width="100%" />
</p>

<p align="center">
  **A local-first assistant for insurance documents, combining OCR, structured data extraction, hybrid search and grounded question answering.**
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?logo=flask" alt="Flask" />
  <img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/Tests-132%20passing-brightgreen" alt="Tests" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
</p>

---

## Overview

InsurFlow is a portfolio prototype designed to simplify the processing and exploration of insurance documents.

Users can upload PDF files, scanned documents or images containing contracts, amendments, claims and invoices. The application extracts relevant information, validates key fields and allows users to ask questions in English or French.

Answers are grounded exclusively in the user's documents and include supporting passages and page references. When the available documents do not contain sufficient evidence, InsurFlow refuses to provide an unsupported answer.

**Key capabilities**
- PDF and image ingestion with OCR.
- Extraction of policy numbers, dates, amounts and guarantees.
- Validation and cross-document consistency checks.
- Hybrid retrieval combining keyword and vector search.
- Source-grounded question answering.
- Short-term conversation context and per-user long-term memory.

> **Project scope:** This is a portfolio prototype evaluated on synthetic insurance documents. It is not intended for processing real customer data or production insurance workflows.

## Application Preview

| Sign in | Document Management |
|:---:|:---:|
| ![Sign in](docs/screenshots/01-sign-in.png) | ![Documents](docs/screenshots/02-documents.png) |

**Question Answering**

![Chat interface](docs/screenshots/03-chat.png)

## Technology Stack

| Layer | Technologies |
|---|---|
| Language & Web | Python 3.11+, Flask, Waitress, HTML, CSS, JavaScript |
| Database | PostgreSQL 15, pgvector, SQLAlchemy, SQLite fallback |
| OCR & PDF | PyMuPDF, RapidOCR |
| Information Extraction | Regex-based extraction, bilingual document layouts, optional LLM fallback |
| Search & Retrieval | PostgreSQL Full-Text Search, pgvector HNSW, FastEmbed, all-MiniLM-L6-v2 |
| LLM Integration | Groq (openai/gpt-oss-20b), Gemini — optional external APIs |
| Memory | Conversation trimming, rolling summaries, per-user semantic memory |
| Testing & Quality | pytest, Ruff |

## Architecture & Workflow

InsurFlow follows three main processing stages:

### 1. Document Ingestion
- File validation and duplicate detection.
- Digital text extraction or page-level OCR.
- Document classification.
- Key information extraction and validation.
- Text chunking and embedding generation.

### 2. Retrieval & Question Answering
- Hybrid retrieval using keyword and vector search.
- Retrieval restricted to the authenticated user's documents.
- Programmatic evidence sufficiency checks.
- Grounded answer generation with source passages and page references.
- Refusal when supporting evidence is insufficient.

### 3. Conversation Memory
- Recent conversation turns and rolling summaries.
- Per-user semantic memory for persistent context.
- Memory management through the account interface.

**Important:** Conversation memory provides context only. It is never treated as evidence for document-based answers.

For implementation details, see:
- [System Architecture](docs/architecture.md)
- [Security Design](docs/security.md)

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Docker and Docker Compose (recommended for PostgreSQL)

### Installation

```bash
git clone https://github.com/khadijaerraja/InsurFlow_ocr.git
cd InsurFlow_ocr

pip install -r requirements.txt
```

Create your environment file:

```bash
cp .env.example .env
```

Configure `.env`:

- Set `POSTGRES_PASSWORD` for PostgreSQL.
- Set `SECRET_KEY` using:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

- Change `POSTGRES_PORT` if port `5432` is already in use.
- Optionally configure `GROQ_API_KEY` or `GEMINI_API_KEY` for external LLM integration.

### Start the Application

Start PostgreSQL:

```bash
docker compose up -d
```

Initialize the database and create a user:

```bash
python -m src.cli init-db

python -m src.cli create-user you@example.com --name "Your Name"
```

Start the web application:

```bash
python -m src.web
```

Open **http://127.0.0.1:8000**, sign in, navigate to **Documents**, and select **Load sample documents**.

The sample dataset contains 21 synthetic documents in English and French, including 7 scanned documents. Initial processing takes approximately 40 seconds.

### Try It Out

Ask the following questions:

**English**
- What is the expiry date of contract INS-001?
- And its annual ceiling?
- What is the capital of France?

**French**
- Quelle est la date d'échéance du contrat INS-006 ?

The last question about France should be refused because the answer is not contained in the uploaded documents.

### SQLite Fallback

Docker is optional. When PostgreSQL is not configured, the application can use a local SQLite database.

The SQLite fallback does not provide PostgreSQL-specific pgvector functionality.

### Additional CLI Commands

```bash
# Load sample documents
python -m src.cli load-samples you@example.com

# Ask a question from the terminal
python -m src.cli ask you@example.com "What is covered under dental care?"

# Run retrieval and refusal evaluation
python -m src.evaluation --user you@example.com

# Run the test suite
python -m pytest -q
```

## Evaluation

The retrieval and refusal evaluation was performed on the synthetic dataset using PostgreSQL and the embedding model.

| Metric | Result |
|---|---:|
| Hybrid retrieval hit rate | 100% |
| Mean Reciprocal Rank (MRR) | 0.98 |
| Evidence recall | 1.00 |
| Unanswerable questions correctly refused | 7 / 7 |

The evaluation dataset is small and synthetic. These results demonstrate the behavior of the prototype under controlled conditions; they should not be interpreted as production performance or generalization to real insurance documents.

**Test suite:** 132 tests covering SQLite and PostgreSQL configurations.

### Key evaluation metrics

| Area | Metric | Result |
|---|---|---:|
| RAG quality (RAGAS, 16 answers, judge gpt-oss-120b) | Faithfulness / Context recall | 1.00 / 1.00 |
| OCR on clean scans, PNG and JPEG | Character error rate / field accuracy | 0.4-1.5 % / 99-100 % |
| OCR on tilted pages (3° rotation, phone-photo simulation) | Character error rate / field accuracy | 26 % / ~59 % |
| Agent safety | Correct refusals / false refusals / prompt injection resisted | 12/12 / 0/43 / 2/2 |

The tilted-page result is the main known weakness: the line-grouping step in `src/ingestion/ocr.py` breaks on rotated pages, and a deskew step before OCR brought the error rate back to 0.5 % in a test. Full protocol, per-format tables, real-document tests and caveats (small synthetic set): [docs/evaluation-results.md](docs/evaluation-results.md).

## Current Limitations

- **Synthetic data:** Extraction rules are designed around the included document layouts.
- **Language mismatch:** The default embedding model is English-oriented. Cross-language retrieval may be limited unless a contract number or another strong keyword is provided.
- **Database migrations:** No migration framework is currently integrated.
- **Ingestion:** Document ingestion uses a single worker.
- **Authentication:** Login throttling is stored in process memory.
- **Prompt injection:** Protection is partial. When an external LLM API is enabled, retrieved excerpts are sent to that provider. Do not upload real personal or confidential information.
- **Amendment conflicts:** The application does not yet resolve which contractual clause takes precedence when an amendment conflicts with the original contract.

## Future Improvements

- **Multilingual retrieval:** Introduce multilingual embeddings, cross-language evaluation and Arabic support.
- **Amendment-aware reasoning:** Identify the currently applicable contractual clauses.
- **Document review:** Add a workflow for correcting extracted fields and rerunning validation.
- **Advanced extraction:** Support additional document layouts and table-aware extraction.
- **Security:** Password reset, email verification, two-factor authentication, shared rate limiting and audit exports.
- **Access control:** Implement role-based permissions and shared team libraries.
- **Multi-tenancy:** Add organisation-level isolation and resource quotas.
- **Deployment & Operations:** Integrate Alembic migrations, background ingestion jobs, CI/CD, monitoring and a dedicated application Docker image.

## License

This project is distributed under the [MIT License](LICENSE).