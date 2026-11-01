# Security

Threat model: a small multi-user web application that stores personal-looking insurance documents (synthetic in this
repository) and sends excerpts to a third-party language model. The attacker is another signed-in user, a malicious
uploaded file, or a web page that tries to make the victim's browser act.

| Risk | Control | Where |
|---|---|---|
| Passwords at rest | scrypt hash with per-user salt, minimum 10 characters | `src/auth.py` |
| Brute force | Throttle on login and registration (per IP + email) | `src/auth.py` `Throttle` |
| Session theft | Signed, `HttpOnly`, `SameSite=Lax` cookie, `Secure` in production, 7-day lifetime | `src/web/app.py` |
| CSRF | Per-session token in `X-CSRF-Token` on every unsafe request, plus same-origin `Origin` check | `guard` in `app.py` |
| Cross-user access | `owner_id` in every query; another user's id behaves as "not found" (404), never 403 | `chat.py`, `pipeline.py`, `retrieval/` |
| XSS | CSP with `script-src 'self'` (no inline script); all dynamic text goes through `esc()` before `innerHTML` | `static/app.js`, headers |
| Clickjacking, sniffing | `X-Frame-Options`, `X-Content-Type-Options`, `Referrer-Policy`, HSTS in production | `app.py` |
| Malicious uploads | Magic-byte check (PDF/PNG/JPEG), 20 MB limit, 60-page limit, stored as `<hash>_<sanitised name>` under the owner's folder, never executed | `pipeline.py`, `pdf_text.py` |
| SQL injection | SQLAlchemy Core with bound parameters; full-text terms are sanitised to `[a-z0-9]` before `to_tsquery` | `retrieval/pg.py` |
| Prompt injection | Excerpts are labelled untrusted; the model has no tools and cannot act; answers must cite excerpts; evidence gate runs in code | `doc_qa.py` |
| LLM invents extracted values | Fallback accepts a value only if it appears verbatim in the text; flagged `source=llm`, document goes to review | `processing/llm_extract.py` |
| Personal data in logs/stores | Names pseudonymised in stored text; audit log records events, never document content; generic 500 errors | `processing/anonymize.py`, `app.py` |
| Weak deployment config | `APP_ENV=production` refuses a missing/weak `SECRET_KEY`, closes registration by default | `src/config.py` |
| Secrets in git | `.env` ignored, `.env.example` has no values; tests strip API keys and never use the network | `.gitignore`, `tests/conftest.py` |
| Right to erase | `DELETE /api/auth/me` (password required) removes rows and stored files | `auth.delete_user_data` |
| Database exposure | Postgres bound to `127.0.0.1` only; password required by `docker-compose.yml` | `docker-compose.yml` |

## Known limits (honest list)

- Throttling is in memory, per process; behind several workers use a shared store (Redis) or a reverse proxy limit.
- Prompt-injection defences reduce risk but do not remove it. Documents are sent to Groq/Gemini when a key is set;
  do not import real personal data. Without a key nothing leaves the machine.
- No TLS in the app itself: run behind a reverse proxy that terminates HTTPS (cookie `Secure` is on in production).
- No email verification or password reset; accounts are created by the operator (`python -m src.cli create-user`).
- Dependencies are not pinned by hash; run `pip-audit` in CI for a real deployment.
