"""Application configuration, read from environment variables (and the local .env file).

Nothing secret is hard-coded or committed: the database password, the session key and the LLM keys come from
the environment. In production (`APP_ENV=production`) the application refuses to start with a weak or missing
session key and keeps self-registration off unless explicitly enabled.
"""

from __future__ import annotations

import os
import pathlib
import secrets
from dataclasses import dataclass
from urllib.parse import quote_plus

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_dotenv(path: pathlib.Path = ROOT / ".env") -> None:
    """Minimal .env reader (KEY=value lines); real environment variables take precedence."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def database_url() -> str:
    """DATABASE_URL if set; otherwise PostgreSQL built from the POSTGRES_* variables; otherwise a local SQLite file."""
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if os.environ.get("POSTGRES_PASSWORD"):
        user = os.environ.get("POSTGRES_USER", "insurflow")
        host = os.environ.get("POSTGRES_HOST", "127.0.0.1")
        port = os.environ.get("POSTGRES_PORT", "5432")
        name = os.environ.get("POSTGRES_DB", "insurflow")
        return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(os.environ['POSTGRES_PASSWORD'])}@{host}:{port}/{name}"
    return f"sqlite:///{ROOT / 'insurflow.db'}"


@dataclass(frozen=True)
class Settings:
    env: str                      # development | production
    secret_key: str
    allow_registration: bool
    cookie_secure: bool
    session_days: int = 7
    max_upload_mb: int = 20

    @property
    def production(self) -> bool:
        return self.env == "production"


def load_settings() -> Settings:
    load_dotenv()
    env = os.environ.get("APP_ENV", "development").lower()
    key = os.environ.get("SECRET_KEY", "")
    if env == "production" and (len(key) < 32 or key.startswith("change")):
        raise RuntimeError("SECRET_KEY must be set to a random value of at least 32 characters in production")
    if not key:  # development convenience: a key that lives only as long as the process
        key = secrets.token_hex(32)
    default_registration = "false" if env == "production" else "true"
    return Settings(
        env=env, secret_key=key,
        allow_registration=os.environ.get("ALLOW_REGISTRATION", default_registration).lower() == "true",
        cookie_secure=os.environ.get("COOKIE_SECURE", "true" if env == "production" else "false").lower() == "true",
    )
