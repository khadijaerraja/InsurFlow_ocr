"""Administration commands (no web server needed).

  python -m src.cli init-db                         create/upgrade tables, extension and indexes (idempotent)
  python -m src.cli create-user EMAIL --name NAME   create an account (the password is asked, never passed as argument)
  python -m src.cli load-samples EMAIL              generate the synthetic demo documents and import them for a user
  python -m src.cli backfill-embeddings             compute vectors for passages that have none (after a model change)
  python -m src.cli ask EMAIL "question"            ask a question from the terminal (no memory)
"""

from __future__ import annotations

import argparse
import getpass
import sys

import sqlalchemy as sa

from src import auth, synthetic_docs
from src.config import database_url, load_dotenv
from src.database import schema as db
from src.doc_qa import answer_document_question, contract_scope
from src.llm import make_llm
from src.pipeline import UPLOAD_DIR, backfill_embeddings, get_setting, ingest_folder
from src.retrieval.index import SearchIndex


def _user_id(engine: sa.Engine, email: str) -> int:
    with engine.connect() as conn:
        uid = conn.execute(sa.select(db.users.c.id).where(db.users.c.email == auth.normalise_email(email))).scalar()
    if uid is None:
        sys.exit(f"No account for {email}. Create it with: python -m src.cli create-user {email}")
    return uid


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db")
    cu = sub.add_parser("create-user"); cu.add_argument("email"); cu.add_argument("--name", required=True)
    ls = sub.add_parser("load-samples"); ls.add_argument("email")
    sub.add_parser("backfill-embeddings")
    ask = sub.add_parser("ask"); ask.add_argument("email"); ask.add_argument("question")
    args = parser.parse_args()
    load_dotenv()
    engine = db.make_engine(database_url())

    if args.command == "init-db":
        print("Database ready:", engine.url.render_as_string(hide_password=True))
    elif args.command == "create-user":
        password = getpass.getpass("Password (min 10 characters): ")
        if getpass.getpass("Repeat password: ") != password:
            sys.exit("The passwords differ.")
        try:
            print("Created user", auth.create_user(engine, args.email, args.name, password))
        except ValueError as exc:
            sys.exit(str(exc))
    elif args.command == "load-samples":
        owner = _user_id(engine, args.email)
        if not any((synthetic_docs.OUTPUT_DIR / "inbox").glob("*.pdf")):
            synthetic_docs.generate()
        index = SearchIndex(engine)
        print(ingest_folder(engine, synthetic_docs.OUTPUT_DIR / "inbox", owner, embed_texts=index.embed_texts,
                            upload_dir=UPLOAD_DIR))
    elif args.command == "backfill-embeddings":
        index = SearchIndex(engine)
        if index.embed_texts is None:
            sys.exit(f"Embedding model unavailable: {index.dense_error}")
        print("Embedded", backfill_embeddings(engine, index.embed_texts), "passages")
    elif args.command == "ask":
        owner = _user_id(engine, args.email)
        index = SearchIndex(engine)
        llm, used = make_llm("auto")
        result = answer_document_question(args.question, index.get(owner), llm,
                                          dense_min=float(get_setting(engine, "dense_min")),
                                          flt=contract_scope(args.question))
        print(result.answer)
        for i, c in enumerate(result.citations, 1):
            print(f"  [{i}] {c.doc_id} (page {c.page})")


if __name__ == "__main__":
    main()
