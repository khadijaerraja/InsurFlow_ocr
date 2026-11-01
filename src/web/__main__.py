"""Start InsurFlow:  python -m src.web [--port 8000] [--no-dense] [--threads 8]

Reads DATABASE_URL (or POSTGRES_*), SECRET_KEY, APP_ENV, ALLOW_REGISTRATION from the environment / .env.
Create the first account from the sign-up page (development) or with `python -m src.cli create-user`.
"""

import argparse

from src.web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m src.web")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-dense", action="store_true", help="keyword search only (no embedding model)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    app = create_app(use_dense=not args.no_dense)
    print(f"InsurFlow running on http://127.0.0.1:{args.port}")
    try:  # waitress is a production-grade WSGI server (works on Windows); Flask's own server is a fallback
        from waitress import serve

        serve(app, host="127.0.0.1", port=args.port, threads=args.threads)
    except ImportError:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
