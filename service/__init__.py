"""
Service layer that wraps the batch evaluation engine (src/ + scripts/) as a
long-running application:

    FastAPI  -> submit eval runs / query results        (service.api)
    RQ+Redis -> execute runs asynchronously             (service.tasks, service.runner)
    Postgres -> persist runs + per-call results          (service.models)
    Streamlit-> dashboard over the above                 (service.dashboard)
    CI gate  -> block a release on eval regression       (service.ci_gate)

Design goals:
  - Zero-infra default: DATABASE_URL falls back to SQLite and jobs run inline,
    so `uvicorn service.api:app` works with nothing else installed.
  - Scale-up path: point DATABASE_URL at Postgres and REDIS_URL at Redis and
    the exact same code runs async via an RQ worker (see docker-compose.yml).
  - Reuse, don't fork: the actual LLM calls, budgeting, todo-building and
    scoring all come from the existing src/ + scripts/ engine unchanged.
"""

__all__ = ["settings"]
