"""
Runtime configuration for the service layer, read from environment variables.

Everything has a sensible zero-infra default so the API/dashboard run locally
without Postgres or Redis. Set the env vars (see .env.example / docker-compose)
to switch to the production stack.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Database -------------------------------------------------------------
# Default: a local SQLite file so the service runs with no external DB.
# Production: postgresql+psycopg://user:pass@host:5432/evals
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(PROJECT_ROOT / 'data' / 'service.db').as_posix()}",
)

# --- Queue ----------------------------------------------------------------
# Redis connection for RQ. When unreachable (or INLINE_JOBS=1) the API falls
# back to running the job synchronously inside the request — handy for local
# dev, tests and demos.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RQ_QUEUE_NAME = os.getenv("RQ_QUEUE_NAME", "evals")

# Force inline (synchronous) execution regardless of Redis availability.
INLINE_JOBS = os.getenv("INLINE_JOBS", "").lower() in {"1", "true", "yes"}

# Worker job timeout (seconds). A full Stage A can take many minutes.
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", str(6 * 60 * 60)))

# --- Runner knobs ---------------------------------------------------------
# Upper bound on total in-flight API calls per run. Per-model RPM/TPM limits
# are still enforced inside LLMClient via config.CONCURRENCY semaphores.
RUN_CONCURRENCY = int(os.getenv("RUN_CONCURRENCY", "20"))

# Keep appending each CallResult to outputs/results.jsonl (in addition to the
# DB) so the existing analyze / build_xlsx / audit chain keeps working.
MIRROR_TO_JSONL = os.getenv("MIRROR_TO_JSONL", "1").lower() in {"1", "true", "yes"}

# --- API ------------------------------------------------------------------
API_TITLE = "Prompt Evaluation Service"
API_VERSION = "0.1.0"

# Optional single bearer token (admin on the "default" tenant) guarding every
# route except GET /health. Ignored when SERVICE_PRINCIPALS is set. When both
# are empty (the default) the API runs open — fine for local dev and tests.
# Set one before exposing the service, or anyone who can reach it could spend
# your LLM budget.
API_TOKEN = os.getenv("API_TOKEN", "")

# Multi-tenant principals as a JSON array; see service/auth.py.
#   [{"key": "...", "name": "acme-admin", "tenant": "acme", "role": "admin"}]
# Unset -> fall back to the single API_TOKEN; both unset -> open mode.
SERVICE_PRINCIPALS = os.getenv("SERVICE_PRINCIPALS", "")

# --- Safety ---------------------------------------------------------------
# Kill switch for actually executing a submitted run. The API surface stays
# fully live (runs are created, listed, scoped, cancelled) but nothing reaches
# an LLM provider, so a run costs $0.
#
# This exists because the browser suite POSTs real runs against a real server.
# Without it, `npx playwright test` would spend money on every execution, need
# provider keys in CI, and — worse — mark runs FAILED when those keys are
# absent, which would make the cancel tests fail for a reason that has nothing
# to do with what they are testing.
DISABLE_RUN_EXECUTION = os.getenv("DISABLE_RUN_EXECUTION", "").lower() in {"1", "true", "yes"}


def redis_available() -> bool:
    """True if a Redis server answers a PING at REDIS_URL."""
    if INLINE_JOBS:
        return False
    try:
        from redis import Redis

        Redis.from_url(REDIS_URL, socket_connect_timeout=1).ping()
        return True
    except Exception:
        return False
