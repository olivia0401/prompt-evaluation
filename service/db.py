"""
SQLAlchemy engine + session factory.

`init_db()` creates tables (fine for SQLite / dev / tests). For Postgres in
production prefer Alembic migrations, but create_all is safe and idempotent.
"""
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from . import settings

Base = declarative_base()

# SQLite needs check_same_thread=False because the runner writes results from
# the request/worker thread; a short-lived session per write keeps it safe.
_engine_kwargs = {"future": True, "pool_pre_ping": True}
if settings.DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables. Idempotent."""
    from . import models  # noqa: F401 — register mappers before create_all

    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope():
    """Transactional session context. Commits on success, rolls back on error."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
