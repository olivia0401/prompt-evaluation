"""
FastAPI surface over the evaluation engine.

    uvicorn service.api:app --reload

Endpoints:
    GET  /health                      liveness + queue mode
    GET  /stages                      available experiment stages
    POST /runs                        submit a run (async via RQ, or inline)
    GET  /runs                        list runs (newest first)
    GET  /runs/{id}                   run detail
    GET  /runs/{id}/results           per-call results (paginated)
    GET  /runs/{id}/metrics           operational metrics for the run
    POST /runs/{id}/cancel            best-effort cancel of a queued run
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from . import repository as repo
from . import settings
from .db import SessionLocal, init_db
from .models import RunStatus
from .schemas import RunCreate, RunMetrics, RunOut
from .tasks import enqueue_run


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "queue": "redis" if settings.redis_available() else "inline",
        "database": settings.DATABASE_URL.split("://", 1)[0],
    }


@app.get("/stages")
def stages():
    from src import config as cfg
    from .runner import STAGE_BUDGET_KEY

    out = []
    for stage, key in STAGE_BUDGET_KEY.items():
        out.append({"stage": stage, "default_budget_usd": cfg.BUDGET_CAP.get(key)})
    return {"stages": out}


@app.post("/runs", response_model=RunOut, status_code=201)
def create_run(body: RunCreate, db: Session = Depends(get_db)):
    from .runner import STAGE_BUDGET_KEY

    if body.stage not in STAGE_BUDGET_KEY:
        raise HTTPException(422, f"Unknown stage '{body.stage}'. Valid: {list(STAGE_BUDGET_KEY)}")

    run = repo.create_run(
        db,
        stage=body.stage,
        budget_usd=body.budget_usd,
        max_calls=body.max_calls,
        concurrency=body.concurrency,
        note=body.note,
    )
    db.commit()
    run_id = run.id

    # enqueue_run runs inline when Redis is absent — it opens its own session,
    # so refresh ours afterwards to return the up-to-date row.
    enqueue_run(run_id)
    db.expire_all()
    run = repo.get_run(db, run_id)
    return RunOut(**run.to_dict())


@app.get("/runs", response_model=list[RunOut])
def list_runs(
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
):
    runs = repo.list_runs(db, limit=limit, offset=offset, status=status)
    return [RunOut(**r.to_dict()) for r in runs]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = repo.get_run(db, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunOut(**run.to_dict())


@app.get("/runs/{run_id}/results")
def get_results(
    run_id: str,
    db: Session = Depends(get_db),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
):
    if repo.get_run(db, run_id) is None:
        raise HTTPException(404, f"Run {run_id} not found")
    rows = repo.list_results(db, run_id, limit=limit, offset=offset, status=status)
    return {"run_id": run_id, "count": len(rows), "results": [r.to_dict() for r in rows]}


@app.get("/runs/{run_id}/metrics", response_model=RunMetrics)
def get_metrics(run_id: str, db: Session = Depends(get_db)):
    if repo.get_run(db, run_id) is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunMetrics(**repo.run_metrics(db, run_id))


@app.post("/runs/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: str, db: Session = Depends(get_db)):
    run = repo.get_run(db, run_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    if run.status in RunStatus.TERMINAL:
        raise HTTPException(409, f"Run already {run.status}")
    if run.status == RunStatus.RUNNING:
        # We can't safely interrupt an in-flight worker; report honestly.
        raise HTTPException(409, "Run is executing; cancel of in-flight runs is not supported")

    # Queued run: try to drop the RQ job, then mark cancelled.
    if run.job_id and settings.redis_available():
        try:
            from redis import Redis
            from rq.job import Job

            Job.fetch(run.job_id, connection=Redis.from_url(settings.REDIS_URL)).cancel()
        except Exception:
            pass
    run.status = RunStatus.CANCELLED
    db.commit()
    return RunOut(**run.to_dict())
