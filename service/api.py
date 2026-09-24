"""
FastAPI surface over the evaluation engine.

    uvicorn service.api:app --reload

Every route below /health is tenant-scoped: a principal only ever sees rows
belonging to its own tenant, and a request for another tenant's run returns 404
(not 403 - see service/auth.py for why).

Endpoints:
    GET  /health                      liveness + queue mode (public)
    GET  /whoami                      the calling principal: name, tenant, role
    GET  /stages                      available experiment stages
    POST /runs                        submit a run (async via RQ, or inline)
    GET  /runs                        list runs (newest first)
    GET  /runs/{id}                   run detail
    GET  /runs/{id}/results           per-call results (paginated)
    GET  /runs/{id}/metrics           operational metrics for the run
    POST /runs/{id}/cancel            best-effort cancel of a queued run
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from sqlalchemy.orm import Session

from . import repository as repo
from . import settings
from .auth import ANONYMOUS_ADMIN, Principal, load_registry
from .db import SessionLocal, init_db
from .models import RunStatus
from .schemas import QualityReportIn, QualityReportOut, RunCreate, RunMetrics, RunOut
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


def get_registry():
    """Resolved per request so tests (and key rotation) can change the roster."""
    return load_registry()


def current_principal(authorization: str | None = Header(default=None)) -> Principal:
    """Resolve the caller. 401 when a key is required and missing or wrong.

    In open mode (no keys configured at all) every caller is an admin on the
    `default` tenant - the local-dev and unit-test posture. That is a
    deliberate, documented default rather than an accident: `GET /whoami`
    reports `"open_mode": true`, so it is visible rather than assumed.
    """
    registry = get_registry()
    if registry.open_mode:
        return ANONYMOUS_ADMIN
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    principal = registry.resolve(token)
    if principal is None:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token")
    return principal


def require_write(principal: Principal = Depends(current_principal)) -> Principal:
    """Create/submit routes. A viewer is authenticated but not authorised: 403.

    403 is right here and 404 is right for cross-tenant, and the difference is
    the point: within your own tenant you already know the resource exists, so
    refusing loudly leaks nothing and tells the caller something useful.
    """
    if not principal.may_write:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{principal.role}' cannot write. Requires operator or admin.",
        )
    return principal


def require_cancel(principal: Principal = Depends(current_principal)) -> Principal:
    """Cancelling destroys in-flight work and is admin-only."""
    if not principal.may_cancel:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{principal.role}' cannot cancel a run. Requires admin.",
        )
    return principal


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": settings.API_VERSION,
        "queue": "redis" if settings.redis_available() else "inline",
        "database": settings.DATABASE_URL.split("://", 1)[0],
    }


@app.get("/whoami")
def whoami(principal: Principal = Depends(current_principal)):
    """Who the presented key says you are.

    The first thing to check when an isolation test fails: the wrong persona is
    a far more common cause than a bug in the scoping.
    """
    return {**principal.to_dict(), "open_mode": get_registry().open_mode}


@app.post("/quality-reports", response_model=QualityReportOut, status_code=201)
def create_quality_report(body: QualityReportIn, db: Session = Depends(get_db),
                          principal: Principal = Depends(require_write)):
    """Store a quality snapshot and return its release-gate decision."""
    from service.quality_gate import check_report

    passed, errors = check_report(body.report)
    report = dict(body.report)
    report["gate_errors"] = errors
    row = repo.create_quality_report(db, report, passed,
                                     tenant_id=principal.tenant_id,
                                     created_by=principal.name)
    db.commit()
    return QualityReportOut(**row.to_dict())


@app.get("/quality-reports", response_model=list[QualityReportOut])
def list_quality_reports(
    db: Session = Depends(get_db),
    principal: Principal = Depends(current_principal),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    return [QualityReportOut(**row.to_dict())
            for row in repo.list_quality_reports(db, tenant_id=principal.tenant_id,
                                                 limit=limit, offset=offset)]


@app.get("/stages")
def stages():
    from src import config as cfg
    from .runner import STAGE_BUDGET_KEY

    out = []
    for stage, key in STAGE_BUDGET_KEY.items():
        out.append({"stage": stage, "default_budget_usd": cfg.BUDGET_CAP.get(key)})
    return {"stages": out}


@app.post("/runs", response_model=RunOut, status_code=201)
def create_run(body: RunCreate, db: Session = Depends(get_db),
               principal: Principal = Depends(require_write)):
    from .runner import STAGE_BUDGET_KEY

    if body.stage not in STAGE_BUDGET_KEY:
        raise HTTPException(422, f"Unknown stage '{body.stage}'. Valid: {list(STAGE_BUDGET_KEY)}")

    run = repo.create_run(
        db,
        tenant_id=principal.tenant_id,
        created_by=principal.name,
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
    run = repo.get_run(db, run_id, tenant_id=principal.tenant_id)
    return RunOut(**run.to_dict())


@app.get("/runs", response_model=list[RunOut])
def list_runs(
    db: Session = Depends(get_db),
    principal: Principal = Depends(current_principal),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
):
    runs = repo.list_runs(db, tenant_id=principal.tenant_id,
                          limit=limit, offset=offset, status=status)
    return [RunOut(**r.to_dict()) for r in runs]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, db: Session = Depends(get_db),
            principal: Principal = Depends(current_principal)):
    run = repo.get_run(db, run_id, tenant_id=principal.tenant_id)
    if run is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunOut(**run.to_dict())


@app.get("/runs/{run_id}/results")
def get_results(
    run_id: str,
    db: Session = Depends(get_db),
    principal: Principal = Depends(current_principal),
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None),
):
    if repo.get_run(db, run_id, tenant_id=principal.tenant_id) is None:
        raise HTTPException(404, f"Run {run_id} not found")
    rows = repo.list_results(db, run_id, tenant_id=principal.tenant_id,
                             limit=limit, offset=offset, status=status)
    return {"run_id": run_id, "count": len(rows), "results": [r.to_dict() for r in rows]}


@app.get("/runs/{run_id}/metrics", response_model=RunMetrics)
def get_metrics(run_id: str, db: Session = Depends(get_db),
                principal: Principal = Depends(current_principal)):
    if repo.get_run(db, run_id, tenant_id=principal.tenant_id) is None:
        raise HTTPException(404, f"Run {run_id} not found")
    return RunMetrics(**repo.run_metrics(db, run_id, tenant_id=principal.tenant_id))


@app.post("/runs/{run_id}/cancel", response_model=RunOut)
def cancel_run(run_id: str, db: Session = Depends(get_db),
               principal: Principal = Depends(require_cancel)):
    run = repo.get_run(db, run_id, tenant_id=principal.tenant_id)
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
