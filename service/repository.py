"""Thin data-access helpers over the ORM. All functions take an explicit session."""
from typing import Optional

from sqlalchemy import func, select

from .models import CallResultRow, Run, RunStatus


def create_run(session, *, stage, budget_usd=None, max_calls=None, concurrency=None, note=None) -> Run:
    run = Run(
        stage=stage,
        status=RunStatus.QUEUED,
        budget_usd=budget_usd,
        max_calls=max_calls,
        concurrency=concurrency,
        note=note,
    )
    session.add(run)
    session.flush()  # populate run.id
    return run


def get_run(session, run_id: str) -> Optional[Run]:
    return session.get(Run, run_id)


def list_runs(session, *, limit: int = 50, offset: int = 0, status: Optional[str] = None):
    stmt = select(Run).order_by(Run.created_at.desc())
    if status:
        stmt = stmt.where(Run.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.execute(stmt).scalars())


def list_results(session, run_id: str, *, limit: int = 500, offset: int = 0, status: Optional[str] = None):
    stmt = select(CallResultRow).where(CallResultRow.run_id == run_id).order_by(CallResultRow.id)
    if status:
        stmt = stmt.where(CallResultRow.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.execute(stmt).scalars())


def done_keys_for_run(session, run_id: str) -> set:
    """Resume support: (brief_id, task, config_id, model_key, run_index) already OK for this run."""
    from src.llm_client import DONE_STATUSES

    stmt = select(
        CallResultRow.brief_id,
        CallResultRow.task,
        CallResultRow.config_id,
        CallResultRow.model_key,
        CallResultRow.run_index,
    ).where(CallResultRow.run_id == run_id, CallResultRow.status.in_(DONE_STATUSES))
    return set(session.execute(stmt).all())


def run_metrics(session, run_id: str) -> dict:
    """Operational metrics computed straight from the stored call rows."""
    run = session.get(Run, run_id)
    if run is None:
        return {}

    rows = list(session.execute(
        select(
            CallResultRow.status,
            CallResultRow.model_key,
            CallResultRow.cost_usd,
            CallResultRow.input_tokens,
            CallResultRow.output_tokens,
            CallResultRow.latency_s,
        ).where(CallResultRow.run_id == run_id)
    ).all())

    status_counts: dict[str, int] = {}
    cost_by_model: dict[str, float] = {}
    calls_by_model: dict[str, int] = {}
    tokens_in = tokens_out = 0
    latency_sum = 0.0
    for status, model_key, cost, tin, tout, lat in rows:
        status_counts[status] = status_counts.get(status, 0) + 1
        cost_by_model[model_key] = round(cost_by_model.get(model_key, 0.0) + (cost or 0.0), 6)
        calls_by_model[model_key] = calls_by_model.get(model_key, 0) + 1
        tokens_in += tin or 0
        tokens_out += tout or 0
        latency_sum += lat or 0.0

    n = len(rows)
    ok = status_counts.get("ok", 0) + status_counts.get("ok_length_violation", 0)
    return {
        "run_id": run_id,
        "status": run.status,
        "call_count": n,
        "total_cost_usd": round(sum(cost_by_model.values()), 6),
        "ok_rate": round(ok / n, 4) if n else None,
        "status_counts": status_counts,
        "cost_by_model": cost_by_model,
        "calls_by_model": calls_by_model,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "mean_latency_s": round(latency_sum / n, 3) if n else None,
    }
