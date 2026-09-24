"""Thin data-access helpers over the ORM. All functions take an explicit session.

Tenancy note: every function that reads or writes a tenant-owned row takes a
required ``tenant_id``. It is a positional-ish keyword with no default on
purpose — a default would let a new call site silently read across tenants, and
that bug is invisible until it is a breach. ``get_run`` returns None for a run
belonging to another tenant, so callers turn it into a 404 without a special
case (a 403 would confirm the id exists).
"""
from typing import Optional

from sqlalchemy import func, select

from .models import CallResultRow, QualityReport, Run, RunStatus
import json


def create_run(session, *, tenant_id, created_by=None, stage, budget_usd=None,
               max_calls=None, concurrency=None, note=None) -> Run:
    run = Run(
        tenant_id=tenant_id,
        created_by=created_by,
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


def get_run(session, run_id: str, *, tenant_id: str) -> Optional[Run]:
    """None when the run does not exist OR belongs to another tenant.

    Collapsing "missing" and "not yours" into one answer is deliberate: the
    caller renders 404 either way, so run ids cannot be enumerated.
    """
    run = session.get(Run, run_id)
    if run is None or run.tenant_id != tenant_id:
        return None
    return run


def list_runs(session, *, tenant_id: str, limit: int = 50, offset: int = 0,
              status: Optional[str] = None):
    stmt = select(Run).where(Run.tenant_id == tenant_id).order_by(Run.created_at.desc())
    if status:
        stmt = stmt.where(Run.status == status)
    stmt = stmt.limit(limit).offset(offset)
    return list(session.execute(stmt).scalars())


def list_results(session, run_id: str, *, tenant_id: str, limit: int = 500,
                 offset: int = 0, status: Optional[str] = None):
    # Join through the parent run rather than trusting run_id alone: call rows
    # have no tenant column of their own, so their scope is their run's scope.
    stmt = (
        select(CallResultRow)
        .join(Run, Run.id == CallResultRow.run_id)
        .where(CallResultRow.run_id == run_id, Run.tenant_id == tenant_id)
        .order_by(CallResultRow.id)
    )
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


def run_metrics(session, run_id: str, *, tenant_id: str) -> dict:
    """Operational metrics computed straight from the stored call rows."""
    run = get_run(session, run_id, tenant_id=tenant_id)
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


def create_quality_report(session, report: dict, passed: bool, *, tenant_id,
                          created_by=None) -> QualityReport:
    provenance = report.get("provenance", {})
    row = QualityReport(
        tenant_id=tenant_id,
        created_by=created_by,
        dataset_version=str(report.get("dataset_version", "unknown")),
        evaluator_version=str(provenance.get("evaluator_version", "unknown")),
        passed=int(bool(passed)),
        report_json=json.dumps(report, ensure_ascii=False, sort_keys=True),
    )
    session.add(row)
    session.flush()
    return row


def list_quality_reports(session, *, tenant_id: str, limit: int = 50, offset: int = 0):
    stmt = (select(QualityReport)
            .where(QualityReport.tenant_id == tenant_id)
            .order_by(QualityReport.created_at.desc()))
    return list(session.execute(stmt.limit(limit).offset(offset)).scalars())
