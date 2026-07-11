"""
Persistence models.

Run            — one submitted evaluation job (a "stage" of the experiment).
CallResultRow  — one LLM call result, mirroring src.llm_client.CallResult,
                 foreign-keyed to its Run. This is the durable home for what
                 used to live only in outputs/results.jsonl.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from .db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RunStatus:
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"

    TERMINAL = {SUCCEEDED, FAILED, BUDGET_EXCEEDED, CANCELLED}


class Run(Base):
    __tablename__ = "runs"

    id = Column(String(32), primary_key=True, default=_uuid)
    stage = Column(String(32), nullable=False)
    status = Column(String(24), nullable=False, default=RunStatus.QUEUED, index=True)

    # request parameters
    budget_usd = Column(Float, nullable=True)
    max_calls = Column(Integer, nullable=True)
    concurrency = Column(Integer, nullable=True)
    note = Column(Text, nullable=True)

    # bookkeeping
    job_id = Column(String(64), nullable=True)  # RQ job id (null when inline)
    planned_calls = Column(Integer, nullable=True)
    call_count = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Float, nullable=False, default=0.0)
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)

    results = relationship(
        "CallResultRow",
        back_populates="run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "stage": self.stage,
            "status": self.status,
            "budget_usd": self.budget_usd,
            "max_calls": self.max_calls,
            "concurrency": self.concurrency,
            "note": self.note,
            "job_id": self.job_id,
            "planned_calls": self.planned_calls,
            "call_count": self.call_count,
            "total_cost_usd": round(self.total_cost_usd or 0.0, 6),
            "error": self.error,
            "created_at": _iso(self.created_at),
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


class CallResultRow(Base):
    __tablename__ = "call_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(32), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)

    # resume identity
    brief_id = Column(String(128), nullable=False)
    task = Column(String(64), nullable=False)
    config_id = Column(String(256), nullable=False)
    model_key = Column(String(32), nullable=False)
    run_index = Column(Integer, nullable=False)  # CallResult.run_id (1..N)

    status = Column(String(32), nullable=False, index=True)
    parsed_output = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)
    finish_reason = Column(String(32), nullable=True)

    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    reasoning_tokens = Column(Integer, nullable=False, default=0)
    cached_input_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)

    latency_s = Column(Float, nullable=False, default=0.0)
    error = Column(Text, nullable=True)
    provider_request_id = Column(String(128), nullable=True)
    ts = Column(String(32), nullable=True)  # provider timestamp string

    run = relationship("Run", back_populates="results")

    __table_args__ = (
        Index("ix_call_results_run_model", "run_id", "model_key"),
        Index("ix_call_results_run_task", "run_id", "task"),
    )

    @classmethod
    def from_call_result(cls, run_id: str, cr) -> "CallResultRow":
        """Build a row from a src.llm_client.CallResult dataclass."""
        return cls(
            run_id=run_id,
            brief_id=cr.brief_id,
            task=cr.task,
            config_id=cr.config_id,
            model_key=cr.model_key,
            run_index=cr.run_id,
            status=cr.status,
            parsed_output=cr.parsed_output,
            raw_response=cr.raw_response,
            finish_reason=cr.finish_reason,
            input_tokens=cr.input_tokens,
            output_tokens=cr.output_tokens,
            reasoning_tokens=cr.reasoning_tokens,
            cached_input_tokens=cr.cached_input_tokens,
            cost_usd=cr.cost_usd,
            latency_s=cr.latency_s,
            error=cr.error,
            provider_request_id=cr.provider_request_id,
            ts=cr.timestamp,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "brief_id": self.brief_id,
            "task": self.task,
            "config_id": self.config_id,
            "model_key": self.model_key,
            "run_index": self.run_index,
            "status": self.status,
            "parsed_output": self.parsed_output,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cost_usd": round(self.cost_usd or 0.0, 6),
            "latency_s": self.latency_s,
            "error": self.error,
            "provider_request_id": self.provider_request_id,
            "ts": self.ts,
        }


def _iso(dt) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
