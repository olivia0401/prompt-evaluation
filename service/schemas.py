"""Pydantic request/response models for the API."""
from typing import Optional

from pydantic import BaseModel, Field

STAGES = ["phase0", "stage_a", "stage_b", "phase4"]


class RunCreate(BaseModel):
    stage: str = Field(..., description="Experiment stage to run.", examples=["phase0"])
    budget_usd: Optional[float] = Field(
        None, description="USD cap for the whole run. Defaults to config.BUDGET_CAP[stage]."
    )
    max_calls: Optional[int] = Field(None, description="Hard cap on API calls this run.")
    concurrency: Optional[int] = Field(None, description="Worker-pool size (in-flight calls).")
    note: Optional[str] = Field(None, description="Free-text label for this run.")


class RunOut(BaseModel):
    id: str
    stage: str
    status: str
    budget_usd: Optional[float] = None
    max_calls: Optional[int] = None
    concurrency: Optional[int] = None
    note: Optional[str] = None
    job_id: Optional[str] = None
    planned_calls: Optional[int] = None
    call_count: int = 0
    total_cost_usd: float = 0.0
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class RunMetrics(BaseModel):
    run_id: str
    status: str
    call_count: int
    total_cost_usd: float
    ok_rate: Optional[float] = None
    status_counts: dict[str, int] = {}
    cost_by_model: dict[str, float] = {}
    calls_by_model: dict[str, int] = {}
    tokens_in: int = 0
    tokens_out: int = 0
    mean_latency_s: Optional[float] = None
