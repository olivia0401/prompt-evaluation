"""
Core run execution — the bridge between a persisted Run row and the existing
batch engine (scripts.run_experiment.build_todo + src.llm_client.LLMClient).

`execute_run(run_id)` is what the RQ worker calls. It is deliberately
dependency-injectable (`client`, `todo`, `client_factory`) so tests can drive
the whole persistence + budgeting + resume path with a fake client and never
touch a real API.

Threading model mirrors scripts.run_experiment: API calls fan out across a
ThreadPoolExecutor, but every DB/JSONL write happens on the main thread as
futures complete — so we never share a SQLAlchemy session across threads.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone

from . import settings
from .db import SessionLocal
from .models import CallResultRow, Run, RunStatus

# stage -> config.BUDGET_CAP key (mirrors scripts.run_experiment.main)
STAGE_BUDGET_KEY = {
    "phase0": "phase_0",
    "stage_a": "phase_1",
    "stage_b": "stage_b",
    "phase4": "phase_4_premium",
}


def _utcnow():
    return datetime.now(timezone.utc)


def _default_todo_builder(stage: str) -> list[dict]:
    from scripts.run_experiment import build_todo

    return build_todo(stage)


def _default_client_factory(*, budget_cap_usd, call_cap, per_model_cap_usd):
    from src.llm_client import LLMClient

    return LLMClient.from_env(
        budget_cap_usd=budget_cap_usd,
        call_cap=call_cap,
        per_model_cap_usd=per_model_cap_usd,
    )


def _resolve_budget(stage: str, requested):
    if requested is not None:
        return requested
    from src import config as cfg

    return cfg.BUDGET_CAP.get(STAGE_BUDGET_KEY.get(stage, ""), None)


def execute_run(
    run_id: str,
    *,
    todo=None,
    client=None,
    todo_builder=None,
    client_factory=None,
    mirror_to_jsonl=None,
) -> dict:
    """
    Execute (or resume) a run to completion. Returns the final Run.to_dict().

    Never raises for per-call API failures — those are recorded as rows with a
    non-ok status, exactly like the batch runner. Only infrastructure failures
    (todo build error, etc.) mark the whole run FAILED.

    `todo_builder` / `client_factory` are resolved at call-time (not baked into
    defaults) so tests can monkeypatch the module-level defaults.
    """
    if mirror_to_jsonl is None:
        mirror_to_jsonl = settings.MIRROR_TO_JSONL
    todo_builder = todo_builder or _default_todo_builder
    client_factory = client_factory or _default_client_factory

    session = SessionLocal()
    try:
        run: Run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        run.status = RunStatus.RUNNING
        run.started_at = _utcnow()
        session.commit()

        stage = run.stage
        budget = _resolve_budget(stage, run.budget_usd)
        concurrency = run.concurrency or settings.RUN_CONCURRENCY

        # 1. Build the job list (skippable in tests via `todo=`).
        try:
            jobs = list(todo) if todo is not None else todo_builder(stage)
        except Exception as e:  # todo build is infrastructure — fail the run
            run.status = RunStatus.FAILED
            run.error = f"todo build failed: {type(e).__name__}: {e}"
            run.finished_at = _utcnow()
            session.commit()
            return run.to_dict()

        # 2. Resume: drop jobs already completed OK for THIS run.
        from .repository import done_keys_for_run

        done = done_keys_for_run(session, run_id)
        jobs = [
            j for j in jobs
            if (j["brief_id"], j["task"], j["config_id"], j["model_key"], j["run_id"]) not in done
        ]
        run.planned_calls = len(jobs)
        session.commit()

        if not jobs:
            _finalize(session, run, budget_exceeded=False)
            return run.to_dict()

        # 3. Client with budget kill-switches (unless injected).
        from src import config as cfg

        if client is None:
            client = client_factory(
                budget_cap_usd=budget,
                call_cap=run.max_calls,
                per_model_cap_usd=cfg.PER_MODEL_BUDGET_CAP_USD,
            )

        jsonl_path = cfg.OUTPUTS_DIR / "results.jsonl" if mirror_to_jsonl else None
        stop_event = threading.Event()

        def _one(job: dict):
            if stop_event.is_set():
                return None
            return client.call(
                model_key=job["model_key"],
                prompt=job["prompt"],
                brief_id=job["brief_id"],
                task=job["task"],
                config_id=job["config_id"],
                run_id=job["run_id"],
            )

        budget_exceeded = False
        written = 0
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
            futures = [ex.submit(_one, j) for j in jobs]
            for fut in as_completed(futures):
                cr = fut.result()
                if cr is None:
                    continue
                # DB write (main thread → single session, safe).
                session.add(CallResultRow.from_call_result(run_id, cr))
                written += 1
                if mirror_to_jsonl and jsonl_path is not None:
                    from src.utils import append_jsonl

                    append_jsonl(jsonl_path, asdict(cr))
                if cr.status == "budget_exceeded":
                    budget_exceeded = True
                    stop_event.set()
                if written % 25 == 0:
                    run.call_count = client.call_count
                    run.total_cost_usd = client.total_cost_usd
                    session.commit()

        run.call_count = client.call_count
        run.total_cost_usd = client.total_cost_usd
        _finalize(session, run, budget_exceeded=budget_exceeded)
        return run.to_dict()

    except Exception as e:
        session.rollback()
        run = session.get(Run, run_id)
        if run is not None:
            run.status = RunStatus.FAILED
            run.error = f"{type(e).__name__}: {e}"
            run.finished_at = _utcnow()
            session.commit()
            return run.to_dict()
        raise
    finally:
        session.close()


def _finalize(session, run: Run, *, budget_exceeded: bool) -> None:
    run.status = RunStatus.BUDGET_EXCEEDED if budget_exceeded else RunStatus.SUCCEEDED
    run.finished_at = _utcnow()
    session.commit()
