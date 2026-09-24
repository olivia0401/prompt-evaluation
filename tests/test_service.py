"""
Hermetic tests for the service layer.

No Redis, no Postgres, no real API: a temp SQLite DB, INLINE_JOBS, a fake LLM
client and an injected todo list exercise persistence, budgeting, resume, the
FastAPI surface and the eval gate end to end.

Env must be set BEFORE importing the service package (settings reads it at import).
"""
import os
import tempfile

_dbfd, _dbpath = tempfile.mkstemp(suffix=".db")
os.close(_dbfd)
os.environ["DATABASE_URL"] = "sqlite:///" + _dbpath.replace("\\", "/")
os.environ["INLINE_JOBS"] = "1"
os.environ["MIRROR_TO_JSONL"] = "0"

import pytest  # noqa: E402

from service import repository as repo  # noqa: E402
from service import runner  # noqa: E402
from service.db import init_db, session_scope  # noqa: E402
from service.models import RunStatus  # noqa: E402
from src.llm_client import CallResult, Status  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _db():
    init_db()
    yield


def fake_todo(stage, briefs=("b1", "b2"), models=("haiku", "gpt5mini")):
    jobs = []
    for b in briefs:
        for m in models:
            jobs.append({
                "brief_id": b, "task": "t1", "config_id": "A:x",
                "model_key": m, "run_id": 1, "prompt": "hello",
            })
    return jobs


class FakeClient:
    """Minimal stand-in for LLMClient: deterministic OK results + budget cap."""

    def __init__(self, *, budget_cap_usd=None, cost=0.001, **_ignore):
        self.budget_cap_usd = budget_cap_usd
        self.cost = cost
        self.call_count = 0
        self.total_cost_usd = 0.0

    def call(self, *, model_key, prompt, brief_id, task, config_id, run_id):
        if self.budget_cap_usd is not None and self.total_cost_usd >= self.budget_cap_usd:
            return CallResult(brief_id=brief_id, task=task, config_id=config_id,
                              model_key=model_key, run_id=run_id, prompt=prompt,
                              status=Status.BUDGET_EXCEEDED, error="cap reached")
        cr = CallResult(brief_id=brief_id, task=task, config_id=config_id,
                        model_key=model_key, run_id=run_id, prompt=prompt,
                        status=Status.OK, parsed_output="out",
                        input_tokens=10, output_tokens=5, cost_usd=self.cost)
        self.call_count += 1
        self.total_cost_usd = round(self.total_cost_usd + self.cost, 6)
        return cr


def _new_run(stage="phase0", tenant_id="default", **kw):
    with session_scope() as s:
        run = repo.create_run(s, tenant_id=tenant_id, stage=stage, **kw)
        return run.id


def test_execute_run_records_results():
    run_id = _new_run()
    out = runner.execute_run(run_id, todo=fake_todo("phase0"), client=FakeClient())
    assert out["status"] == RunStatus.SUCCEEDED
    assert out["call_count"] == 4
    with session_scope() as s:
        rows = repo.list_results(s, run_id, tenant_id="default")
        assert len(rows) == 4
        m = repo.run_metrics(s, run_id, tenant_id="default")
        assert m["ok_rate"] == 1.0
        assert set(m["calls_by_model"]) == {"haiku", "gpt5mini"}


def test_budget_exceeded_marks_run():
    run_id = _new_run(stage="stage_a", budget_usd=0.0025)
    # cap 0.0025 at 0.001/call -> ~3 calls then budget_exceeded
    out = runner.execute_run(run_id, todo=fake_todo("stage_a"),
                             client=FakeClient(budget_cap_usd=0.0025))
    assert out["status"] == RunStatus.BUDGET_EXCEEDED


def test_resume_skips_completed():
    run_id = _new_run()
    runner.execute_run(run_id, todo=fake_todo("phase0"), client=FakeClient())
    # Second pass: everything already OK -> no new rows, still succeeds.
    c2 = FakeClient()
    runner.execute_run(run_id, todo=fake_todo("phase0"), client=c2)
    assert c2.call_count == 0
    with session_scope() as s:
        assert len(repo.list_results(s, run_id, tenant_id="default")) == 4


def test_todo_build_failure_marks_failed():
    run_id = _new_run()

    def boom(stage):
        raise RuntimeError("no briefs")

    out = runner.execute_run(run_id, todo_builder=boom, client=FakeClient())
    assert out["status"] == RunStatus.FAILED
    assert "no briefs" in out["error"]


def test_api_end_to_end(monkeypatch):
    from fastapi.testclient import TestClient

    from service import api

    monkeypatch.setattr(runner, "_default_todo_builder", lambda stage: fake_todo(stage))
    monkeypatch.setattr(runner, "_default_client_factory",
                        lambda **kw: FakeClient(budget_cap_usd=kw.get("budget_cap_usd")))

    client = TestClient(api.app)
    assert client.get("/health").json()["queue"] == "inline"

    r = client.post("/runs", json={"stage": "phase0", "note": "smoke"})
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["status"] == RunStatus.SUCCEEDED
    run_id = run["id"]

    assert client.get(f"/runs/{run_id}").json()["call_count"] == 4
    metrics = client.get(f"/runs/{run_id}/metrics").json()
    assert metrics["call_count"] == 4 and metrics["ok_rate"] == 1.0
    results = client.get(f"/runs/{run_id}/results").json()
    assert results["count"] == 4
    assert client.get("/runs/does-not-exist").status_code == 404


def test_api_token_guards_runs(monkeypatch):
    """When API_TOKEN is set, POST /runs needs a valid bearer token; /health
    stays open."""
    from fastapi.testclient import TestClient

    from service import api
    from service import settings as svc_settings

    monkeypatch.setattr(svc_settings, "API_TOKEN", "s3cret")
    monkeypatch.setattr(runner, "_default_todo_builder", lambda stage: fake_todo(stage))
    monkeypatch.setattr(runner, "_default_client_factory",
                        lambda **kw: FakeClient(budget_cap_usd=kw.get("budget_cap_usd")))

    client = TestClient(api.app)

    assert client.get("/health").status_code == 200  # liveness stays open
    assert client.post("/runs", json={"stage": "phase0"}).status_code == 401
    assert client.post("/runs", json={"stage": "phase0"},
                       headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post("/runs", json={"stage": "phase0", "note": "auth"},
                     headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 201, ok.text


def test_ci_gate(tmp_path):
    import pandas as pd

    from service import ci_gate

    scored = tmp_path / "scored.csv"
    pd.DataFrame([
        {"task": "t1", "config_id": "A:x", "model_key": "haiku", "status": "ok", "cosine": 0.80, "f1": None},
        {"task": "t1", "config_id": "A:x", "model_key": "gpt5mini", "status": "ok", "cosine": 0.82, "f1": None},
        {"task": "keywords", "config_id": "B:y", "model_key": "haiku", "status": "ok", "cosine": None, "f1": 0.70},
    ]).to_csv(scored, index=False)

    metrics = ci_gate.compute_metrics(scored)
    assert metrics["tasks"]["t1"]["metric"] == "cosine"
    assert metrics["tasks"]["keywords"]["metric"] == "f1"
    assert metrics["ok_rate"] == 1.0

    baseline = {"tolerance": 0.02, "min_ok_rate": 0.9,
                "tasks": {"t1": {"metric": "cosine", "value": 0.81},
                          "keywords": {"metric": "f1", "value": 0.69}}}
    passed, _ = ci_gate.check(metrics, baseline)
    assert passed is True

    # Regress t1 well beyond tolerance -> fail.
    baseline["tasks"]["t1"]["value"] = 0.90
    passed, _ = ci_gate.check(metrics, baseline)
    assert passed is False


def test_quality_report_api_stores_pass_and_fail(monkeypatch):
    from fastapi.testclient import TestClient
    from service import api

    report = {
        "schema": "quality-report/v1",
        "dataset_version": "golden-test",
        "metrics": {
            "groundedness": 0.95, "citation_completeness": 0.98,
            "unsupported_claim_rate": 0.01, "entity_resolution_f1": 0.95,
            "judge_weighted_kappa": 0.75, "source_acceptable_rate": 0.95,
        },
        "denominators": {"claims": 120, "mentions": 340, "sources": 95,
                         "judge_ratings": 40},
        "provenance": {"evaluator_version": "test", "golden_manifest_sha256": "abc"},
    }
    client = TestClient(api.app)
    response = client.post("/quality-reports", json={"report": report})
    assert response.status_code == 201
    assert response.json()["passed"] is True
    assert client.get("/quality-reports").status_code == 200

    report["metrics"]["groundedness"] = 0.2
    response = client.post("/quality-reports", json={"report": report})
    assert response.status_code == 201
    assert response.json()["passed"] is False
    assert any("groundedness" in e for e in response.json()["errors"])


def test_ci_gate_pins_the_baselined_recipe(tmp_path):
    """The gate must score the recipe the baseline names, not the run's best.

    Regression test for a gate that took `max` over every config: recipe A wins
    today, recipe B wins tomorrow, and the gate silently compares two different
    recipes' numbers and calls the gap a quality regression.
    """
    import pandas as pd

    from service import ci_gate

    scored = tmp_path / "scored.csv"
    pd.DataFrame([
        # The baselined recipe held steady at 0.80...
        {"task": "t1", "config_id": "A:pinned", "model_key": "haiku", "status": "ok",
         "cosine": 0.80, "f1": None},
        # ...while an unrelated experimental recipe happens to score higher.
        {"task": "t1", "config_id": "Z:experimental", "model_key": "haiku", "status": "ok",
         "cosine": 0.95, "f1": None},
    ]).to_csv(scored, index=False)

    pinned = ci_gate.compute_metrics(scored, pinned={"t1": "A:pinned"})
    assert pinned["tasks"]["t1"]["config"] == "A:pinned"
    assert pinned["tasks"]["t1"]["value"] == 0.80
    assert pinned["tasks"]["t1"]["pin_status"] == "pinned"

    # With no pin (baseline creation) the leader is the right answer.
    fresh = ci_gate.compute_metrics(scored)
    assert fresh["tasks"]["t1"]["config"] == "Z:experimental"
    assert fresh["tasks"]["t1"]["pin_status"] == "best"


def test_ci_gate_fails_when_the_baselined_recipe_is_absent(tmp_path):
    """A run that no longer contains the pinned recipe is unverifiable, not passing."""
    import pandas as pd

    from service import ci_gate

    scored = tmp_path / "scored.csv"
    pd.DataFrame([
        {"task": "t1", "config_id": "Z:other", "model_key": "haiku", "status": "ok",
         "cosine": 0.99, "f1": None},
    ]).to_csv(scored, index=False)

    metrics = ci_gate.compute_metrics(scored, pinned={"t1": "A:pinned"})
    assert metrics["tasks"]["t1"]["pin_status"] == "missing"

    baseline = {"tolerance": 0.036, "min_ok_rate": 0.9,
                "tasks": {"t1": {"metric": "cosine", "value": 0.80, "config": "A:pinned"}}}
    passed, lines = ci_gate.check(metrics, baseline)
    assert passed is False
    assert any("recipe gone" in line for line in lines)


def test_ci_gate_tolerance_is_not_tighter_than_measured_noise():
    """A gate that fires inside the noise band teaches everyone to ignore it."""
    from src import config as cfg
    from service import ci_gate

    assert ci_gate.DEFAULT_TOLERANCE >= cfg.NOISE_FLOOR_COSINE
