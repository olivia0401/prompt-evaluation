"""Smoke tests for scripts/run_experiment.build_todo. No API calls."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_runner():
    """Load run_experiment.py as a module (it lives under scripts/)."""
    here = Path(__file__).resolve().parent.parent / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_phase0_todo_has_expected_call_count():
    runner = _import_runner()
    todo = runner.build_todo("phase0")
    # 3 briefs × 17 configs × 2 models × 1 run = 102
    assert len(todo) == 102


def test_phase0_todo_has_required_keys():
    runner = _import_runner()
    todo = runner.build_todo("phase0")
    sample = todo[0]
    for k in ("brief_id", "task", "config_id", "model_key", "run_id", "prompt"):
        assert k in sample, f"missing key: {k}"
    assert sample["prompt"]  # non-empty


def test_phase0_todo_covers_both_models():
    runner = _import_runner()
    todo = runner.build_todo("phase0")
    models = {t["model_key"] for t in todo}
    assert models == {"haiku", "gpt5mini"}


def test_phase0_todo_includes_keyword_task():
    runner = _import_runner()
    todo = runner.build_todo("phase0")
    assert any(t["task"] == "keywords" for t in todo)


def test_phase0_todo_keys_unique():
    runner = _import_runner()
    todo = runner.build_todo("phase0")
    keys = [
        (t["brief_id"], t["task"], t["config_id"], t["model_key"], t["run_id"])
        for t in todo
    ]
    assert len(keys) == len(set(keys)), "duplicate (brief, task, config, model, run) keys"


def test_stage_b_missing_scored_raises_clearly(tmp_path, monkeypatch):
    """Stage B requires scored.csv (from analyze --score) — clear error if missing."""
    import shutil
    runner = _import_runner()
    # Point cfg.OUTPUTS_DIR to a temp dir with no scored.csv
    from src import config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="scored.csv"):
        runner.build_todo("stage_b")


def test_stage_b_picks_top_2_per_task(tmp_path, monkeypatch):
    """Top-2 configs per task by mean cosine; tie-break by worst-case (min)."""
    import pandas as pd
    runner = _import_runner()
    from src import config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path)

    # Synthetic scored.csv: 3 configs across 3 briefs for one task, +
    # 2 configs for the keyword task. We want top-2 by mean, ties by worst.
    rows = []
    # concept_relevant — config A wins (mean 0.7), B second (mean 0.6), C loses (0.4)
    for brief, score in [("Plonts", 0.7), ("Data Fabric", 0.7), ("Board", 0.7)]:
        rows.append({"brief_id": brief, "task": "concept_relevant",
                     "config_id": "A:product", "model_key": "haiku",
                     "run_id": 1, "cosine": score, "f1": None})
    for brief, score in [("Plonts", 0.6), ("Data Fabric", 0.6), ("Board", 0.6)]:
        rows.append({"brief_id": brief, "task": "concept_relevant",
                     "config_id": "A:audience", "model_key": "haiku",
                     "run_id": 1, "cosine": score, "f1": None})
    for brief, score in [("Plonts", 0.4), ("Data Fabric", 0.4), ("Board", 0.4)]:
        rows.append({"brief_id": brief, "task": "concept_relevant",
                     "config_id": "A:personality", "model_key": "haiku",
                     "run_id": 1, "cosine": score, "f1": None})
    # keyword task — version A wins by mean F1
    for brief, f1 in [("Plonts", 0.75), ("Data Fabric", 0.75), ("Board", 0.75)]:
        rows.append({"brief_id": brief, "task": "keywords",
                     "config_id": "A:_full_brief", "model_key": "haiku",
                     "run_id": 1, "cosine": None, "f1": f1})
    for brief, f1 in [("Plonts", 0.50), ("Data Fabric", 0.50), ("Board", 0.50)]:
        rows.append({"brief_id": brief, "task": "keywords",
                     "config_id": "B:_full_brief", "model_key": "haiku",
                     "run_id": 1, "cosine": None, "f1": f1})

    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "scored.csv", index=False)

    todo = runner.build_todo("stage_b")
    # Top-2 sentence configs: A:product, A:audience  (NOT A:personality)
    sentence_configs = {t["config_id"] for t in todo if t["task"] == "concept_relevant"}
    assert sentence_configs == {"A:product", "A:audience"}
    assert "A:personality" not in sentence_configs
    # Keyword top-2 includes both since only 2 exist
    kw_configs = {t["config_id"] for t in todo if t["task"] == "keywords"}
    assert kw_configs == {"A:_full_brief", "B:_full_brief"}
    # Runs are 2 and 3 (NOT 1 — Stage A already covers run_id=1)
    run_ids = {t["run_id"] for t in todo}
    assert run_ids == {2, 3}
    # Both cheap models present
    models = {t["model_key"] for t in todo}
    assert models == {"haiku", "gpt5mini"}


def test_unknown_stage_raises():
    runner = _import_runner()
    with pytest.raises(ValueError):
        runner.build_todo("nonsense")


# --- Auto-build chain (mocked subprocess; no API, no Drive) ---

def _fake_completed(stdout="", stderr="", returncode=0):
    return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)


def test_checkpoint_number_map_matches_checkpoint_name():
    """Every stage with a checkpoint name must also have a CP number."""
    runner = _import_runner()
    assert set(runner._CHECKPOINT_NUMBER) == set(runner._CHECKPOINT_NAME)


def test_auto_build_chain_happy_path(monkeypatch):
    """All 5 steps OK, Doc + Sheet URLs captured from stdout."""
    runner = _import_runner()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if "build_engineering_note" in " ".join(cmd):
            return _fake_completed(stdout="✓ Doc created: https://docs.google.com/document/d/eng456/edit")
        if "build_xlsx" in " ".join(cmd):
            return _fake_completed(stdout="✓ Google Sheets updated: https://docs.google.com/spreadsheets/d/wb789/edit")
        if "audit_workbook" in " ".join(cmd):
            return _fake_completed(stdout="=== Workbook audit (6 passed, 0 failed) ===")
        return _fake_completed(stdout="scored 42 rows")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arts = runner._auto_build_artifacts("stage_a", dry_run_build=False)

    # 5 steps in order: score → summarize → engineering_note → workbook → audit
    assert [a["label"] for a in arts] == [
        "score", "summarize", "engineering_note", "workbook", "audit",
    ]
    assert all(a["status"] == "OK" for a in arts)
    # URLs only on the two upload steps
    urls = {a["label"]: a["url"] for a in arts}
    assert urls["engineering_note"] == "https://docs.google.com/document/d/eng456/edit"
    assert urls["workbook"] == "https://docs.google.com/spreadsheets/d/wb789/edit"
    assert urls["score"] is None
    assert urls["audit"] is None


def test_auto_build_score_failure_skips_workbook_and_audit(monkeypatch):
    """If `analyze --score` fails:
       - workbook        SKIPPED (needs scored.csv)
       - audit           SKIPPED (needs both scored.csv + workbook)
       - engineering_note still runs (reads results.jsonl directly)."""
    runner = _import_runner()

    def fake_run(cmd, **kwargs):
        if "analyze" in " ".join(cmd) and "--score" in cmd:
            return _fake_completed(stderr="ImportError: missing openai", returncode=1)
        return _fake_completed(stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arts = runner._auto_build_artifacts("stage_a", dry_run_build=False)

    by_label = {a["label"]: a for a in arts}
    assert by_label["score"]["status"] == "FAILED"
    assert by_label["workbook"]["status"] == "SKIPPED"
    assert by_label["audit"]["status"] == "SKIPPED"
    assert "depends on score" in by_label["workbook"]["error"]
    # engineering_note doesn't depend on scored.csv, so it still runs
    assert by_label["engineering_note"]["status"] == "OK"


def test_auto_build_audit_surfaces_as_failed_when_nonzero_exit(monkeypatch):
    """Audit failure (returncode=1) is reported as FAILED but doesn't block the chain."""
    runner = _import_runner()

    def fake_run(cmd, **kwargs):
        if "audit_workbook" in " ".join(cmd):
            return _fake_completed(
                stdout="=== Workbook audit (4 passed, 2 failed) ===",
                returncode=1,
            )
        return _fake_completed(stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arts = runner._auto_build_artifacts("stage_a", dry_run_build=False)
    by_label = {a["label"]: a for a in arts}
    assert by_label["audit"]["status"] == "FAILED"
    # Other steps still ran and succeeded
    assert by_label["workbook"]["status"] == "OK"
    assert by_label["engineering_note"]["status"] == "OK"


def test_auto_build_summarize_failure_does_not_block_workbook(monkeypatch):
    """If summarize fails but score succeeds:
       - workbook         still runs (only needs scored.csv)
       - engineering_note still runs."""
    runner = _import_runner()

    def fake_run(cmd, **kwargs):
        if "analyze" in " ".join(cmd) and "--summarize" in cmd:
            return _fake_completed(stderr="pandas error", returncode=1)
        return _fake_completed(stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    arts = runner._auto_build_artifacts("stage_a", dry_run_build=False)
    by_label = {a["label"]: a for a in arts}
    assert by_label["summarize"]["status"] == "FAILED"
    assert by_label["workbook"]["status"] == "OK"
    assert by_label["engineering_note"]["status"] == "OK"


def test_auto_build_dry_run_adds_dry_run_flag(monkeypatch):
    """--dry-run-build appends --dry-run to upload steps only."""
    runner = _import_runner()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _fake_completed(stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner._auto_build_artifacts("stage_a", dry_run_build=True)

    by_step = {next(p for p in c if p.startswith("scripts.")): c for c in calls}
    # Score and summarize do NOT take --dry-run
    assert "--dry-run" not in by_step["scripts.analyze"]
    # Upload steps DO
    assert "--dry-run" in by_step["scripts.build_engineering_note"]
    assert "--dry-run" in by_step["scripts.build_xlsx"]


def test_auto_build_returns_empty_for_unknown_stage():
    runner = _import_runner()
    assert runner._auto_build_artifacts("stage_c", dry_run_build=False) == []
