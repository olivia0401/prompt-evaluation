"""Unit tests for scripts/audit_data."""
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def audit_env(tmp_path, monkeypatch):
    """Point cfg.OUTPUTS_DIR / RESULTS_DIR / BRIEFS_FILE at a temp workspace.

    Returns a dict of helpers for writing fixture files.
    """
    outputs = tmp_path / "outputs"
    results = tmp_path / "Results"
    outputs.mkdir()
    results.mkdir()
    briefs_file = tmp_path / "briefs.yml"

    from src import config as cfg
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(cfg, "RESULTS_DIR", results)
    monkeypatch.setattr(cfg, "BRIEFS_FILE", briefs_file)
    monkeypatch.setattr(cfg, "PROJECT_ROOT", tmp_path)

    return {
        "outputs": outputs,
        "results": results,
        "briefs_file": briefs_file,
        "root": tmp_path,
    }


def _write_briefs(path: Path, n: int = 23) -> list[dict]:
    briefs = [{"current_name": f"Brief{i:02d}",
               "concept_relevant": "gt sentence",
               "keywords": ["k1", "k2"]}
              for i in range(n)]
    path.write_text(yaml.safe_dump(briefs), encoding="utf-8")
    return briefs


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )


def _make_call(brief_id="Brief00", task="concept_relevant",
               config_id="A:_full_brief", model_key="haiku",
               run_id=1, status="ok", raw_response="abc", cost_usd=0.001):
    return {
        "brief_id": brief_id, "task": task, "config_id": config_id,
        "model_key": model_key, "run_id": run_id, "status": status,
        "raw_response": raw_response, "cost_usd": cost_usd,
        "prompt": "p", "input_tokens": 100, "output_tokens": 50,
    }


def test_full_coverage_passes_b1_b2_b3(audit_env):
    """23 briefs × 9 tasks × 1 config × 1 model — completeness checks should pass."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = []
    tasks = ["concept_relevant", "position_relevant", "emotion_relevant",
             "function_relevant", "benefit_relevant", "category_relevant",
             "feature_relevant", "context_relevant", "keywords"]
    for i in range(23):
        for t in tasks:
            rows.append(_make_call(brief_id=f"Brief{i:02d}", task=t))
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["B1"].status == "PASS"
    assert by_code["B2"].status == "PASS"
    assert by_code["B3"].status == "PASS"


def test_partial_brief_coverage_fails_b1(audit_env):
    """Only 3 briefs of 23 — B1 should FAIL with missing list."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = [_make_call(brief_id=f"Brief{i:02d}") for i in range(3)]
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["B1"].status == "FAIL"
    assert "3/23" in by_code["B1"].detail or "3 / 23" in by_code["B1"].detail


def test_uneven_cell_coverage_fails_b3(audit_env):
    """Some cells have 3 briefs, others 1 — B3 flags inconsistency."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = []
    # 3 briefs for full_brief, 1 brief for a single-field config
    for i in range(3):
        rows.append(_make_call(brief_id=f"Brief{i:02d}", config_id="A:_full_brief"))
    rows.append(_make_call(brief_id="Brief00", config_id="A:product"))
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["B3"].status == "FAIL"


def test_rate_limit_leftover_warns_c2(audit_env):
    """rate_limited rows still in JSONL — C2 surfaces them (WARN, not FAIL)."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = [_make_call(status="ok"), _make_call(status="rate_limited")]
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["C2"].status == "WARN"
    assert "rate_limited" in by_code["C2"].detail


def test_per_model_keysuccess_fails_c1(audit_env):
    """20 distinct Haiku keys stuck on rate_limited + 1 ok key = 1/21 = 4.8% success.

    C1 now measures per-unique-key eventual success rather than per-row, so
    retry trails don't inflate the failure count. This test exercises the
    failure path by ensuring each rate_limited row is a DIFFERENT key.
    """
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = []
    for i in range(20):
        # Each row gets a distinct brief_id → distinct resume key → still stuck.
        rows.append(_make_call(
            brief_id=f"Brief{i:02d}",
            model_key="haiku", status="rate_limited",
        ))
    rows.append(_make_call(brief_id="Brief99", model_key="haiku", status="ok"))
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["C1"].status == "FAIL"
    assert "1/21" in by_code["C1"].detail or "haiku" in by_code["C1"].detail.lower()


def test_per_model_keysuccess_passes_after_retry_c1(audit_env):
    """20 keys all rate_limited THEN later succeeded = 100% per-key success → PASS.

    This is the realistic "we hit rate limits but resume eventually got us"
    pattern. The old per-row C1 would have failed this (20/40 = 50%); the
    new per-key C1 correctly recognises every key as done.
    """
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = []
    for i in range(20):
        # First attempt rate_limited, second attempt ok — same key both times.
        rows.append(_make_call(brief_id=f"Brief{i:02d}",
                               model_key="haiku", status="rate_limited"))
        rows.append(_make_call(brief_id=f"Brief{i:02d}",
                               model_key="haiku", status="ok"))
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["C1"].status == "PASS"


def test_retry_then_ok_passes_c4(audit_env):
    """A key with rate_limited THEN ok is fine — resume pattern, not a duplicate bug."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = [_make_call(status="rate_limited"), _make_call(status="ok")]
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    # All unique keys (just one here) eventually succeeded.
    assert by_code["C4"].status == "PASS"


def test_all_rate_limited_warns_c4(audit_env):
    """A key that never reached ok — C4 WARNs (rerun needed)."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = [_make_call(status="rate_limited"), _make_call(status="rate_limited")]
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["C4"].status == "WARN"
    assert "1 keys never reached ok" in by_code["C4"].detail


def test_metric_out_of_range_fails_e2(audit_env):
    """cosine = 1.5 is out of [0,1] — E2 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call()])
    scored = pd.DataFrame([{
        "brief_id": "Brief00", "task": "concept_relevant",
        "config_id": "A:_full_brief", "model_key": "haiku", "run_id": 1,
        "status": "ok", "cosine": 1.5, "rouge_l": 0.1, "length_compliant": True,
    }])
    scored.to_csv(audit_env["outputs"] / "scored.csv", index=False)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    by_code = {r.code: r for r in results}

    assert by_code["E2"].status == "FAIL"
    assert "cosine" in by_code["E2"].detail


def test_clean_data_exits_zero_except_b4(audit_env):
    """Clean dataset (23 briefs × 9 tasks × 1 config × 2 models).

    B4 (Stage A coverage = 95% of 142-config matrix) cannot pass with a 1-config
    fixture, so we exclude B4 from the must-pass set.
    """
    _write_briefs(audit_env["briefs_file"], n=23)
    # G5 prerequisite: provide a complete .gitignore
    (audit_env["root"] / ".gitignore").write_text(
        ".env\ncredentials.json\ntoken.json\noutputs/\nembedding_cache.jsonl\n",
        encoding="utf-8",
    )
    tasks = ["concept_relevant", "position_relevant", "emotion_relevant",
             "function_relevant", "benefit_relevant", "category_relevant",
             "feature_relevant", "context_relevant", "keywords"]
    rows = []
    for i in range(23):
        for t in tasks:
            for m in ("haiku", "gpt5mini"):
                rows.append(_make_call(brief_id=f"Brief{i:02d}", task=t, model_key=m))
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    results = audit_data.run_all_checks()
    unexpected_fails = [r for r in results if r.status == "FAIL" and r.code != "B4"]
    assert not unexpected_fails, unexpected_fails


# ============ Group G — security ============

def test_hardcoded_api_key_in_src_fails_g2(audit_env):
    """A `sk-ant-...` literal in src/foo.py — G2 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call()])
    src_dir = audit_env["root"] / "src"
    src_dir.mkdir()
    (src_dir / "leaky.py").write_text(
        'API_KEY = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz12345"\n',
        encoding="utf-8",
    )

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["G2"].status == "FAIL"
    assert "leaky.py" in by_code["G2"].detail


def test_api_key_in_raw_response_fails_g3(audit_env):
    """Model echoed an API key in its output — G3 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = [_make_call(raw_response="Here it is: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234")]
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["G3"].status == "FAIL"


def test_pii_email_in_raw_response_fails_g4(audit_env):
    """Email regex hits in raw_response — G4 FAILs (email is hard signal)."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = [_make_call(raw_response="Contact us at john.doe@example.com for more.")]
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["G4"].status == "FAIL"
    assert "email=1" in by_code["G4"].detail


def test_gitignore_missing_pattern_fails_g5(audit_env):
    """.gitignore without 'outputs/' — G5 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call()])
    (audit_env["root"] / ".gitignore").write_text(".env\n", encoding="utf-8")

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["G5"].status == "FAIL"
    assert "outputs/" in by_code["G5"].detail


def test_gitignore_complete_passes_g5(audit_env):
    """All required patterns present — G5 PASSes."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call()])
    (audit_env["root"] / ".gitignore").write_text(
        ".env\ncredentials.json\ntoken.json\noutputs/\nembedding_cache.jsonl\n",
        encoding="utf-8",
    )

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["G5"].status == "PASS"


# ============ Group H — output quality ============

def test_high_refusal_rate_fails_h1(audit_env):
    """20 rows, 5 refusals (25%) — above 5% ceiling, H1 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    rows = ([_make_call(raw_response="I cannot help with that.") for _ in range(5)]
            + [_make_call(raw_response="A normal answer about the brand.") for _ in range(15)])
    _write_jsonl(audit_env["outputs"] / "results.jsonl", rows)

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["H1"].status == "FAIL"
    assert "25.0%" in by_code["H1"].detail


def test_keyword_count_compliance_fails_h3(audit_env):
    """Keyword task returns 5 keywords instead of 10 — H3 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call(task="keywords")])
    # Hand-craft scored.csv: 5 keyword rows, only 1 has 10 keywords parsed
    scored = pd.DataFrame([
        {"brief_id": f"B{i}", "task": "keywords",
         "config_id": "A:_full_brief", "model_key": "haiku", "run_id": 1,
         "status": "ok",
         "prediction_parsed": json.dumps(["k1", "k2", "k3"]),  # only 3
         "cosine": None, "f1": 0.3, "length_compliant": True}
        for i in range(5)
    ])
    scored.to_csv(audit_env["outputs"] / "scored.csv", index=False)

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["H3"].status == "FAIL"
    assert "0.0%" in by_code["H3"].detail


# ============ Group I — stability ============

def test_stage_b_high_std_fails_i1(audit_env):
    """Cosine std > 0.05 across runs — I1 FAILs."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call()])
    # 1 brief × 1 task × 1 config × 1 model × 3 runs, cosine spread = 0.3
    scored = pd.DataFrame([
        {"brief_id": "Brief00", "task": "concept_relevant",
         "config_id": "A:product", "model_key": "haiku", "run_id": ri,
         "status": "ok", "cosine": c, "f1": None, "length_compliant": True}
        for ri, c in zip([1, 2, 3], [0.3, 0.6, 0.9])
    ])
    scored.to_csv(audit_env["outputs"] / "scored.csv", index=False)

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["I1"].status == "FAIL"


def test_stable_reruns_pass_i1(audit_env):
    """Reruns within 0.01 of each other — I1 PASSes."""
    _write_briefs(audit_env["briefs_file"], n=23)
    _write_jsonl(audit_env["outputs"] / "results.jsonl", [_make_call()])
    scored = pd.DataFrame([
        {"brief_id": "Brief00", "task": "concept_relevant",
         "config_id": "A:product", "model_key": "haiku", "run_id": ri,
         "status": "ok", "cosine": c, "f1": None, "length_compliant": True}
        for ri, c in zip([1, 2, 3], [0.50, 0.51, 0.49])
    ])
    scored.to_csv(audit_env["outputs"] / "scored.csv", index=False)

    from scripts import audit_data
    by_code = {r.code: r for r in audit_data.run_all_checks()}
    assert by_code["I1"].status == "PASS"
