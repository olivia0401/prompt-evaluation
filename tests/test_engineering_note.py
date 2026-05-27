"""Offline tests for scripts/build_engineering_note. No Google API calls."""
import json

from scripts.build_engineering_note import (
    build_payload,
    render_html,
    KNOWN_ISSUES,
    _match_issues_for_error,
    _collect_runtime_errors_from_jsonl,
)


def test_catalog_matches_opus_temperature_error():
    err = ("BadRequestError: Error code: 400 - {'type': 'error', 'error': "
           "{'type': 'invalid_request_error', 'message': "
           "'`temperature` is deprecated for this model.'}}")
    hits = _match_issues_for_error(err)
    assert len(hits) == 1
    assert hits[0].severity == "fix"


def test_catalog_matches_gpt55_reasoning_effort_error():
    err = ("BadRequestError: Error code: 400 - {'error': {'message': "
           "\"Unsupported value: 'reasoning_effort' does not support "
           "'minimal' with this model.\"}}")
    hits = _match_issues_for_error(err)
    assert len(hits) == 1
    assert hits[0].severity == "fix"


def test_catalog_matches_openai_temperature_zero_as_note():
    """temperature=0 probe failure on reasoning models is a NOTE, not a fix."""
    err = ("BadRequestError: Error code: 400 - {'error': {'message': "
           "\"Unsupported value: 'temperature' does not support 0 with "
           "this model.\"}}")
    hits = _match_issues_for_error(err)
    assert any(h.severity == "note" for h in hits)


def test_payload_groups_by_issue_not_by_model():
    """Three OpenAI reasoning models all triggering the same temperature=0
    note must produce ONE Note section listing all three, not three duplicates."""
    sample = []
    for mk in ("gpt5mini", "gpt5", "gpt55"):
        sample.append({
            "model_key": mk, "model_id": f"{mk}-id", "provider": "openai",
            "call_ok": True, "verdict": "OK",
            "call_error": None,
            "temperature_error": "BadRequestError: 'temperature' does not support 0 with this model.",
        })
    p = build_payload(sample, "test.json")
    assert len(p.notes) == 1, f"expected one merged note, got {len(p.notes)}"
    assert set(p.notes[0].affected_models) == {"gpt5mini", "gpt5", "gpt55"}
    assert p.issues == []  # no fix required


def test_payload_status_table_references_issue_numbers():
    """opus47 row's note should say 'see Issue 1' since its fix is a fix-class issue."""
    sample = [
        {"model_key": "opus47", "model_id": "claude-opus-4-7", "provider": "anthropic",
         "call_ok": True, "verdict": "OK",
         "call_error": "BadRequestError: `temperature` is deprecated for this model.",
         "temperature_error": None},
        {"model_key": "haiku", "model_id": "claude-haiku", "provider": "anthropic",
         "call_ok": True, "verdict": "OK",
         "call_error": None, "temperature_error": None},
    ]
    p = build_payload(sample, "test.json")
    opus_row = next(r for r in p.status_rows if r.model_key == "opus47")
    haiku_row = next(r for r in p.status_rows if r.model_key == "haiku")
    assert "Issue" in opus_row.note
    assert haiku_row.note == "clean"


def test_aggregation_across_runs_preserves_fixed_errors():
    """A bug in run-1 fixed in run-2 should still produce an Issue section."""
    run1 = [
        {"model_key": "opus47", "model_id": "claude-opus-4-7", "provider": "anthropic",
         "call_ok": False, "verdict": "FAIL",
         "call_error": "BadRequestError: `temperature` is deprecated for this model.",
         "temperature_error": None},
    ]
    run2 = [
        {"model_key": "opus47", "model_id": "claude-opus-4-7", "provider": "anthropic",
         "call_ok": True, "verdict": "OK",
         "call_error": None, "temperature_error": None},
    ]
    p = build_payload([run1, run2], "history.json")
    opus_row = next(r for r in p.status_rows if r.model_key == "opus47")
    assert opus_row.status == "OK"  # latest run wins
    assert len(p.issues) == 1
    assert "opus47" in p.issues[0].affected_models


def test_uncategorised_error_surfaces_for_followup():
    sample = [
        {"model_key": "mystery", "model_id": "x", "provider": "openai",
         "call_ok": False, "verdict": "FAIL",
         "call_error": "Some never-before-seen error",
         "temperature_error": None},
    ]
    p = build_payload(sample, "test.json")
    assert len(p.uncategorised) == 1
    assert "mystery" in p.uncategorised[0].affected_models


def test_render_html_has_simplified_structure():
    sample = [
        {"model_key": "haiku", "model_id": "x", "provider": "anthropic",
         "call_ok": True, "verdict": "OK", "call_error": None,
         "temperature_error": None},
        {"model_key": "opus47", "model_id": "y", "provider": "anthropic",
         "call_ok": True, "verdict": "OK",
         "call_error": "BadRequestError: `temperature` is deprecated for this model.",
         "temperature_error": None},
    ]
    p = build_payload(sample, "test.json")
    out = render_html(p)
    assert out.startswith("<html>")
    assert "<h2>Status</h2>" in out
    assert "Issues encountered" in out
    # Status table must exist with both models
    assert "haiku" in out and "opus47" in out
    # No more "Per-model log" or repeated per-model blocks
    assert "Per-model log" not in out


def test_render_html_escapes_dangerous_chars():
    sample = [
        {"model_key": "<script>evil()</script>", "model_id": "x", "provider": "openai",
         "call_ok": False, "verdict": "FAIL",
         "call_error": "<img src=x onerror=evil()>",
         "temperature_error": None},
    ]
    p = build_payload(sample, "test.json")
    out = render_html(p)
    assert "<script>evil()" not in out
    assert "<img src=x onerror=evil" not in out
    assert "&lt;script&gt;" in out


def test_catalog_entries_all_have_severity():
    for k in KNOWN_ISSUES:
        assert k.severity in ("fix", "note"), \
            f"{k.title}: severity must be 'fix' or 'note', got {k.severity!r}"
        assert k.title, f"missing title for pattern {k.match_pattern!r}"


def test_no_errors_means_no_issues_no_notes():
    sample = [
        {"model_key": "haiku", "model_id": "x", "provider": "anthropic",
         "call_ok": True, "verdict": "OK", "call_error": None,
         "temperature_error": None},
    ]
    p = build_payload(sample, "test.json")
    assert p.issues == []
    assert p.notes == []
    assert p.uncategorised == []
    assert p.status_rows[0].note == "clean"


# --- Haiku rate-limit catalog + runtime-error wiring (CP2, 2026-05-19) ---

def test_catalog_matches_haiku_rate_limit_error():
    """The 429 Anthropic error string from CP2 Stage A must match the catalog."""
    err = ("Error code: 429 - {'type': 'error', 'error': "
           "{'type': 'rate_limit_error', 'message': \"This request would "
           "exceed your organization's rate limit of 50 requests per minute\"}}")
    hits = _match_issues_for_error(err)
    assert len(hits) == 1, f"expected exactly one match, got {[h.title for h in hits]}"
    assert hits[0].severity == "fix"
    assert "src/llm_client.py" in hits[0].touched_files
    assert "src/config.py" in hits[0].touched_files


def test_collect_runtime_errors_groups_by_model_status_error(tmp_path):
    """Thousands of identical 429s collapse to one record per (model, status, err prefix)."""
    p = tmp_path / "results.jsonl"
    lines = []
    # 3 distinct rate_limited haiku rows (same error -> 1 group)
    for i in range(3):
        lines.append(json.dumps({
            "model_key": "haiku", "status": "rate_limited",
            "error": "Error code: 429 - rate_limit_error: 50 RPM",
        }))
    # 1 ok haiku row -> dropped
    lines.append(json.dumps({
        "model_key": "haiku", "status": "ok", "error": None,
    }))
    # 1 timeout gpt5mini row -> different group
    lines.append(json.dumps({
        "model_key": "gpt5mini", "status": "timeout",
        "error": "Request timed out after 60s",
    }))
    p.write_text("\n".join(lines), encoding="utf-8")

    recs = _collect_runtime_errors_from_jsonl(p)
    # 2 groups: (haiku, rate_limited, ...) and (gpt5mini, timeout, ...)
    assert len(recs) == 2
    by_model = {r["model_key"]: r for r in recs}
    assert by_model["haiku"]["verdict"] == "rate_limited"
    assert by_model["haiku"]["provider"] == "anthropic"
    assert by_model["haiku"]["call_ok"] is False
    assert by_model["gpt5mini"]["verdict"] == "timeout"
    assert by_model["gpt5mini"]["provider"] == "openai"


def test_collect_runtime_errors_missing_file_returns_empty(tmp_path):
    """No results.jsonl -> empty list, not a crash."""
    assert _collect_runtime_errors_from_jsonl(tmp_path / "nope.jsonl") == []


def test_runtime_errors_feed_into_payload_as_fix_issue():
    """A runtime rate_limited record drives the catalog match end-to-end."""
    runtime_run = [{
        "model_key": "haiku", "model_id": "(runtime)", "provider": "anthropic",
        "call_ok": False, "verdict": "rate_limited",
        "call_error": ("Error code: 429 - {'type': 'error', 'error': "
                       "{'type': 'rate_limit_error', 'message': 'over RPM'}}"),
        "temperature_error": None,
    }]
    verify_run = [{
        "model_key": "haiku", "model_id": "claude-haiku-4-5", "provider": "anthropic",
        "call_ok": True, "verdict": "OK", "call_error": None,
        "temperature_error": None,
    }]
    # Order matches main(): runtime errors first, verification runs after,
    # so the latest status comes from the (passing) verification run.
    p = build_payload([runtime_run, verify_run], "test")
    haiku_row = next(r for r in p.status_rows if r.model_key == "haiku")
    assert haiku_row.status == "OK"  # latest verify wins
    assert any("rate-limit" in i.title.lower() for i in p.issues), \
        f"rate-limit issue not in {[i.title for i in p.issues]}"
