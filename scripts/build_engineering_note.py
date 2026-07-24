"""
Generate an engineering note as a Google Doc, summarising the errors
encountered during model verification and how each was resolved.

Pipeline:
  outputs/model_verification_*.json (latest) +
  KNOWN_ISSUES catalog (this file)
        →  per-model: matched issues with root cause + fix
        →  HTML
        →  Drive API upload → Google Doc → URL

Uses the shared OAuth + Drive helper in src.google_drive.

Usage:
    python -m scripts.build_engineering_note
    python -m scripts.build_engineering_note --dry-run
    python -m scripts.build_engineering_note --json outputs/model_verification_<ts>.json

Why a catalog and not a one-shot text dump?
  When a new model launches with a new param-deprecation, we want this file
  to grow rather than need ad-hoc post-mortems. Add an entry to KNOWN_ISSUES
  and the eng note self-documents.
"""
from __future__ import annotations

import sys

# Windows default codec is GBK on Chinese locales — can't print '' / ''.
# Force UTF-8 so terminal logging never crashes the upload step.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import glob
import html
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src import config as cfg
from src.google_drive import upload_html_as_google_doc as upload_as_google_doc


# ---------- Known-issue catalog ----------
#
# Each entry:
#   match_pattern : substring or regex looked for inside call_error /
#                   temperature_error. Case-insensitive substring by default;
#                   set "regex": True for a real pattern.
#   root_cause    : why the API returned this error. Stable across re-runs.
#   fix           : the change applied in our codebase. Update when re-fixed.
#   touched_files : files modified by the fix; clickable in the doc.

@dataclass
class KnownIssue:
    match_pattern: str
    title: str              # short headline rendered in the doc
    severity: str           # 'fix' = code change applied; 'note' = no action needed
    root_cause: str
    fix: str
    touched_files: list[str] = field(default_factory=list)
    regex: bool = False

    def matches(self, error_text: str) -> bool:
        if not error_text:
            return False
        if self.regex:
            return re.search(self.match_pattern, error_text, re.IGNORECASE) is not None
        return self.match_pattern.lower() in error_text.lower()


KNOWN_ISSUES: list[KnownIssue] = [
    KnownIssue(
        match_pattern="`temperature` is deprecated for this model",
        title="Opus 4.7 doesn't accept `temperature`",
        severity="fix",
        root_cause=(
            "Anthropic deprecated the `temperature` parameter on Opus 4.7. "
            "Passing it — even temperature=0 — returns HTTP 400. Default sampling "
            "behaviour is used regardless."
        ),
        fix=(
            "Removed `temperature: 0` from MODELS['opus47']['params'] in "
            "src/config.py. The model now receives only `max_tokens: 300`."
        ),
        touched_files=["src/config.py"],
    ),
    KnownIssue(
        match_pattern="'reasoning_effort' does not support 'minimal'",
        title="GPT-5.5 dropped `reasoning_effort='minimal'`",
        severity="fix",
        root_cause=(
            "GPT-5.5 only accepts none | low | medium | high | xhigh for "
            "reasoning_effort. 'minimal' worked on gpt-5-mini and gpt-5 but "
            "is no longer valid on gpt-5.5."
        ),
        fix=(
            "Changed MODELS['gpt55']['params']['reasoning_effort'] from "
            "'minimal' to 'none' in src/config.py. 'none' is the cheapest "
            "available option and matches our 'no extra reasoning tokens' "
            "intent for screening tasks."
        ),
        touched_files=["src/config.py"],
    ),
    KnownIssue(
        match_pattern="does not support 0 with this model",
        title="`temperature=0` probe fails on OpenAI reasoning models (expected)",
        severity="note",
        root_cause=(
            "OpenAI reasoning models (gpt-5-mini, gpt-5, gpt-5.5) only accept "
            "the default temperature (1). The temperature=0 probe in "
            "scripts/verify_models.py is supposed to fail here — that's how the "
            "script confirms a model IS a reasoning model. Main call path is "
            "unaffected: the production params dict for these models never "
            "passes `temperature`."
        ),
        fix="No code change required.",
        touched_files=["scripts/verify_models.py"],
    ),
    KnownIssue(
        match_pattern="Stage B needs a shortlist",
        title="Stage B (stability reruns) was unimplemented at CP2",
        severity="fix",
        root_cause=(
            "scripts/run_experiment.build_todo's stage_b branch raised "
            "NotImplementedError. The Plan §5 spec — top-2 configs per task × "
            "23 briefs × 2 cheap models × 2 extra reruns — was documented but "
            "not coded. Effect: CP3 was a hard block; running "
            "`--stage stage_b` crashed before any API call."
        ),
        fix=(
            "Implemented the stage_b branch in scripts/run_experiment.py. "
            "Reads outputs/scored.csv, picks top-2 (task, config_id) per task "
            "by mean cosine (sentence) / mean F1 (keyword), ties broken by "
            "higher worst-case score (min across briefs). Emits todos for the "
            "shortlist × all 23 briefs × ['haiku','gpt5mini'] × run_id ∈ {2,3} "
            "(Stage A already covers run_id=1, so resume key skips it). Added "
            "two offline tests in tests/test_runner.py (missing-scored guard + "
            "top-2 selection logic)."
        ),
        touched_files=[
            "scripts/run_experiment.py",
            "tests/test_runner.py",
        ],
    ),
    KnownIssue(
        match_pattern="rate_limit_error",
        title="Anthropic Haiku rate-limit blowup at high concurrency (Stage A, CP2)",
        severity="fix",
        root_cause=(
            "Stage A on 2026-05-19 ran 23 briefs × all-phase configs × 2 cheap "
            "models with --concurrency 15 (single global ThreadPoolExecutor). "
            "All 15 worker threads dispatched to the Anthropic SDK at once, "
            "blowing past the org's 50 RPM cap on Haiku. Result: haiku had "
            "3,248/4,249 (77%) calls land as status=rate_limited; gpt5mini was "
            "unaffected (99.5% ok) because OpenAI's Tier-1 RPM is far higher. "
            "Three contributing causes: (1) no app-level retry — _call_anthropic "
            "did a single SDK call and let RateLimitError fall straight to "
            "_classify_error → Status.RATE_LIMITED; (2) src.config.CONCURRENCY "
            "existed but was dead code — the runner never built per-model "
            "semaphores from it; (3) SDK default max_retries=2 wasn't enough "
            "under sustained burst load."
        ),
        fix=(
            "Three-layer throttle in src/llm_client.py: "
            "(a) LLMClient.__init__ now takes model_concurrency and builds a "
            "threading.Semaphore per model_key, acquired in call() AFTER the "
            "budget kill-switches; "
            "(b) from_env() auto-wires config.CONCURRENCY (haiku lowered from "
            "15 → 5); "
            "(c) OpenAI / Anthropic SDK clients constructed with "
            "max_retries=8 (default was 2). SDK does exponential backoff with "
            "jitter and respects Retry-After headers on 429. "
            "scripts/run_experiment.py --concurrency default raised 1 → 20 "
            "since per-model semaphores are now the real throttle. "
            "Rate-limited rows in outputs/results.jsonl auto-re-run because "
            "load_done_keys only treats DONE_STATUSES as completed."
        ),
        touched_files=[
            "src/llm_client.py",
            "src/config.py",
            "scripts/run_experiment.py",
            "tests/test_budget_caps.py",
        ],
    ),
]


# ---------- Payload (pure) ----------

@dataclass
class StatusRow:
    """One row of the per-model status table."""
    model_key: str
    model_id: str
    provider: str
    status: str               # "OK" / "still failing"
    note: str                 # short — may reference "Issue N" / "Note N"


@dataclass
class GroupedSection:
    """An entry in 'Issues' (severity=fix) or 'Notes' (severity=note).
    Grouped BY catalog entry, not by model — so the same fix isn't
    repeated for every affected model."""
    number: int               # 1-based, separate sequence per severity
    title: str
    severity: str             # 'fix' / 'note'
    affected_models: list[str]
    error_excerpt: str        # one representative error string
    root_cause: str
    fix: str
    touched_files: list[str]


@dataclass
class EngNotePayload:
    title: str
    generated_at: str
    source_json: str
    overall_summary: str
    status_rows: list[StatusRow]
    issues: list[GroupedSection]   # severity=fix
    notes: list[GroupedSection]    # severity=note
    uncategorised: list[GroupedSection]  # errors not in catalog — surface, don't hide


def _all_verification_jsons() -> list[Path]:
    """All model_verification_*.json files in outputs/, oldest first."""
    paths = sorted(glob.glob(str(cfg.OUTPUTS_DIR / "model_verification_*.json")))
    if not paths:
        raise SystemExit(
            "No model_verification_*.json files in outputs/. Run "
            "`python -m scripts.verify_models` first."
        )
    return [Path(p) for p in paths]


def _latest_verification_json() -> Path:
    return _all_verification_jsons()[-1]


def _collect_runtime_errors_from_jsonl(path: Path) -> list[dict]:
    """
    Read outputs/results.jsonl and collapse non-ok rows into the same dict
    shape build_payload consumes (one record per (model_key, status, error
    prefix) combo, with the error text in `call_error`).

    Why this exists: verify_models probes can't reproduce sustained-burst
    failures like Anthropic 429s. Those only show up in real Stage A/B runs
    via outputs/results.jsonl. Aggregating them here lets the engineering
    note document runtime-only failure modes alongside verification ones.

    Returns [] if the file is missing or has no non-ok rows.
    """
    if not path.exists():
        return []

    # Group by (model_key, status, error-prefix) to dedupe — a Stage A run
    # produces thousands of identical 429s; we only need one record per
    # group to drive catalog matching.
    seen: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = r.get("status")
            if status in (None, "ok", "ok_length_violation"):
                continue
            err = r.get("error") or ""
            if not err:
                continue
            mk = r.get("model_key", "?")
            # First 120 chars is enough to bucket "rate_limit_error" vs
            # "timeout" vs other 4xx without exploding cardinality.
            key = (mk, status, err[:120])
            if key in seen:
                continue
            seen[key] = {
                "model_key": mk,
                "model_id": "(runtime)",
                "provider": "anthropic" if mk in ("haiku", "sonnet", "opus47") else "openai",
                "call_ok": False,
                "verdict": status,
                "call_error": err,
                "temperature_error": None,
            }
    return list(seen.values())


def _match_issues_for_error(error_text: str) -> list[KnownIssue]:
    if not error_text:
        return []
    return [k for k in KNOWN_ISSUES if k.matches(error_text)]


def _short(text: str, n: int = 240) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n] + " …"


def build_payload(verification_data, source_json: str) -> EngNotePayload:
    """
    Produce a deduplicated, issue-grouped payload.

    verification_data accepts either:
      - list[dict]      : one verification run
      - list[list[dict]]: multiple runs (oldest→newest); errors aggregated
                          across all of them so fixed-and-forgotten bugs
                          still show up in the doc.

    Output structure:
      • status_rows   : one row per model, with a short note pointing at
                        relevant Issue/Note numbers.
      • issues        : grouped by KnownIssue (severity='fix'), with the
                        affected_models list — same fix never printed twice.
      • notes         : grouped by KnownIssue (severity='note'); for
                        false-positive errors and "no action needed" cases.
      • uncategorised : errors that didn't match the catalog; surfaced so
                        new failure modes don't get silently hidden.
    """
    # Normalise to runs: list[list[dict]]
    if verification_data and isinstance(verification_data[0], dict):
        runs: list[list[dict]] = [verification_data]
    else:
        runs = list(verification_data)

    # Per-model: latest record + unique (field, error_text) pairs across runs
    by_model: dict[str, dict] = {}
    for run in runs:
        for rec in run:
            mk = rec.get("model_key", "?")
            entry = by_model.setdefault(mk, {"latest": rec, "errors": []})
            entry["latest"] = rec
            for field_name in ("call_error", "temperature_error"):
                err = rec.get(field_name)
                if err and (field_name, err) not in entry["errors"]:
                    entry["errors"].append((field_name, err))

    # Group by catalog entry: KnownIssue id -> {affected_models, representative error}
    # id is the catalog object itself (use id() as key).
    catalog_groups: dict[int, dict] = {}
    uncategorised_groups: dict[str, dict] = {}

    for mk, entry in by_model.items():
        for field_name, err in entry["errors"]:
            matched = _match_issues_for_error(err)
            if matched:
                for k in matched:
                    g = catalog_groups.setdefault(id(k), {
                        "issue": k,
                        "affected_models": [],
                        "error_excerpt": _short(err),
                    })
                    if mk not in g["affected_models"]:
                        g["affected_models"].append(mk)
            else:
                # Bucket by the (short) error string so repeated unknown errors
                # collapse into one group.
                key = _short(err, 120)
                u = uncategorised_groups.setdefault(key, {
                    "title": "UNCATEGORISED — needs catalog entry",
                    "affected_models": [],
                    "error_excerpt": _short(err),
                })
                if mk not in u["affected_models"]:
                    u["affected_models"].append(mk)

    # Convert to numbered GroupedSection lists, separately for fix vs note.
    issues: list[GroupedSection] = []
    notes: list[GroupedSection] = []
    for grp in catalog_groups.values():
        k: KnownIssue = grp["issue"]
        bucket = issues if k.severity == "fix" else notes
        bucket.append(GroupedSection(
            number=len(bucket) + 1,
            title=k.title,
            severity=k.severity,
            affected_models=grp["affected_models"],
            error_excerpt=grp["error_excerpt"],
            root_cause=k.root_cause,
            fix=k.fix,
            touched_files=list(k.touched_files),
        ))

    uncategorised: list[GroupedSection] = []
    for grp in uncategorised_groups.values():
        uncategorised.append(GroupedSection(
            number=len(uncategorised) + 1,
            title=grp["title"],
            severity="uncategorised",
            affected_models=grp["affected_models"],
            error_excerpt=grp["error_excerpt"],
            root_cause="Not in KNOWN_ISSUES catalog.",
            fix=("Add an entry to KNOWN_ISSUES in "
                 "scripts/build_engineering_note.py once you've diagnosed it."),
            touched_files=[],
        ))

    # Per-model status rows, with references to which Issue/Note they belong to
    def _refs_for_model(mk: str) -> str:
        refs = []
        for sec in issues:
            if mk in sec.affected_models:
                refs.append(f"Issue {sec.number}")
        for sec in notes:
            if mk in sec.affected_models:
                refs.append(f"Note {sec.number}")
        for sec in uncategorised:
            if mk in sec.affected_models:
                refs.append(f"Uncategorised {sec.number}")
        return "see " + ", ".join(refs) if refs else "clean"

    status_rows: list[StatusRow] = []
    for mk, entry in by_model.items():
        rec = entry["latest"]
        status_rows.append(StatusRow(
            model_key=mk,
            model_id=rec.get("model_id", "?"),
            provider=rec.get("provider", "?"),
            status="OK" if rec.get("call_ok") else "still failing",
            note=_refs_for_model(mk),
        ))

    n_ok = sum(1 for r in status_rows if r.status == "OK")
    n_total = len(status_rows)
    summary = (
        f"{n_ok}/{n_total} models passing in the most recent run. "
        f"{len(issues)} required code change(s); {len(notes)} false-positive note(s); "
        f"{len(uncategorised)} uncategorised. "
        f"Aggregated over {len(runs)} verification run(s)."
    )

    return EngNotePayload(
        title=f"Engineering Notes — Model Verification — {datetime.now():%Y-%m-%d}",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        source_json=source_json,
        overall_summary=summary,
        status_rows=status_rows,
        issues=issues,
        notes=notes,
        uncategorised=uncategorised,
    )


# ---------- HTML rendering ----------

def render_html(p: EngNotePayload) -> str:
    def esc(s: object) -> str:
        return html.escape(str(s), quote=False)

    # --- Status table ---
    def _status_color(s: str) -> str:
        return "#1b5e20" if s == "OK" else "#b71c1c"

    status_rows_html = []
    for r in p.status_rows:
        badge = "[OK] OK" if r.status == "OK" else "" + r.status
        status_rows_html.append(
            "<tr>"
            f"<td><code>{esc(r.model_key)}</code></td>"
            f"<td>{esc(r.provider)}</td>"
            f"<td><code>{esc(r.model_id)}</code></td>"
            f"<td style='color:{_status_color(r.status)}'>{esc(badge)}</td>"
            f"<td>{esc(r.note)}</td>"
            "</tr>"
        )
    status_table = (
        "<table border='1' cellpadding='6' cellspacing='0' "
        "style='border-collapse:collapse'>"
        "<thead><tr>"
        "<th>Model</th><th>Provider</th><th>Model ID</th>"
        "<th>Status</th><th>Note</th>"
        "</tr></thead>"
        f"<tbody>{''.join(status_rows_html)}</tbody></table>"
    )

    # --- Issue / Note / Uncategorised sections ---
    def _section_block(label: str, sec: GroupedSection, border_color: str) -> str:
        affected = ", ".join(f"<code>{esc(m)}</code>" for m in sec.affected_models)
        files = (
            "Files: " + ", ".join(f"<code>{esc(f)}</code>" for f in sec.touched_files)
            if sec.touched_files else ""
        )
        sub = " · ".join(part for part in [f"Affected: {affected}", files] if part)
        return (
            f"<div style='border-left:3px solid {border_color};"
            f"padding:6px 12px;margin:10px 0;background:#fafafa'>"
            f"<h4 style='margin:0 0 4px'>{label} {sec.number}: {esc(sec.title)}</h4>"
            f"<p style='color:#555;font-size:12px;margin:0 0 6px'>{sub}</p>"
            f"<p style='margin:0 0 4px'><strong>Error:</strong></p>"
            f"<pre style='background:#fff3e0;padding:6px;white-space:pre-wrap;"
            f"font-size:12px;margin:0 0 6px'>{esc(sec.error_excerpt)}</pre>"
            f"<p style='margin:0 0 4px'><strong>Root cause:</strong> {esc(sec.root_cause)}</p>"
            f"<p style='margin:0 0 4px'><strong>Fix:</strong> {esc(sec.fix)}</p>"
            "</div>"
        )

    issues_html = (
        "".join(_section_block("Issue", s, "#1976d2") for s in p.issues)
        if p.issues else "<p style='color:#777'><i>No code changes required.</i></p>"
    )
    notes_html = (
        "".join(_section_block("Note", s, "#9e9e9e") for s in p.notes)
        if p.notes else ""
    )
    uncat_html = (
        "".join(_section_block("Uncategorised", s, "#d32f2f") for s in p.uncategorised)
        if p.uncategorised else ""
    )

    notes_section = (
        f"<h2>Notes (no action needed)</h2>{notes_html}"
        if p.notes else ""
    )
    uncat_section = (
        f"<h2 style='color:#d32f2f'>Uncategorised — needs follow-up</h2>"
        "<p style='color:#555;font-size:13px'>These errors weren't in the catalog. "
        "Diagnose, then add a <code>KnownIssue</code> entry to "
        "<code>scripts/build_engineering_note.py</code>.</p>"
        f"{uncat_html}"
        if p.uncategorised else ""
    )

    return f"""<html><head><meta charset="utf-8"></head><body>
<h1>{esc(p.title)}</h1>
<p style="color:#777;font-size:12px">
Generated at <code>{esc(p.generated_at)}</code> from <code>{esc(p.source_json)}</code>
</p>

<h2 style="background:#e3f2fd;padding:6px 10px;border-left:4px solid #1976d2">Summary</h2>
<p>{esc(p.overall_summary)}</p>

<h2>Status</h2>
{status_table}

<h2>Issues encountered &amp; fixed</h2>
{issues_html}

{notes_section}

{uncat_section}

<hr>
<p style="color:#777;font-size:11px">
Source: <code>{esc(p.source_json)}</code> · Regenerate by running
<code>python -m scripts.build_engineering_note</code> after a new
<code>verify_models</code> probe.
</p>
</body></html>"""


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default=None,
                    help="Path to a single model_verification_*.json. "
                         "Default: aggregate across ALL runs in outputs/ so "
                         "fixed-and-forgotten bugs still get documented.")
    ap.add_argument("--latest-only", action="store_true",
                    help="Use only the most recent verification JSON. "
                         "(Hides any errors fixed in earlier runs.)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write report.html locally; don't upload to Drive.")
    ap.add_argument("--folder-id", type=str, default=None,
                    help="(Optional) Drive folder ID to drop the doc into.")
    args = ap.parse_args()

    if args.json:
        srcs = [Path(args.json)]
    elif args.latest_only:
        srcs = [_latest_verification_json()]
    else:
        srcs = _all_verification_jsons()

    runs: list[list[dict]] = []
    # Prepend runtime errors from outputs/results.jsonl so verification runs
    # (newer) still win for the "latest" status, but their errors are still
    # picked up for catalog matching and Issues grouping.
    runtime_errors = _collect_runtime_errors_from_jsonl(cfg.OUTPUTS_DIR / "results.jsonl")
    if runtime_errors:
        runs.append(runtime_errors)
        print(f"Aggregated {len(runtime_errors)} runtime error group(s) from results.jsonl")
    for s in srcs:
        if not s.exists():
            raise SystemExit(f"File not found: {s}")
        with open(s, encoding="utf-8") as f:
            runs.append(json.load(f))
    source_label = (
        str(srcs[0]) if len(srcs) == 1
        else f"{len(srcs)} verification runs ({srcs[0].name} → {srcs[-1].name})"
    )
    if runtime_errors:
        source_label += " + outputs/results.jsonl runtime errors"

    payload = build_payload(runs, source_label)
    html_str = render_html(payload)

    if args.dry_run:
        out = cfg.RESULTS_DIR / f"engineering_note_{datetime.now():%Y%m%d_%H%M%S}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html_str, encoding="utf-8")
        print(f"[dry-run] wrote: {out}")
        return

    url = upload_as_google_doc(html_str, payload.title, args.folder_id)
    print(f"\n[OK] Engineering-note Google Doc created: {url}\n")


if __name__ == "__main__":
    main()
