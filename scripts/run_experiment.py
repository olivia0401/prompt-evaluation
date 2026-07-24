"""
Resumable experiment runner.

Primary storage: outputs/results.jsonl (one CallResult per line).
Resume key: (brief_id, task, config_id, model_key, run_id).
On each invocation the runner reads existing JSONL, skips already-done keys,
and appends new results. Failed / non-DONE entries are re-attempted on rerun.
"""
import argparse
import json
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

# Windows default codec is GBK on Chinese locales — can't print '' / ''.
# Force UTF-8 so checkpoint banners (and any subprocess output we re-print)
# never crash mid-run.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src import config as cfg
from src.llm_client import LLMClient, Status, DONE_STATUSES
from src.utils import append_jsonl


OUT_JSONL = cfg.OUTPUTS_DIR / "results.jsonl"
SUMMARY_CSV = cfg.OUTPUTS_DIR / "summary_by_config.csv"

# Phase 4 models: medium + premium, run on top-1 cheap-screen winner per task.
# Cheap is already in the data from Stage A — we don't re-run those.
# Running medium AND premium gives a 3-tier "quality ladder" instead of a
# 2-tier "cheap vs premium" jump — exposes the full quality curve:
#   "what quality can we get with the best models" — for each tier, what's
#   the ceiling on the top configs, and is medium → premium worth the price?
PHASE4_MODELS = ["sonnet", "gpt5", "opus47", "gpt55"]


def load_done_keys(path: Path) -> set:
    """Read JSONL, return set of (brief_id, task, config_id, model_key, run_id) that completed OK."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in DONE_STATUSES:
                done.add((
                    r["brief_id"],
                    r["task"],
                    r["config_id"],
                    r["model_key"],
                    r["run_id"],
                ))
    return done


def _load_top_configs_per_task(summary_path: Path = SUMMARY_CSV) -> dict[str, dict]:
    """
    Read summary_by_config.csv (built by analyze.py --summarize) and pick the
    top-1 (task -> {config_id, prompt_version}) by mean_cosine.

    Tie-break: highest worst_cosine (more robust across briefs), then alphabetical
    config_id for determinism.

    Returns: {task: {"config_id": str, "prompt_version": str}}
    Raises SystemExit if the CSV is missing — Phase 4 cannot proceed without it.
    """
    import pandas as pd

    if not summary_path.exists():
        raise SystemExit(
            f"Missing {summary_path}. Phase 4 (premium re-run) needs Stage A/B "
            f"results first. Run:\n"
            f"  python -m scripts.run_experiment --stage stage_a\n"
            f"  python -m scripts.analyze --score\n"
            f"  python -m scripts.analyze --summarize"
        )

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"{summary_path} is empty — no configs to promote.")

    # Use cheap-model rows (Stage A) as the basis for picking winners, since
    # Phase 4's whole point is "do the best cheap-screen winners hold up on
    # premium models?".
    cheap_keys = {k for k, v in cfg.MODELS.items() if v["tier"] == "cheap"}
    cheap = df[df["model_key"].isin(cheap_keys)]
    if cheap.empty:
        cheap = df  # fallback — no cheap-tier rows present, use whatever exists

    # Average across cheap models so the pick isn't biased by a single provider.
    agg = (
        cheap.groupby(["task", "config_id"])
        .agg(mean_cosine=("mean_cosine", "mean"), worst_cosine=("worst_cosine", "mean"))
        .reset_index()
    )
    agg = agg.sort_values(
        ["task", "mean_cosine", "worst_cosine", "config_id"],
        ascending=[True, False, False, True],
    )
    top = agg.groupby("task").head(1)

    out: dict[str, dict] = {}
    for _, row in top.iterrows():
        cid = str(row["config_id"])
        # config_id format: "<version>:<fields>" e.g. "A:_full_brief" or "B:product+audience"
        version, _, _ = cid.partition(":")
        out[str(row["task"])] = {"config_id": cid, "prompt_version": version or "A"}
    return out


def _config_id_to_fields(config_id: str):
    """Inverse of Config.config_id. Returns either a sentinel string or a tuple of fields."""
    from src.prompt_builder import FULL_BRIEF, PROMPT_IMPLIED, TRIVIAL
    _, _, rhs = config_id.partition(":")
    if rhs in {FULL_BRIEF, PROMPT_IMPLIED, TRIVIAL, "_baseline"}:
        # "_baseline" was the legacy name for prompt-implied; treat the same.
        return PROMPT_IMPLIED if rhs == "_baseline" else rhs
    return tuple(sorted(rhs.split("+")))


def build_todo(stage: str) -> list[dict]:
    """
    Build the full job list for one stage.

    Returns list of dicts: {brief_id, task, config_id, model_key, run_id, prompt}

    Stage definitions:
      phase0  : PHASE0_BRIEF_NAMES (3 briefs) x 17 configs x 2 cheap models x 1 run
      stage_a : 23 briefs x all-phase configs x 2 cheap models x 1 run
      stage_b : caller passes shortlisted configs via a separate flow (NotImplementedError)
      phase4  : top-1 config per task (from summary_by_config.csv) x curated brief
                subset x 2 PREMIUM models. Bound to ≤£1 by config.BUDGET_CAP.
      stage_c : same, for medium models x 3 runs (deferred)

    Resume / dedup happens at the runner level (load_done_keys filters out
    completed entries before any API call).
    """
    from src.prompt_builder import (
        FULL_BRIEF,
        PROMPT_IMPLIED,
        TRIVIAL,
        PHASE0_BRIEF_NAMES,
        brief_id,
        build_prompt,
        list_configs_for_stage,
        list_phase0_configs,
        load_briefs,
        load_prompts,
    )

    SENTINELS = {FULL_BRIEF, PROMPT_IMPLIED, TRIVIAL}

    all_briefs = load_briefs()
    templates = load_prompts()

    if stage == "phase0":
        # Hand-picked briefs from prompt_builder.PHASE0_BRIEF_NAMES
        wanted = set(PHASE0_BRIEF_NAMES)
        briefs = [b for b in all_briefs if brief_id(b) in wanted]
        missing = wanted - {brief_id(b) for b in briefs}
        if missing:
            raise SystemExit(f"PHASE0_BRIEF_NAMES not found in briefs.yml: {missing}")
        configs = list_phase0_configs()
        models = ["haiku", "gpt5mini"]
        runs = [1]
    elif stage == "stage_a":
        briefs = all_briefs
        configs = list_configs_for_stage("stage_a")
        models = ["haiku", "gpt5mini"]
        runs = [1]
    elif stage == "stage_b":
        # Plan §5: top-2 configs per task from Stage A × 23 briefs × cheap models
        # × 2 extra runs (Stage A already ran run_id=1; Stage B adds run_id=2, 3).
        # Sentence tasks ranked by mean cosine; keyword task by mean F1.
        # Ties broken by higher worst-case score (min across briefs).
        scored_csv = cfg.OUTPUTS_DIR / "scored.csv"
        if not scored_csv.exists():
            raise SystemExit(
                f"Missing {scored_csv}. Stage B needs scored Stage-A results to "
                f"pick winners from. Run:\n"
                f"  python -m scripts.analyze --score"
            )

        import pandas as pd
        from src.prompt_builder import Config, KEYWORD_TASK

        scored = pd.read_csv(scored_csv)
        sent = scored[scored["cosine"].notna()]
        kw   = scored[scored["f1"].notna()]

        shortlist: dict[str, list[str]] = {}  # task -> [config_id, config_id]

        if not sent.empty:
            agg = (
                sent.groupby(["task", "config_id"])["cosine"]
                .agg(mean_score="mean", worst_score="min")
                .reset_index()
                .sort_values(
                    ["task", "mean_score", "worst_score"],
                    ascending=[True, False, False],
                )
            )
            for task, g in agg.groupby("task"):
                shortlist[task] = g.head(2)["config_id"].tolist()

        if not kw.empty:
            kw_agg = (
                kw.groupby("config_id")["f1"]
                .agg(mean_score="mean", worst_score="min")
                .reset_index()
                .sort_values(["mean_score", "worst_score"], ascending=[False, False])
            )
            shortlist[KEYWORD_TASK] = kw_agg.head(2)["config_id"].tolist()

        if not shortlist:
            raise SystemExit(
                f"{scored_csv} has no usable rows — Stage B can't pick winners."
            )

        configs: list[Config] = []
        for task, cfg_ids in shortlist.items():
            for cid in cfg_ids:
                fields = _config_id_to_fields(cid)
                version, _, _ = cid.partition(":")
                version = version or "A"
                if isinstance(fields, str):
                    configs.append(Config(task=task, fields=(fields,), prompt_version=version))
                else:
                    configs.append(Config(task=task, fields=fields, prompt_version=version))

        briefs = all_briefs
        models = ["haiku", "gpt5mini"]
        runs = [2, 3]  # Stage A already produced run_id=1.

        print(f"[stage_b] Top-2 shortlist (will rerun on {len(briefs)} briefs × "
              f"{len(models)} models × runs {runs}):")
        for task, cfg_ids in sorted(shortlist.items()):
            print(f"  {task:18s} -> {cfg_ids}")
    elif stage == "phase4":
        # Premium re-run on top-1 cheap-screen winner per task.
        # Brief subset = PHASE0_BRIEF_NAMES (curated for category diversity).
        # Budget envelope kept tight by config.BUDGET_CAP["phase_4_premium"] (~£1).
        wanted = set(PHASE0_BRIEF_NAMES)
        briefs = [b for b in all_briefs if brief_id(b) in wanted]
        missing = wanted - {brief_id(b) for b in briefs}
        if missing:
            raise SystemExit(f"PHASE0_BRIEF_NAMES not found in briefs.yml: {missing}")

        from src.prompt_builder import Config
        top = _load_top_configs_per_task()
        if not top:
            raise SystemExit("No top configs resolved from summary_by_config.csv.")

        configs = []
        for task, info in top.items():
            fields = _config_id_to_fields(info["config_id"])
            if isinstance(fields, str):
                configs.append(Config(task=task, fields=(fields,), prompt_version=info["prompt_version"]))
            else:
                configs.append(Config(task=task, fields=fields, prompt_version=info["prompt_version"]))

        models = PHASE4_MODELS
        runs = [1]
        print(f"[phase4] Promoting {len(configs)} top configs to medium+premium models {models}")
        for c in configs:
            print(f"  {c.task}: {c.config_id}")
    elif stage == "stage_c":
        raise NotImplementedError(
            "Stage C needs the shortlist + medium-model config. Implement after "
            "Stage B selection is complete."
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

    todo: list[dict] = []
    for brief in briefs:
        bid = brief_id(brief)
        for c in configs:
            # Config.fields is a tuple. build_prompt expects either a
            # sentinel string (FULL_BRIEF / PROMPT_IMPLIED) or a tuple of
            # field names. Unpack the 1-tuple-of-sentinel case.
            if len(c.fields) == 1 and c.fields[0] in SENTINELS:
                fields_arg = c.fields[0]
            else:
                fields_arg = c.fields

            try:
                prompt = build_prompt(
                    task=c.task,
                    fields=fields_arg,
                    brief=brief,
                    prompt_version=c.prompt_version,
                    templates=templates,
                )
            except KeyError as e:
                # Missing template — skip rather than crash the whole stage
                print(f"  [skip] {bid} / {c.task} / {c.config_id}: {e}")
                continue

            for model in models:
                for run_id in runs:
                    todo.append({
                        "brief_id": bid,
                        "task": c.task,
                        "config_id": c.config_id,
                        "model_key": model,
                        "run_id": run_id,
                        "prompt": prompt,
                    })
    return todo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=["phase0", "stage_a", "stage_b", "phase4", "stage_c"],
        help=(
            "phase0  = pilot on 3 briefs (cheap models). "
            "stage_a = Phases 1-3 on 23 briefs (cheap). "
            "stage_b = stability reruns + optional Sonnet judge (not auto-wired). "
            "phase4  = top-1 config per task on PREMIUM models (Opus 4.7 + GPT-5.5), "
            "          bounded to ~£1 — final validation step."
        ),
    )
    ap.add_argument("--max-calls", type=int, default=None,
                    help="Hard cap on API calls this run (in addition to budget cap).")
    ap.add_argument("--budget-usd", type=float, default=None,
                    help="USD cap (whole stage). Defaults to config.BUDGET_CAP[stage].")
    ap.add_argument("--per-model-budget-usd", type=float,
                    default=cfg.PER_MODEL_BUDGET_CAP_USD,
                    help=(
                        "USD cap for any single model_key. "
                        f"Defaults to config.PER_MODEL_BUDGET_CAP_USD "
                        f"(currently ${cfg.PER_MODEL_BUDGET_CAP_USD})."
                    ))
    ap.add_argument("--dry-run", action="store_true",
                    help="Build todo and print counts, don't call API.")
    ap.add_argument("--concurrency", type=int, default=20,
                    help="Worker-pool size. Per-model RPM/TPM limits are now "
                         "enforced inside LLMClient via config.CONCURRENCY "
                         "semaphores, so this is just the upper bound on "
                         "total in-flight calls. 20 is fine for Stage A "
                         "(haiku gates itself at 5). Use 1 for debugging.")
    # Default ON: every phase end auto-produces the engineering-note Doc and
    # the deliverable workbook. Use BooleanOptionalAction so both
    # --auto-build and --no-auto-build are valid (the latter is the off-switch).
    ap.add_argument("--auto-build", action=argparse.BooleanOptionalAction, default=True,
                    help="After the phase ends, chain analyze --score, "
                         "--summarize, build_engineering_note, build_xlsx. "
                         "URLs are surfaced in the STOP gate. "
                         "Default ON. Disable with --no-auto-build.")
    ap.add_argument("--dry-run-build", action="store_true",
                    help="With --auto-build: write HTML/xlsx locally to "
                         "Results/ instead of uploading to Drive. "
                         "Useful to preview before committing to a Drive upload.")
    args = ap.parse_args()

    stage_key = {
        "phase0":  "phase_0",
        "stage_a": "phase_1",            # Stage A covers Phases 1-3; phase_1 cap is the largest
        "stage_b": "stage_b",
        "phase4":  "phase_4_premium",    # Hard-bounded to ≤£1 (project rule)
        "stage_c": "stage_c",
    }[args.stage]
    budget = args.budget_usd if args.budget_usd is not None else cfg.BUDGET_CAP[stage_key]

    client = LLMClient.from_env(
        call_cap=args.max_calls,
        budget_cap_usd=budget,
        per_model_cap_usd=args.per_model_budget_usd,
    )

    done = load_done_keys(OUT_JSONL)
    print(f"Resume: {len(done)} calls already done in {OUT_JSONL.name}")

    todo = build_todo(args.stage)
    todo = [
        t for t in todo
        if (t["brief_id"], t["task"], t["config_id"], t["model_key"], t["run_id"]) not in done
    ]
    print(f"Will run {len(todo)} calls. Budget cap: ${budget:.2f}")

    if args.dry_run:
        print("[dry-run] not calling API.")
        return

    print(f"Concurrency: {args.concurrency}")
    start = time.monotonic()

    # Thread-safe primitives:
    # - write_lock serializes JSONL appends (file is the source of truth)
    # - stop_event short-circuits remaining work when BUDGET_EXCEEDED is hit
    # - counter_lock guards `completed` so progress prints stay sequential
    write_lock = threading.Lock()
    stop_event = threading.Event()
    counter_lock = threading.Lock()
    state = {"completed": 0}

    def process_one(t: dict):
        if stop_event.is_set():
            return None
        result = client.call(
            model_key=t["model_key"],
            prompt=t["prompt"],
            brief_id=t["brief_id"],
            task=t["task"],
            config_id=t["config_id"],
            run_id=t["run_id"],
        )
        with write_lock:
            append_jsonl(OUT_JSONL, asdict(result))
        if result.status == Status.BUDGET_EXCEEDED:
            stop_event.set()
        with counter_lock:
            state["completed"] += 1
            n = state["completed"]
        # Progress log every 25 (or last) calls
        if n % 25 == 0 or n == len(todo):
            print(
                f"[{n}/{len(todo)}] {result.model_key} {result.task} "
                f"{result.config_id} status={result.status} "
                f"in={result.input_tokens} out={result.output_tokens} "
                f"r={result.reasoning_tokens} "
                f"cost=${client.total_cost_usd:.4f}"
            )
        return result

    budget_hit_count = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(process_one, t) for t in todo]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                print(f"  [unexpected] {type(e).__name__}: {e}")
                continue
            if result is None:
                continue
            if result.status == Status.BUDGET_EXCEEDED:
                budget_hit_count += 1
                if budget_hit_count == 1:
                    print(f"\nBUDGET EXCEEDED — stopping new calls. "
                          f"In-flight ones may still complete.")

    elapsed = time.monotonic() - start
    print(f"\nDone in {elapsed/60:.1f} min. Calls: {client.call_count}, "
          f"Total cost: ${client.total_cost_usd}")

    per_model = client.per_model_cost_snapshot()
    if per_model:
        cap = args.per_model_budget_usd
        print("Per-model spend (cap=${:.2f}/model):".format(cap))
        for mk, c in sorted(per_model.items(), key=lambda kv: -kv[1]):
            flag = "  [WARN] AT CAP" if c >= cap else ""
            print(f"  {mk:<10} ${c:>7.4f}{flag}")

    artifacts: list[dict] = []
    if args.auto_build and args.stage in _CHECKPOINT_NUMBER:
        artifacts = _auto_build_artifacts(args.stage, dry_run_build=args.dry_run_build)

    _print_checkpoint_gate(args.stage, client.total_cost_usd, per_model, artifacts)


_NEXT_STEP = {
    "phase0":  "stage_a",
    "stage_a": "stage_b",
    "stage_b": "phase4",
    "phase4":  None,           # final
    "stage_c": None,
}

_CHECKPOINT_NAME = {
    "phase0":  "CP1 — Pilot",
    "stage_a": "CP2 — Stage A (cheap-model screening)",
    "stage_b": "CP3 — Stage B (stability + judge)",
    "phase4":  "CP4 — Premium re-run (FINAL)",
}

# stage -> CP integer. Kept for the STOP-gate banner label only.
_CHECKPOINT_NUMBER = {
    "phase0":  1,
    "stage_a": 2,
    "stage_b": 3,
    "phase4":  4,
}

# Regex for the first Google Docs/Sheets URL printed by a build_* script.
_URL_RE = re.compile(r"https://(?:docs|drive)\.google\.com/\S+")


def _run_build_step(label: str, cmd: list[str]) -> dict:
    """
    Run one build sub-command. Captures stdout/stderr, extracts the first
    Google-Doc URL if any, never raises — returns a dict the STOP gate can
    print.

    On non-zero exit, includes the tail of stderr so the failure is
    actionable. We never let one failed build step abort the whole chain;
    every artifact is independent on disk.
    """
    print(f"\n[auto-build] {label}: {' '.join(cmd[2:])}")
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as e:
        return {"label": label, "status": "FAILED", "error": f"executable not found: {e}", "url": None}
    except Exception as e:
        return {"label": label, "status": "FAILED", "error": f"{type(e).__name__}: {e}", "url": None}

    stdout = (r.stdout or "").strip()
    stderr = (r.stderr or "").strip()
    if r.returncode != 0:
        # Last ~6 stderr lines is usually enough to diagnose
        tail = "\n    ".join(stderr.splitlines()[-6:]) if stderr else "(no stderr)"
        print(f"  [WARN] {label} failed (rc={r.returncode}):\n    {tail}")
        return {"label": label, "status": "FAILED", "error": tail, "url": None}

    m = _URL_RE.search(stdout)
    url = m.group(0).rstrip(".,)") if m else None
    # Show a tiny tail of stdout so progress is visible
    if stdout:
        tail = stdout.splitlines()[-3:]
        for line in tail:
            print(f"  {line}")
    return {"label": label, "status": "OK", "error": None, "url": url}


def _auto_build_artifacts(stage: str, dry_run_build: bool) -> list[dict]:
    """
    Post-experiment chain. Idempotent; safe to re-run.

      1. analyze --score        (paid embedding; ~$0.02–0.05 / Stage A)
      2. analyze --summarize    (offline)
      3. build_engineering_note (Drive upload — Google Doc)
      4. build_xlsx             (Drive upload — replaces RESULTS_SHEETS_ID in place)
      5. audit_workbook         (cross-check every auto-generated claim vs
                                 scored.csv. Non-blocking: a failure prints
                                 [WARN] in the STOP gate but downstream stages
                                 are not gated on it.)

    Steps 3 and 4 are kept local (no Drive upload) when dry_run_build=True.

    Dependency map for fail-soft skipping:
      - workbook needs scored.csv (build_xlsx reads scored.csv directly;
        --summarize is optional but generates summary_by_config.csv used by
        the Premium-tier panel) → skipped if score fails.
      - audit needs both scored.csv AND the workbook on disk → skipped if
        either prerequisite failed.
      - engineering_note is independent (reads results.jsonl directly) and
        always attempted.
    """
    if stage not in _CHECKPOINT_NUMBER:
        return []

    py = sys.executable
    dry = ["--dry-run"] if dry_run_build else []

    plan = [
        ("score",             [py, "-m", "scripts.analyze", "--score"]),
        ("summarize",         [py, "-m", "scripts.analyze", "--summarize"]),
        ("engineering_note",  [py, "-m", "scripts.build_engineering_note"] + dry),
        ("workbook",          [py, "-m", "scripts.build_xlsx"] + dry),
        ("audit",             [py, "-m", "scripts.audit_workbook"]),
    ]

    print(f"\n{'='*78}\n  AUTO-BUILD: chaining post-experiment artifacts for {stage}")
    if dry_run_build:
        print("  (--dry-run-build: HTML/xlsx written locally to Results/, no Drive upload)")
    else:
        print("  Uploading to Google Drive — needs credentials.json + token.json.")
        print("  Embedding score step may cost ~$0.02–0.05 in API charges.")
        print("  Workbook upload REPLACES the existing RESULTS_SHEETS_ID Sheet in-place (URL unchanged).")
    print("=" * 78)

    results: list[dict] = []
    score_ok = True
    workbook_ok = True
    for name, cmd in plan:
        # Cascade skip: workbook needs scored.csv. Audit needs both
        # scored.csv AND the built workbook. Engineering note reads
        # results.jsonl directly and is independent.
        if name == "workbook" and not score_ok:
            results.append({"label": name, "status": "SKIPPED",
                            "error": "depends on score (which failed above)",
                            "url": None})
            workbook_ok = False
            continue
        if name == "audit" and (not score_ok or not workbook_ok):
            results.append({"label": name, "status": "SKIPPED",
                            "error": "depends on score + workbook (which failed above)",
                            "url": None})
            continue
        r = _run_build_step(name, cmd)
        results.append(r)
        if name == "score" and r["status"] != "OK":
            score_ok = False
        if name == "workbook" and r["status"] != "OK":
            workbook_ok = False
    return results


def _print_checkpoint_gate(
    stage: str,
    cost_usd: float,
    per_model: dict[str, float] | None = None,
    artifacts: list[dict] | None = None,
) -> None:
    """Hard stop-and-wait message at the end of every phase.

    Every phase ends in `>>> STOP <<<` — the next phase should not be
    started until the artefact URLs above have been reviewed.
    """
    name = _CHECKPOINT_NAME.get(stage, stage)
    nxt = _NEXT_STEP.get(stage)
    bar = "=" * 78
    print(f"\n{bar}")
    print(f"  CHECKPOINT REACHED: {name}")
    print(f"  Spent this phase:  ${cost_usd:.4f}")
    print(f"  Budget ceiling:    £50 total (~$63)  |  per-model cap: ${cfg.PER_MODEL_BUDGET_CAP_USD}")
    if per_model:
        top = max(per_model.values())
        worst = max(per_model.items(), key=lambda kv: kv[1])
        print(f"  Highest single-model spend: {worst[0]} = ${top:.4f}")

    if artifacts:
        print("")
        print("  AUTO-BUILD ARTIFACTS")
        for a in artifacts:
            tag = {"OK": "[OK]", "FAILED": "[WARN]", "SKIPPED": "↷"}.get(a["status"], "?")
            line = f"    {tag} {a['label']:<18}"
            if a["url"]:
                line += a["url"]
            elif a["status"] == "OK":
                line += "(local — no Drive upload)"
            else:
                line += f"({a['status']}: {a.get('error') or '—'})"
            print(line)
        failed = [a for a in artifacts if a["status"] == "FAILED"]
        if failed:
            print("")
            print("  [WARN] One or more auto-build steps failed — fix and retry manually:")
            for a in failed:
                if a["label"] == "score":
                    print(f"     python -m scripts.analyze --score")
                elif a["label"] == "summarize":
                    print(f"     python -m scripts.analyze --summarize")
                elif a["label"] == "engineering_note":
                    print(f"     python -m scripts.build_engineering_note")
                elif a["label"] == "workbook":
                    print(f"     python -m scripts.build_xlsx")
                elif a["label"] == "audit":
                    print(f"     python -m scripts.audit_workbook")

    print(f"")
    print(f"  >>> STOP <<<")
    print(f"  1. Open the URLs above, sanity-check the engineering note + workbook.")
    print(f"  2. Review before running the next stage.")
    if nxt:
        print(f"  3. When ready: python -m scripts.run_experiment --stage {nxt}")
    else:
        print(f"  3. Final phase — run `python -m scripts.build_xlsx` "
              f"to refresh the deliverable workbook if needed.")
    print(f"{bar}\n")


if __name__ == "__main__":
    main()
