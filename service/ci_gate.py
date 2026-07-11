"""
Eval regression gate — block a release when quality drops.

Reads the scored results (outputs/scored.csv, produced by
`python -m scripts.analyze --score`), reduces them to one headline metric per
task (the winning config's mean score), and compares against a committed
baseline (service/eval_baseline.json). Exits non-zero if any task regresses by
more than `tolerance`, or if the overall ok-rate falls below `min_ok_rate`.

    python -m service.ci_gate                     # check against baseline (CI)
    python -m service.ci_gate --update-baseline   # snapshot current as the new baseline
    python -m service.ci_gate --scored path.csv   # check a specific scored.csv

Sentence tasks are gated on cosine; the keyword task on F1. Per-task metric is
the best config's mean (max over config_id of the per-config mean) so the gate
tracks the recipe you would actually ship, not the average of all recipes.
"""
import argparse
import json
import sys
from pathlib import Path

from src import config as cfg

BASELINE_PATH = Path(__file__).resolve().parent / "eval_baseline.json"
DEFAULT_TOLERANCE = 0.02
DEFAULT_MIN_OK_RATE = 0.90

OK_STATUSES = {"ok", "ok_length_violation"}


def compute_metrics(scored_csv: Path) -> dict:
    """Return {'tasks': {task: {'metric','value','n'}}, 'ok_rate': float, 'n': int}."""
    import pandas as pd

    if not scored_csv.exists():
        raise SystemExit(
            f"Missing {scored_csv}. Run `python -m scripts.analyze --score` first."
        )
    df = pd.read_csv(scored_csv)
    if df.empty:
        raise SystemExit(f"{scored_csv} is empty.")

    tasks: dict[str, dict] = {}
    for task, g in df.groupby("task"):
        # Keyword task carries f1; sentence tasks carry cosine.
        metric = "f1" if ("f1" in g and g["f1"].notna().any()) else "cosine"
        sub = g[g[metric].notna()]
        if sub.empty:
            continue
        per_config = sub.groupby("config_id")[metric].mean()
        best = per_config.max()
        tasks[str(task)] = {
            "metric": metric,
            "value": round(float(best), 6),
            "best_config": str(per_config.idxmax()),
            "n": int(sub[metric].notna().sum()),
        }

    ok_rate = None
    if "status" in df:
        ok = df["status"].isin(OK_STATUSES).sum()
        ok_rate = round(float(ok) / len(df), 4) if len(df) else None

    return {"tasks": tasks, "ok_rate": ok_rate, "n": int(len(df))}


def load_baseline(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(path: Path, metrics: dict, *, tolerance: float, min_ok_rate: float) -> None:
    payload = {
        "tolerance": tolerance,
        "min_ok_rate": min_ok_rate,
        "ok_rate": metrics.get("ok_rate"),
        "tasks": {t: {"metric": m["metric"], "value": m["value"]} for t, m in metrics["tasks"].items()},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote baseline -> {path} ({len(payload['tasks'])} tasks)")


def check(metrics: dict, baseline: dict) -> tuple[bool, list[str]]:
    """Return (passed, lines). A missing baseline passes (nothing to regress from)."""
    tolerance = baseline.get("tolerance", DEFAULT_TOLERANCE)
    min_ok_rate = baseline.get("min_ok_rate", DEFAULT_MIN_OK_RATE)
    base_tasks = baseline.get("tasks", {})

    lines: list[str] = []
    passed = True
    lines.append(f"{'task':22s} {'metric':7s} {'baseline':>9s} {'current':>9s} {'Δ':>8s}  verdict")
    lines.append("-" * 68)

    for task in sorted(set(base_tasks) | set(metrics["tasks"])):
        cur = metrics["tasks"].get(task)
        base = base_tasks.get(task)
        if base is None:
            lines.append(f"{task:22s} {'-':7s} {'(new)':>9s} "
                         f"{(cur['value'] if cur else 0):>9.4f} {'—':>8s}  NEW")
            continue
        if cur is None:
            lines.append(f"{task:22s} {base['metric']:7s} {base['value']:>9.4f} "
                         f"{'MISSING':>9s} {'—':>8s}  FAIL (task gone)")
            passed = False
            continue
        delta = cur["value"] - base["value"]
        regressed = delta < -tolerance
        verdict = "FAIL" if regressed else "ok"
        if regressed:
            passed = False
        lines.append(f"{task:22s} {cur['metric']:7s} {base['value']:>9.4f} "
                     f"{cur['value']:>9.4f} {delta:>+8.4f}  {verdict}")

    ok_rate = metrics.get("ok_rate")
    if ok_rate is not None:
        verdict = "FAIL" if ok_rate < min_ok_rate else "ok"
        if ok_rate < min_ok_rate:
            passed = False
        lines.append("-" * 68)
        lines.append(f"ok_rate                        {min_ok_rate:>9.4f} {ok_rate:>9.4f} "
                     f"{ok_rate - min_ok_rate:>+8.4f}  {verdict}")

    lines.append("-" * 68)
    lines.append(f"tolerance={tolerance}  min_ok_rate={min_ok_rate}  "
                 f"RESULT={'PASS' if passed else 'FAIL'}")
    return passed, lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Eval regression gate.")
    ap.add_argument("--scored", type=Path, default=cfg.OUTPUTS_DIR / "scored.csv")
    ap.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    ap.add_argument("--min-ok-rate", type=float, default=DEFAULT_MIN_OK_RATE)
    ap.add_argument("--update-baseline", action="store_true",
                    help="Write current metrics as the new baseline and exit 0.")
    args = ap.parse_args(argv)

    metrics = compute_metrics(args.scored)

    if args.update_baseline:
        write_baseline(args.baseline, metrics, tolerance=args.tolerance, min_ok_rate=args.min_ok_rate)
        return 0

    baseline = load_baseline(args.baseline)
    if not baseline:
        print("No baseline found — nothing to regress against. "
              "Create one with `python -m service.ci_gate --update-baseline`.")
        for t, m in sorted(metrics["tasks"].items()):
            print(f"  {t:22s} {m['metric']:7s} {m['value']:.4f}")
        return 0

    passed, lines = check(metrics, baseline)
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
