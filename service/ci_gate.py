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

Sentence tasks are gated on cosine; the keyword task on F1.

Two properties this gate has to get right, both learned the hard way:

1. **Pin the recipe.** The per-task metric is the mean of the config recorded in
   the baseline — NOT `max` over every config. `max` over ~142 recipes is an
   order statistic: it is biased upward and has far more run-to-run variance
   than any single recipe's mean, and it can silently switch which recipe it is
   describing between runs. You would be comparing recipe A's score today
   against recipe B's score yesterday and calling the difference a regression.
   `max` is used only when creating a fresh baseline, where picking the leader
   is the point.

2. **Never gate tighter than the noise.** The tolerance defaults to the
   empirically measured 2-sigma rerun noise (cfg.NOISE_FLOOR_COSINE). The old
   default of 0.02 was *below* that floor, so identical quality could trip the
   gate purely from sampling noise — a red build that means nothing, which is
   how teams learn to ignore the gate.
"""
import argparse
import json
import sys
from pathlib import Path

from src import config as cfg

BASELINE_PATH = Path(__file__).resolve().parent / "eval_baseline.json"
# Tie the release tolerance to the measured noise floor rather than a hand-picked
# constant. If the floor is re-measured, the gate moves with it automatically.
DEFAULT_TOLERANCE = float(getattr(cfg, "NOISE_FLOOR_COSINE", 0.036))
DEFAULT_MIN_OK_RATE = 0.90

OK_STATUSES = {"ok", "ok_length_violation"}


def compute_metrics(scored_csv: Path, pinned: dict | None = None) -> dict:
    """Return {'tasks': {task: {...}}, 'ok_rate': float, 'n': int}.

    ``pinned`` maps task -> config_id (normally read from the baseline). A task
    with a pinned config is scored on THAT config only; tasks without one fall
    back to the best config, which is what you want when writing a new baseline.
    """
    import pandas as pd

    if not scored_csv.exists():
        raise SystemExit(
            f"Missing {scored_csv}. Run `python -m scripts.analyze --score` first."
        )
    df = pd.read_csv(scored_csv)
    if df.empty:
        raise SystemExit(f"{scored_csv} is empty.")

    pinned = pinned or {}
    tasks: dict[str, dict] = {}
    for task, g in df.groupby("task"):
        # Keyword task carries f1; sentence tasks carry cosine.
        metric = "f1" if ("f1" in g and g["f1"].notna().any()) else "cosine"
        sub = g[g[metric].notna()]
        if sub.empty:
            continue
        per_config = sub.groupby("config_id")[metric].mean()

        want = pinned.get(str(task))
        if want is not None and want in per_config.index:
            config, value, pin_status = want, float(per_config[want]), "pinned"
        elif want is not None:
            # The baselined recipe is absent from this run. Comparing some other
            # recipe against it would be meaningless, so surface it instead.
            tasks[str(task)] = {
                "metric": metric,
                "value": None,
                "config": want,
                "pin_status": "missing",
                "n": 0,
            }
            continue
        else:
            config, value, pin_status = str(per_config.idxmax()), float(per_config.max()), "best"

        n = int(sub[sub["config_id"] == config][metric].notna().sum())
        tasks[str(task)] = {
            "metric": metric,
            "value": round(value, 6),
            "config": config,
            "pin_status": pin_status,
            "n": n,
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
        # `config` and `n` are provenance, not decoration: without the config the
        # next run cannot reproduce which recipe this number describes.
        "tasks": {
            t: {"metric": m["metric"], "value": m["value"],
                "config": m.get("config"), "n": m.get("n")}
            for t, m in metrics["tasks"].items()
        },
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
        if cur is None or cur.get("value") is None:
            why = "recipe gone" if cur else "task gone"
            lines.append(f"{task:22s} {base['metric']:7s} {base['value']:>9.4f} "
                         f"{'MISSING':>9s} {'—':>8s}  FAIL ({why})")
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

    # Load the baseline FIRST so its pinned recipes drive scoring. Scoring the
    # run's own best recipe and then comparing it to a baseline built from a
    # different recipe is the bug this ordering prevents.
    baseline = load_baseline(args.baseline) if not args.update_baseline else {}
    pinned = {
        t: m["config"]
        for t, m in (baseline.get("tasks") or {}).items()
        if isinstance(m, dict) and m.get("config")
    }
    metrics = compute_metrics(args.scored, pinned=pinned)

    if args.update_baseline:
        write_baseline(args.baseline, metrics, tolerance=args.tolerance, min_ok_rate=args.min_ok_rate)
        return 0

    if not baseline:
        print("No baseline found — nothing to regress against. "
              "Create one with `python -m service.ci_gate --update-baseline`.")
        for t, m in sorted(metrics["tasks"].items()):
            print(f"  {t:22s} {m['metric']:7s} {m['value']:.4f}  [{m.get('config')}]")
        return 0

    passed, lines = check(metrics, baseline)
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
