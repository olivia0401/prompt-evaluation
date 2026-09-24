"""Judge calibration: does the AI judge agree with a human, and how sure are we?

Pairs the blinded human ratings from ``scripts.rate_samples`` against the AI
judge's scores in the deliverable workbook and reports Cohen's weighted kappa
with a bootstrap confidence interval.

    python -m scripts.rate_samples --rater olivia     # collect the human side
    python -m scripts.compute_kappa                   # this script

What it reports, and why each part is here:

* **Both weightings.** Linear and quadratic kappa on the same ratings. They can
  differ by 0.2 — enough to move a result from "reject the judge" to "trust the
  judge" — so a kappa without its weighting scheme is not a result. Linear is
  the headline because it is the stricter of the two.
* **A bootstrap CI.** On ~30 ratings the point estimate is soft: a kappa of 0.72
  whose interval runs [0.48, 0.88] has not cleared a 0.7 bar, it has landed near
  it. The decision uses the lower bound, not the point estimate.
* **An error-direction breakdown.** *How* the judge disagrees matters more than
  how much. A judge that is uniformly one point generous is recalibratable; one
  that disagrees in both directions is measuring something else.
* **Intra-rater kappa** when a second pass exists — how well the human agrees
  with themselves. That is the ceiling: a judge cannot honestly agree with you
  more than you agree with you.

Refuses to print a headline kappa below 30 paired ratings. A kappa on 12 samples
is not a small result, it is a number that cannot support a decision.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import config as cfg
from scripts.rate_samples import sample_id
from src.quality_evaluators import weighted_kappa

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RATINGS_PATH = cfg.OUTPUTS_DIR / "human_ratings.jsonl"
XLSX_PATH = cfg.RESULTS_DIR / "Prompt Eval Results.xlsx"
KAPPA_PATH = cfg.OUTPUTS_DIR / "kappa.json"
SHEET_CANDIDATES = ["Human Review", "人工评审", "Appendix"]
HEADER_CANDIDATES = ["Human 1-5", "人工 1-5"]
COL_TASK, COL_RECIPE, COL_MODEL, COL_GT, COL_OUT, COL_JUDGE = 1, 2, 3, 4, 5, 7

MIN_PAIRS = 30
TRUST_FLOOR = 0.70
REVIEW_FLOOR = 0.40


# ------------------------------------------------------------------- loading

def load_judge_scores() -> dict[str, int]:
    """sample_id -> the AI judge's 1-5 score, from the workbook."""
    import openpyxl

    if not XLSX_PATH.exists():
        raise SystemExit(f"Missing {XLSX_PATH}. Run `python -m scripts.build_xlsx` first.")
    wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
    sheet = next((s for s in SHEET_CANDIDATES if s in wb.sheetnames), None)
    if sheet is None:
        raise SystemExit(f"None of {SHEET_CANDIDATES} in {XLSX_PATH}. Tabs: {wb.sheetnames}")
    ws = wb[sheet]

    header_row = None
    for r in range(1, 60):
        for c in range(1, 12):
            v = ws.cell(r, c).value
            if isinstance(v, str) and any(h in v for h in HEADER_CANDIDATES):
                header_row = r
                break
        if header_row:
            break
    if header_row is None:
        raise SystemExit(f"No {HEADER_CANDIDATES} header in '{sheet}'.")

    scores: dict[str, int] = {}
    r = header_row + 1
    while ws.cell(r, COL_TASK).value:
        sid = sample_id(
            str(ws.cell(r, COL_TASK).value or "").strip(),
            str(ws.cell(r, COL_RECIPE).value or "").strip(),
            str(ws.cell(r, COL_MODEL).value or "").strip(),
            str(ws.cell(r, COL_GT).value or "").strip(),
            str(ws.cell(r, COL_OUT).value or "").strip(),
        )
        j = ws.cell(r, COL_JUDGE).value
        if isinstance(j, (int, float)):
            scores[sid] = int(round(j))
        r += 1
    return scores


def load_human_ratings() -> list[dict]:
    if not RATINGS_PATH.exists():
        raise SystemExit(
            f"Missing {RATINGS_PATH}.\n"
            f"Collect the human side first:\n"
            f"    python -m scripts.rate_samples --rater <you>"
        )
    rows = []
    with open(RATINGS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


# ------------------------------------------------------------------ statistics

def bootstrap_kappa_ci(human: list[int], judge: list[int], *, weights: str,
                       n_resamples: int = 2000, seed: int = 42,
                       confidence: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap over rating PAIRS (resample pairs, not raters)."""
    rng = random.Random(seed)
    n = len(human)
    values = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        h = [human[i] for i in idx]
        j = [judge[i] for i in idx]
        try:
            values.append(weighted_kappa(h, j, weights=weights))
        except ValueError:
            # A resample with no rating variance is undefined, not 1.0. Skipping
            # it slightly narrows the interval; counting it as 1.0 would inflate
            # the upper bound with samples that contain no information at all.
            continue
    if len(values) < n_resamples * 0.5:
        return (float("nan"), float("nan"))
    values.sort()
    lo_i = int((1 - confidence) / 2 * len(values))
    hi_i = int((1 - (1 - confidence) / 2) * len(values)) - 1
    return (values[lo_i], values[max(hi_i, lo_i)])


def error_profile(human: list[int], judge: list[int]) -> dict:
    diffs = [j - h for h, j in zip(human, judge)]
    n = len(diffs)
    return {
        "exact_agreement": sum(d == 0 for d in diffs) / n,
        "within_one": sum(abs(d) <= 1 for d in diffs) / n,
        "judge_generous": sum(d > 0 for d in diffs) / n,
        "judge_harsh": sum(d < 0 for d in diffs) / n,
        "mean_signed_error": sum(diffs) / n,
        "mae": sum(abs(d) for d in diffs) / n,
        "distribution": dict(sorted(Counter(diffs).items())),
    }


# ---------------------------------------------------------------------- report

def _bar(label: str, value: float, width: int = 34) -> str:
    filled = max(0, min(width, round(value * width)))
    return f"  {label:<22} {'█' * filled}{'·' * (width - filled)}  {value:5.1%}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rater", default=None, help="Restrict to one rater id")
    ap.add_argument("--allow-small", action="store_true",
                    help="Report even below the 30-pair floor (clearly labelled)")
    args = ap.parse_args(argv)

    judge_scores = load_judge_scores()
    rows = load_human_ratings()
    if args.rater:
        rows = [r for r in rows if r["rater"] == args.rater]

    pass1 = [r for r in rows if r.get("pass") == 1 and r.get("rating")]
    # De-duplicate: last rating for a given (rater, sample) wins.
    latest: dict[tuple[str, str], dict] = {}
    for r in pass1:
        latest[(r["rater"], r["sample_id"])] = r

    human, judge, kept = [], [], []
    for (rater, sid), r in sorted(latest.items()):
        if sid in judge_scores:
            human.append(int(r["rating"]))
            judge.append(judge_scores[sid])
            kept.append(r)

    n = len(human)
    raters = sorted({r["rater"] for r in kept})

    print()
    print("=" * 72)
    print("  JUDGE CALIBRATION")
    print("=" * 72)
    print(f"  paired ratings : {n}")
    print(f"  raters         : {', '.join(raters) if raters else '(none)'}")
    print(f"  judge scores   : {len(judge_scores)} available in the workbook")

    if n == 0:
        print("\n  No human ratings match a workbook sample yet.")
        print("  Start with: python -m scripts.rate_samples --rater <you>")
        return 1

    if n < MIN_PAIRS and not args.allow_small:
        print(f"\n  Below the {MIN_PAIRS}-pair floor — no kappa reported.")
        print(f"  {MIN_PAIRS - n} more rating(s) needed. A kappa at n={n} cannot")
        print("  support the decision it would be used for.")
        print("  Override for a look with --allow-small (it will be labelled).")
        return 1

    provisional = n < MIN_PAIRS
    try:
        k_lin = weighted_kappa(human, judge, weights="linear")
        k_quad = weighted_kappa(human, judge, weights="quadratic")
    except ValueError as exc:
        print(f"\n  Cannot compute kappa: {exc}")
        return 1

    lo, hi = bootstrap_kappa_ci(human, judge, weights="linear")
    prof = error_profile(human, judge)

    print()
    print("-" * 72)
    print(f"  weighted kappa (linear)     {k_lin:.3f}   [{lo:.3f}, {hi:.3f}]  95% bootstrap")
    print(f"  weighted kappa (quadratic)  {k_quad:.3f}   (reported for comparison only)")
    print("-" * 72)
    print()
    print(_bar("exact agreement", prof["exact_agreement"]))
    print(_bar("within one point", prof["within_one"]))
    print(_bar("judge more generous", prof["judge_generous"]))
    print(_bar("judge harsher", prof["judge_harsh"]))
    print()
    print(f"  mean signed error (judge - human): {prof['mean_signed_error']:+.2f}")
    print(f"  mean absolute error:               {prof['mae']:.2f}")
    print(f"  error distribution:                {prof['distribution']}")

    # Intra-rater ceiling from the second pass, when one exists.
    intra = None
    pass2 = {(r["rater"], r["sample_id"]): r for r in rows
             if r.get("pass") == 2 and r.get("rating")}
    if pass2:
        h1, h2 = [], []
        for key, r2 in sorted(pass2.items()):
            r1 = latest.get(key)
            if r1:
                h1.append(int(r1["rating"]))
                h2.append(int(r2["rating"]))
        if len(h1) >= 5:
            try:
                intra = weighted_kappa(h1, h2, weights="linear")
                print()
                print(f"  intra-rater kappa (linear, n={len(h1)}): {intra:.3f}")
                print("    ^ the ceiling. A judge cannot honestly agree with you")
                print("      more than you agree with yourself.")
            except ValueError:
                pass

    # Decision, taken on the LOWER bound rather than the point estimate.
    basis = lo if lo == lo else k_lin
    if basis >= TRUST_FLOOR:
        decision = "HIGH"
        detail = ("Judge is calibrated. Its ratings may stand in for human review "
                  "on the remaining outputs.")
    elif basis >= REVIEW_FLOOR:
        decision = "MEDIUM"
        detail = ("Moderate. Collect ~30 more ratings, or tighten the rubric — an "
                  "ambiguous rubric usually explains more disagreement than a weak judge.")
    else:
        decision = "LOW"
        detail = ("Judge is not measuring what the rubric describes. Do not use its "
                  "scores as evidence; rewrite the rubric or fall back to humans.")

    print()
    print("=" * 72)
    print(f"  DECISION: {decision}"
          + ("   (PROVISIONAL — below the 30-pair floor)" if provisional else ""))
    print(f"  {detail}")
    print(f"  Taken on the CI lower bound ({basis:.3f}), not the point estimate.")
    print("=" * 72)

    payload = {
        "kappa_linear": round(k_lin, 4),
        "kappa_quadratic": round(k_quad, 4),
        "ci95_linear": [round(lo, 4), round(hi, 4)],
        "n_pairs": n,
        "raters": raters,
        "intra_rater_kappa_linear": round(intra, 4) if intra is not None else None,
        "error_profile": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in prof.items()},
        "decision": decision,
        "decision_basis": "ci_lower_bound",
        "provisional": provisional,
        "weighting_note": "Headline kappa is LINEAR weighted. Quadratic reported for comparison.",
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    KAPPA_PATH.parent.mkdir(parents=True, exist_ok=True)
    KAPPA_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(f"\n-> {KAPPA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
