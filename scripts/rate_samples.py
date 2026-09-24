"""Blinded human-rating harness — the input side of judge calibration.

The AI judge (Sonnet) rates thousands of outputs. Whether those ratings mean
anything depends entirely on one number: agreement with a human on a sample.
This script collects that human side properly, because a rating collected badly
produces a kappa that is worse than no kappa at all — it looks like evidence.

Four things it does that a spreadsheet column does not:

1. **Blinds the rater.** The judge's score and the automatic cosine are hidden.
   Seeing "Sonnet: 5" before you rate is an anchor, and an anchored human
   agreeing with the judge is not evidence that the judge is right.
2. **Randomises order per rater.** The workbook lists samples in descending
   cosine, so rating top-to-bottom means drifting down a quality gradient and
   recalibrating as you go. A seeded shuffle removes the gradient.
3. **Supports a second pass** (``--pass 2``) over a re-shuffled subset, which
   gives intra-rater agreement — an upper bound on any judge-vs-human kappa.
   If you cannot agree with yourself at 0.8, no judge can agree with you at 0.9.
4. **Records provenance.** Rater id, pass number, timestamp and free-text notes
   per rating, so a kappa can always be traced back to who produced it and when.

Usage::

    python -m scripts.rate_samples --rater olivia            # pass 1, all samples
    python -m scripts.rate_samples --rater olivia --pass 2 --sample 10
    python -m scripts.rate_samples --rater olivia --status   # progress, no prompts

Ratings land in ``outputs/human_ratings.jsonl`` (append-only, resume-safe:
re-running skips what this rater already rated on this pass).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from src import config as cfg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RATINGS_PATH = cfg.OUTPUTS_DIR / "human_ratings.jsonl"
XLSX_PATH = cfg.RESULTS_DIR / "Prompt Eval Results.xlsx"
SHEET_CANDIDATES = ["Human Review", "人工评审", "Appendix"]
HEADER_CANDIDATES = ["Human 1-5", "人工 1-5"]

# Column layout of the Human-Review table (1-based, matches build_xlsx).
COL_TASK, COL_RECIPE, COL_MODEL, COL_GT, COL_OUT, COL_COSINE, COL_JUDGE = 1, 2, 3, 4, 5, 6, 7

RUBRIC = """\
  5  Fully captures the ground truth's meaning. Any difference is wording.
  4  Captures the main point; one secondary element is missing or imprecise.
  3  Partially right — the core idea is there but distorted, over-general,
     or padded with something the ground truth does not claim.
  2  Mostly wrong: recognisably about the same subject, but the substantive
     claim differs from the ground truth.
  1  Wrong, empty, a refusal, or an answer to a different question.

  Rate MEANING, not style. Do not reward fluency and do not punish a blunt
  phrasing that says the right thing."""


# ---------------------------------------------------------------- sample source

def sample_id(task: str, recipe: str, model: str, ground_truth: str, ai_output: str) -> str:
    """Content-addressed id: (task, recipe, model) + a hash of what was shown.

    (task, recipe, model) alone is not unique — the workbook can hold two rows
    with the same triple for different briefs, and their ratings would then
    overwrite each other. Hashing the shown text also gives the id a useful
    property: if a sample's content changes, the id changes, so an old rating
    can never silently attach itself to different text. Same idea as the golden
    manifest, one level down.
    """
    # Length-prefixed so no pair of (ground_truth, ai_output) can collide with
    # a different pair that happens to concatenate to the same string.
    payload = f"{len(ground_truth)}:{ground_truth}{len(ai_output)}:{ai_output}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{task}|{recipe}|{model}|{digest[:10]}"


def load_samples() -> list[dict]:
    """Read the Human-Review table out of the deliverable workbook."""
    import openpyxl

    if not XLSX_PATH.exists():
        raise SystemExit(
            f"Missing {XLSX_PATH}.\nBuild it first: python -m scripts.build_xlsx"
        )
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
        raise SystemExit(f"No {HEADER_CANDIDATES} header found in '{sheet}'.")

    samples = []
    r = header_row + 1
    while ws.cell(r, COL_TASK).value:
        task = str(ws.cell(r, COL_TASK).value).strip()
        recipe = str(ws.cell(r, COL_RECIPE).value or "").strip()
        model = str(ws.cell(r, COL_MODEL).value or "").strip()
        gt = str(ws.cell(r, COL_GT).value or "").strip()
        out = str(ws.cell(r, COL_OUT).value or "").strip()
        samples.append({
            "sample_id": sample_id(task, recipe, model, gt, out),
            "task": task,
            "recipe": recipe,
            "model": model,
            "ground_truth": gt,
            "ai_output": out,
        })
        r += 1
    if not samples:
        raise SystemExit(f"Found the header in '{sheet}' but no sample rows under it.")
    return samples


# ---------------------------------------------------------------- ratings store

def load_ratings() -> list[dict]:
    if not RATINGS_PATH.exists():
        return []
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


def append_rating(row: dict) -> None:
    RATINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- presentation

def _wrap(text: str, indent: str = "     ") -> str:
    if not text:
        return indent + "(empty)"
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def show_sample(sample: dict, position: int, total: int) -> None:
    print("\n" + "=" * 92)
    print(f"  [{position}/{total}]  task: {sample['task']}")
    print("=" * 92)
    print("\n  GROUND TRUTH")
    print(_wrap(sample["ground_truth"]))
    print("\n  AI OUTPUT")
    print(_wrap(sample["ai_output"]))
    print()
    # recipe/model deliberately withheld until after the rating — knowing an
    # output came from the cheap model or the "no brief" baseline is an anchor
    # in exactly the same way the judge's score is.


def prompt_rating(sample: dict) -> tuple[int, str] | None:
    while True:
        try:
            raw = input("  Rating 1-5  (r = show rubric, s = skip, q = save & quit): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — everything rated so far is already saved.")
            return None
        if raw in ("q", "quit"):
            return None
        if raw in ("s", "skip"):
            return (0, "")
        if raw in ("r", "rubric", "?"):
            print("\n" + RUBRIC + "\n")
            continue
        if raw in {"1", "2", "3", "4", "5"}:
            note = input("  Note (optional, Enter to skip): ").strip()
            return (int(raw), note)
        print("  Enter 1, 2, 3, 4, 5 — or r / s / q.")


# ---------------------------------------------------------------- commands

def cmd_status(samples: list[dict], rater: str) -> int:
    ratings = load_ratings()
    print(f"\nSamples available: {len(samples)}")
    by_rater: dict[str, dict[int, int]] = {}
    for row in ratings:
        by_rater.setdefault(row["rater"], {}).setdefault(row["pass"], 0)
        by_rater[row["rater"]][row["pass"]] += 1
    if not by_rater:
        print("No ratings recorded yet.")
    for who, passes in sorted(by_rater.items()):
        marker = "  <- you" if who == rater else ""
        detail = ", ".join(f"pass {p}: {n}" for p, n in sorted(passes.items()))
        print(f"  {who:<16} {detail}{marker}")

    n1 = by_rater.get(rater, {}).get(1, 0)
    print(f"\n{rater}: {n1}/{len(samples)} rated on pass 1.")
    if n1 < 30:
        print(f"  {30 - n1} more needed before a kappa is worth computing.")
    else:
        print("  Enough for a kappa. Run: python -m scripts.compute_kappa")
    if by_rater.get(rater, {}).get(2, 0) < 10:
        print("  Tip: --pass 2 --sample 10 gives intra-rater agreement, which bounds")
        print("       how high any judge-vs-human kappa could honestly be.")
    return 0


def cmd_rate(samples: list[dict], rater: str, pass_no: int, sample_n: int | None,
             seed: int) -> int:
    done = {
        r["sample_id"] for r in load_ratings()
        if r["rater"] == rater and r["pass"] == pass_no and r.get("rating")
    }
    queue = [s for s in samples if s["sample_id"] not in done]

    # Seeded shuffle: removes the workbook's descending-cosine gradient, and is
    # reproducible so an interrupted session resumes in the same order.
    rng = random.Random(f"{rater}|{pass_no}|{seed}")
    rng.shuffle(queue)
    if sample_n:
        queue = queue[:sample_n]

    if not queue:
        print(f"\nNothing left to rate for {rater} on pass {pass_no}.")
        return cmd_status(samples, rater)

    print("\n" + "=" * 92)
    print(f"  BLINDED RATING  ·  rater: {rater}  ·  pass {pass_no}  ·  {len(queue)} to go")
    print("=" * 92)
    print("\n  The judge's score and the automatic cosine are hidden on purpose.")
    print("  Rate what you actually think, then compare. Ctrl-C is safe — every")
    print("  rating is written as you go.\n")
    print(RUBRIC)

    rated = 0
    for i, sample in enumerate(queue, start=1):
        show_sample(sample, i, len(queue))
        result = prompt_rating(sample)
        if result is None:
            break
        rating, note = result
        if rating == 0:
            continue
        append_rating({
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "recipe": sample["recipe"],
            "model": sample["model"],
            "rater": rater,
            "pass": pass_no,
            "rating": rating,
            "note": note,
            "rated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        rated += 1

    print(f"\n\nSaved {rated} rating(s) -> {RATINGS_PATH}")
    return cmd_status(samples, rater)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rater", required=True, help="Your identifier, e.g. olivia")
    ap.add_argument("--pass", dest="pass_no", type=int, default=1, choices=[1, 2],
                    help="1 = the calibration pass; 2 = intra-rater reliability re-rate")
    ap.add_argument("--sample", type=int, default=None,
                    help="Rate only the first N of the shuffled queue (use with --pass 2)")
    ap.add_argument("--seed", type=int, default=7, help="Shuffle seed (reproducible order)")
    ap.add_argument("--status", action="store_true", help="Show progress and exit")
    args = ap.parse_args(argv)

    samples = load_samples()
    if args.status:
        return cmd_status(samples, args.rater)
    return cmd_rate(samples, args.rater, args.pass_no, args.sample, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
