"""
Absolute 1–5 AI judge for the Human-Review sample.

Mirrors scripts.pairwise_judge but produces absolute scores instead of A/B
comparisons. Used to populate the "Sonnet 1-5" column in the Appendix
tab of the deliverable workbook. Pair with `compute_kappa` once humans
have also filled the "Human 1-5" column for a Cohen's-κ check.

Sampling is delegated to build_xlsx._sample_for_human_review so the same
30 stratified rows are scored as those displayed in the workbook (no row
drift between the two artefacts).

Resume key: (brief_id, task, config_id, model_key, run_id). Re-running the
script skips already-scored entries — safe to interrupt and resume.

Usage:
    python -m scripts.ai_judge_absolute
    python -m scripts.ai_judge_absolute --sample-n 30 --budget 0.50

Output: outputs/ai_judge_absolute.jsonl  +  console summary.
Cost cap: $0.50 by default (30 samples × Sonnet ≈ $0.05 typical).
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config as cfg
from src.llm_client import DONE_STATUSES, LLMClient, Status

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


JUDGE_MODEL = "sonnet"
DEFAULT_BUDGET = 0.50
OUT_PATH = cfg.OUTPUTS_DIR / "ai_judge_absolute.jsonl"
SCORED_CSV = cfg.OUTPUTS_DIR / "scored.csv"


ABSOLUTE_TEMPLATE = """You are evaluating an AI-generated brand description against ground truth.

TASK: Write a brand description focused on "{task}".

GROUND TRUTH:
{ground_truth}

AI OUTPUT:
{ai_output}

Rate how well the AI output matches the ground truth in meaning and tone, on a 1-5 scale:
5 — Excellent: meaning closely matches ground truth, tone aligned.
4 — Good: most of the meaning is there, minor gaps or wording differences.
3 — Adequate: rough alignment but noticeable misses or off-tone segments.
2 — Poor: only partial alignment, significant gaps in meaning.
1 — Bad: largely off-target, wrong topic, or unrelated.

Reply with ONLY a single digit 1, 2, 3, 4, or 5. No prose, no explanation."""


_SCORE_RE = re.compile(r"[1-5]")


def _parse_score(text: Optional[str]) -> Optional[int]:
    """Extract a single 1–5 integer from the judge response.

    Tolerates wrappers like 'Score: 4', '4.', '4/5', '"4"'. Returns None if no
    digit in 1-5 appears in the first ~20 characters (longer strings are
    suspicious — the judge probably ignored the 'single digit' instruction).
    """
    if not text:
        return None
    s = text.strip()
    # First scan a short prefix to avoid pulling a digit out of a long monologue
    head = s[:20]
    m = _SCORE_RE.search(head)
    if m:
        return int(m.group(0))
    return None


def _done_keys(path: Path) -> set:
    """Set of (brief_id, task, config_id, model_key, run_id) already scored."""
    if not path.exists():
        return set()
    out = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("score") is None:
                continue
            out.add((
                r.get("brief_id"), r.get("task"), r.get("config_id"),
                r.get("model_key"), r.get("run_id"),
            ))
    return out


def _build_sample(scored: pd.DataFrame, sample_n: int) -> pd.DataFrame:
    """Reuse build_xlsx._sample_for_human_review so the AI judge scores the
    SAME rows the workbook displays. Avoids row-drift between artifacts."""
    from scripts.build_xlsx import _sample_for_human_review
    return _sample_for_human_review(scored, n=sample_n)


def judge_one(client: LLMClient, row: pd.Series) -> Optional[dict]:
    """Run one absolute-score judge call. Returns dict or None on failure."""
    task = row["task"]
    ground_truth = (row.get("ground_truth") or "").strip()
    prediction = (row.get("prediction") or "").strip()
    if not ground_truth or not prediction:
        return None
    prompt = ABSOLUTE_TEMPLATE.format(
        task=task, ground_truth=ground_truth, ai_output=prediction,
    )
    r = client.call(
        model_key=JUDGE_MODEL,
        prompt=prompt,
        brief_id=row["brief_id"],
        task=f"absolute:{task}",
        config_id=row["config_id"],
        run_id=int(row.get("run_id", 1)),
    )
    if r.status == Status.BUDGET_EXCEEDED:
        raise SystemExit("Budget cap hit — stopping. Re-run to continue (resume-safe).")
    if r.status not in DONE_STATUSES:
        return None
    score = _parse_score(r.raw_response)
    return {
        "brief_id": row["brief_id"],
        "task": task,
        "config_id": row["config_id"],
        "model_key": row["model_key"],
        "run_id": int(row.get("run_id", 1)),
        "cosine": float(row["cosine"]) if pd.notna(row.get("cosine")) else None,
        "score": score,
        "raw_response": (r.raw_response or "")[:200],
        "judge_model": JUDGE_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample-n", type=int, default=30,
                    help="Number of stratified samples to judge (default 30).")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help=f"USD budget cap (default ${DEFAULT_BUDGET}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the sample list without calling the judge.")
    args = ap.parse_args()

    if not SCORED_CSV.exists():
        raise SystemExit(
            f"Missing {SCORED_CSV}. Run `python -m scripts.analyze --score` first."
        )

    scored = pd.read_csv(SCORED_CSV)
    sample = _build_sample(scored, args.sample_n)
    if sample.empty:
        raise SystemExit(
            "Stratified sample is empty — no scored sentence-task rows found."
        )

    print(f"Stratified sample: {len(sample)} rows across "
          f"{sample['task'].nunique()} tasks.")
    print(f"Judge: {JUDGE_MODEL}  ·  Budget cap: ${args.budget:.2f}")

    if args.dry_run:
        print("\n[dry-run] Would judge these rows:")
        for _, r in sample.iterrows():
            print(f"  {r['task']:18s}  {r['brief_id']:24s}  "
                  f"{r['config_id']:24s}  cos={r['cosine']:.2f}")
        return

    done = _done_keys(OUT_PATH)
    todo = sample[~sample.apply(
        lambda r: (r["brief_id"], r["task"], r["config_id"],
                   r["model_key"], int(r.get("run_id", 1))) in done,
        axis=1,
    )]
    print(f"Already scored: {len(done)}.  To judge this run: {len(todo)}.")
    if todo.empty:
        print("Nothing to do. Delete outputs/ai_judge_absolute.jsonl to re-judge.")
        return

    client = LLMClient.from_env(budget_cap_usd=args.budget)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parsed = 0
    parse_fails = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            rec = judge_one(client, row)
            if rec is None:
                print(f"  [{i}/{len(todo)}] {row['task']:18s} "
                      f"{row['brief_id']:20s}  ⚠️ judge call failed")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if rec["score"] is not None:
                parsed += 1
                print(f"  [{i}/{len(todo)}] {row['task']:18s} "
                      f"{row['brief_id']:20s}  cos={row['cosine']:.2f} → {rec['score']}")
            else:
                parse_fails += 1
                print(f"  [{i}/{len(todo)}] {row['task']:18s} "
                      f"{row['brief_id']:20s}  ⚠️ unparseable response: "
                      f"{rec['raw_response'][:60]!r}")

    print()
    print("=" * 70)
    print(f"Parsed scores:   {parsed}")
    print(f"Parse failures:  {parse_fails}")
    print(f"Cost spent:      ${client.total_cost_usd:.4f}  (cap ${args.budget:.2f})")
    print(f"Calls made:      {client.call_count}")
    print(f"Dump:            {OUT_PATH}")
    print()
    print("Next: rebuild the workbook so the Sonnet 1-5 column picks up these scores.")
    print("  python -m scripts.build_xlsx")
    print()
    print("Then have humans fill the Human 1-5 column in the Sheet, download the xlsx,")
    print("and run `python -m scripts.compute_kappa` for the agreement score.")


if __name__ == "__main__":
    main()
