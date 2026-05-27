"""
Pairwise comparison judge — compare two recipes head-to-head with Claude Sonnet.

Industry rationale:
  - Absolute scoring (1-5 stars) is calibration-dependent and noisy.
  - Pairwise comparison ("which is better, A or B?") is more reliable and is
    the basis of Chatbot Arena, RLHF reward modeling, and recent eval research
    (Zheng et al. 2023, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena").

Position-bias mitigation:
  For every comparison we run the prompt twice with A/B swapped and only count
  a win when BOTH orders agree. Disagreement → tie. This neutralizes the
  systematic preference some judges show for whichever output appears first.

Usage:
    python -m scripts.pairwise_judge \\
        --recipe-a A:_full_brief \\
        --recipe-b A:_prompt_implied \\
        --model haiku

Output: outputs/pairwise_results.jsonl  +  console summary.
Cost cap: $2.
"""
import argparse
import json
import sys
from datetime import datetime

from src import config as cfg
from src.llm_client import DONE_STATUSES, LLMClient, Status
from src.prompt_builder import brief_id, load_briefs
from src.utils import read_jsonl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


JUDGE_MODEL = "sonnet"
BUDGET_CAP = 2.0
OUT_PATH = cfg.OUTPUTS_DIR / "pairwise_results.jsonl"

PAIRWISE_TEMPLATE = """You are evaluating two AI-generated descriptions for a brand-brief task.

TASK: Write a brand description focused on "{task}".

GROUND TRUTH (the correct answer the AIs were trying to match):
{ground_truth}

OUTPUT A:
{output_a}

OUTPUT B:
{output_b}

Which output better matches the ground truth in meaning and tone?

Reply with EXACTLY one token from this list: A | B | TIE
- "A" if Output A is meaningfully closer to ground truth
- "B" if Output B is meaningfully closer to ground truth
- "TIE" if both are roughly equivalent

Reply with just the single token."""


def _parse_vote(text: str) -> str:
    """Normalize a free-form judge response into {'A', 'B', 'tie'}."""
    if not text:
        return "tie"
    s = text.strip().upper()
    # Strip common wrappers like quotes, periods, "Answer:" preamble
    s = s.lstrip("\"' .:-")
    if s.startswith("A") and not s.startswith("ABOUT"):
        return "A"
    if s.startswith("B") and not s.startswith("BOTH"):
        return "B"
    if "TIE" in s or "EQUIV" in s or "BOTH" in s:
        return "tie"
    return "tie"  # default conservative


def pairwise_one(client: LLMClient, task: str, ground_truth: str,
                 output_a: str, output_b: str,
                 brief_id_str: str, recipe_a: str, recipe_b: str) -> dict | None:
    """
    Run a pairwise comparison with position swap.

    Returns dict with vote_order1, vote_order2, winner ('A' / 'B' / 'tie').
    Returns None on judge failure.
    """
    # Pass 1: A=A, B=B
    p1 = PAIRWISE_TEMPLATE.format(
        task=task, ground_truth=ground_truth,
        output_a=output_a, output_b=output_b,
    )
    r1 = client.call(
        model_key=JUDGE_MODEL, prompt=p1,
        brief_id=brief_id_str, task=f"pairwise:{task}",
        config_id=f"{recipe_a}__vs__{recipe_b}", run_id=1,
    )
    if r1.status == Status.BUDGET_EXCEEDED:
        raise SystemExit("Budget cap hit. Stop.")
    if r1.status not in DONE_STATUSES:
        return None

    vote1 = _parse_vote(r1.raw_response)

    # Pass 2: A and B swapped (position-bias control)
    p2 = PAIRWISE_TEMPLATE.format(
        task=task, ground_truth=ground_truth,
        output_a=output_b, output_b=output_a,
    )
    r2 = client.call(
        model_key=JUDGE_MODEL, prompt=p2,
        brief_id=brief_id_str, task=f"pairwise:{task}",
        config_id=f"{recipe_b}__vs__{recipe_a}", run_id=2,
    )
    if r2.status == Status.BUDGET_EXCEEDED:
        raise SystemExit("Budget cap hit. Stop.")
    if r2.status not in DONE_STATUSES:
        return None

    vote2_raw = _parse_vote(r2.raw_response)
    # Flip vote2 because A and B were swapped in pass 2
    flip = {"A": "B", "B": "A", "tie": "tie"}
    vote2_canonical = flip[vote2_raw]

    # Both orders must agree → winner. Otherwise judge is position-biased on
    # this pair → conservative tie.
    winner = vote1 if vote1 == vote2_canonical else "tie"

    return {
        "task": task,
        "brief_id": brief_id_str,
        "recipe_a": recipe_a,
        "recipe_b": recipe_b,
        "vote_pass1": vote1,
        "vote_pass2_raw": vote2_raw,
        "vote_pass2_canonical": vote2_canonical,
        "winner": winner,
        "raw_judge_pass1": r1.raw_response[:200],
        "raw_judge_pass2": r2.raw_response[:200],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--recipe-a", required=True,
                        help="config_id of the first recipe (e.g., A:_full_brief)")
    parser.add_argument("--recipe-b", required=True,
                        help="config_id of the second recipe (e.g., A:_prompt_implied)")
    parser.add_argument("--model", default="haiku",
                        help="Which model's outputs to compare (default: haiku)")
    args = parser.parse_args()

    print(f"Pairwise comparison: {args.recipe_a}  vs  {args.recipe_b}")
    print(f"Comparing outputs from model: {args.model}")
    print(f"Judge: {JUDGE_MODEL}  ·  Budget cap: ${BUDGET_CAP}")
    print()

    # Load briefs
    briefs_by_id = {brief_id(b): b for b in load_briefs()}

    # Load all results, keep only the two recipes for the specified model
    results_path = cfg.OUTPUTS_DIR / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}. Run the experiment first.")

    all_rows = read_jsonl(results_path)
    rows_a = {(r["task"], r["brief_id"]): r["raw_response"]
              for r in all_rows
              if r.get("config_id") == args.recipe_a
              and r.get("model_key") == args.model
              and r.get("status") in DONE_STATUSES}
    rows_b = {(r["task"], r["brief_id"]): r["raw_response"]
              for r in all_rows
              if r.get("config_id") == args.recipe_b
              and r.get("model_key") == args.model
              and r.get("status") in DONE_STATUSES}

    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        raise SystemExit(
            f"No (task, brief) pairs have BOTH recipes complete for model={args.model}. "
            f"Run those configurations first."
        )
    print(f"Found {len(common)} (task, brief) pairs to compare.\n")

    client = LLMClient.from_env(budget_cap_usd=BUDGET_CAP)

    # Aggregates
    wins_a, wins_b, ties = 0, 0, 0
    per_task = {}  # task -> dict of counts
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for (task, bid) in common:
            output_a = rows_a[(task, bid)]
            output_b = rows_b[(task, bid)]
            brief = briefs_by_id.get(bid)
            if brief is None:
                continue
            ground_truth = brief.get(task, "")
            if not ground_truth:
                continue

            result = pairwise_one(
                client, task, ground_truth, output_a, output_b,
                bid, args.recipe_a, args.recipe_b,
            )
            if not result:
                print(f"  [skip] {task:18s} {bid:20s}  judge failed")
                continue

            result["_meta"] = {
                "judge_model": JUDGE_MODEL,
                "compared_model": args.model,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

            w = result["winner"]
            if w == "A":
                wins_a += 1
            elif w == "B":
                wins_b += 1
            else:
                ties += 1

            tb = per_task.setdefault(task, {"A": 0, "B": 0, "tie": 0})
            tb[w] += 1

            print(f"  {task:18s} {bid:22s}  pass1={result['vote_pass1']:4s}  "
                  f"pass2={result['vote_pass2_canonical']:4s}  → {w}")

    # ---- Summary ----
    total = wins_a + wins_b + ties
    if total == 0:
        print("\nNo successful comparisons.")
        return

    print("\n" + "=" * 70)
    print(f"Pairwise summary  ({total} comparisons, model={args.model}, judge={JUDGE_MODEL})")
    print("=" * 70)
    print(f"  {args.recipe_a:30s}  {wins_a:3d} wins  ({100*wins_a/total:5.1f}%)")
    print(f"  {args.recipe_b:30s}  {wins_b:3d} wins  ({100*wins_b/total:5.1f}%)")
    print(f"  TIE / position-biased / disagree     "
          f"  {ties:3d}        ({100*ties/total:5.1f}%)")
    print()
    print("Per task:")
    print(f"  {'Task':20s} {args.recipe_a[:18]:>20s}  {args.recipe_b[:18]:>20s}  TIE")
    for task in sorted(per_task):
        c = per_task[task]
        print(f"  {task:20s} {c['A']:20d}  {c['B']:20d}  {c['tie']:3d}")

    print(f"\nDump:           {OUT_PATH}")
    print(f"Cost spent:     ${client.total_cost_usd:.4f}")
    print(f"Calls made:     {client.call_count}")


if __name__ == "__main__":
    main()
