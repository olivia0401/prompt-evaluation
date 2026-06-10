"""
Empirically measure the within-config noise floor sigma, used to set the
"practically equivalent" and "significantly better" decision thresholds.

Method:
  K briefs x 8 sentence tasks x N reruns x M cheap models.
  For each (brief, task, model), embed every output and compute cosine vs
  ground truth. sigma_cell = stdev of the N cosine values for that cell. We
  then pool sigma ACROSS briefs (mean/max) so the reported noise floor reflects
  the whole brief set, not one lucky/unlucky brief.

  Why multiple briefs: a σ measured on a single brief is like taking one
  person's temperature and declaring the whole class's normal range. Different
  briefs sit at different points on the cosine scale and have different
  run-to-run spread, so a one-brief σ can badly under- or over-state the real
  tie-band used to call winners.

Default: 5 briefs (evenly spaced across briefs.yml), 5 reruns, [haiku, gpt5mini].

Auto-skips any model that returns an access error on the first probe (so a
not-yet-propagated model doesn't burn 40 wasted calls).

Output:
  outputs/noise_floor_<timestamp>.json - full per-task data
  Console - summary table with recommended thresholds.

Cost: roughly $0.02 (40 Haiku + 40 mini calls + ~90 embeddings).
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from src import config as cfg
from src.evaluators import EmbeddingClient, _cosine
from src.llm_client import DONE_STATUSES, LLMClient, Status
from src.prompt_builder import (
    FULL_BRIEF,
    SENTENCE_TASKS,
    brief_id,
    build_prompt,
    load_briefs,
    load_prompts,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# Sample several briefs spread across the set rather than relying on one.
# Falls back gracefully if briefs.yml has fewer than this many.
N_BRIEFS = 5
PREFERRED_BRIEFS = ["Plonts"]  # keep one stable anchor for run-to-run comparison
RERUNS = 5
DEFAULT_MODELS = ["haiku", "gpt5mini"]
OUT_PATH = cfg.OUTPUTS_DIR / f"noise_floor_{datetime.now():%Y%m%d_%H%M%S}.json"


def select_briefs(all_briefs: list[dict], n: int) -> list[dict]:
    """Pick n briefs spread evenly across the set (plus any preferred anchors)."""
    by_id = {brief_id(b): b for b in all_briefs}
    chosen: list[dict] = []
    seen: set[str] = set()
    for name in PREFERRED_BRIEFS:
        if name in by_id and name not in seen:
            chosen.append(by_id[name])
            seen.add(name)
    remaining = [b for b in all_briefs if brief_id(b) not in seen]
    if remaining and len(chosen) < n:
        step = max(1, len(remaining) // max(1, (n - len(chosen))))
        for b in remaining[::step]:
            if len(chosen) >= n:
                break
            if brief_id(b) not in seen:
                chosen.append(b)
                seen.add(brief_id(b))
    return chosen[:n]


def probe_accessible(client: LLMClient, models: list[str]) -> list[str]:
    """One smoke call per model; drop the ones that fail."""
    accessible = []
    for m in models:
        r = client.call(
            model_key=m,
            prompt="ok",
            brief_id="probe",
            task="probe",
            config_id="probe",
            run_id=0,
        )
        if r.status in DONE_STATUSES:
            accessible.append(m)
            print(f"  [probe] {m:10s} ok")
        else:
            print(f"  [probe] {m:10s} skipped — {r.status}: {(r.error or '')[:80]}")
    return accessible


def main():
    briefs_all = load_briefs()
    sample = select_briefs(briefs_all, N_BRIEFS)
    if not sample:
        raise SystemExit("No briefs found in briefs.yml")
    brief_names = [brief_id(b) for b in sample]
    print(f"Measuring σ on {len(sample)} briefs={brief_names}, "
          f"{RERUNS} reruns × {len(SENTENCE_TASKS)} tasks")
    print(f"Budget cap: ${cfg.BUDGET_CAP['noise_floor']}\n")

    templates = load_prompts()
    client = LLMClient.from_env(budget_cap_usd=cfg.BUDGET_CAP["noise_floor"])
    embedder = EmbeddingClient(api_key=cfg.OPENAI_API_KEY)

    print("Probing model accessibility...")
    models = probe_accessible(client, DEFAULT_MODELS)
    if not models:
        raise SystemExit("No accessible models. Fix verify_models first.")
    print()

    # results[model][f"{brief}::{task}"] = {sigma, ...}
    results: dict[str, dict] = {}

    for model in models:
        results[model] = {}
        for brief in sample:
            bid = brief_id(brief)
            for task in SENTENCE_TASKS:
                prompt = build_prompt(task, FULL_BRIEF, brief, templates=templates)
                gt = brief[task]
                gt_emb = embedder.embed(gt)

                cosines = []
                outputs = []
                for run in range(1, RERUNS + 1):
                    r = client.call(
                        model_key=model,
                        prompt=prompt,
                        brief_id=bid,
                        task=task,
                        config_id="noise_floor",
                        run_id=run,
                    )
                    if r.status == Status.BUDGET_EXCEEDED:
                        print(f"\nBUDGET EXCEEDED at {model} {bid}/{task} run={run}. Stop.")
                        _finalize(results, client, sample, partial=True)
                        return
                    if r.status not in DONE_STATUSES:
                        print(f"  [skip] {model} {bid}/{task} run={run}: "
                              f"{r.status} {(r.error or '')[:80]}")
                        continue
                    out_emb = embedder.embed(r.raw_response)
                    cos = _cosine(out_emb, gt_emb)
                    cosines.append(cos)
                    outputs.append(r.raw_response[:120])
                    print(f"  {model:10s} {bid:12s} {task:18s} run={run} cos={cos:.4f}")

                cell_key = f"{bid}::{task}"
                if len(cosines) >= 2:
                    results[model][cell_key] = {
                        "brief": bid,
                        "task": task,
                        "n_runs": len(cosines),
                        "cosines": [round(c, 4) for c in cosines],
                        "outputs": outputs,
                        "mean_cosine": round(mean(cosines), 4),
                        "sigma": round(stdev(cosines), 4),
                    }
                else:
                    results[model][cell_key] = {
                        "brief": bid, "task": task,
                        "n_runs": len(cosines), "error": "insufficient data",
                    }
            print()

    _finalize(results, client, sample, partial=False)


def _finalize(results: dict, client: LLMClient, sample: list, partial: bool):
    """Aggregate σ across briefs, write JSON, print summary."""
    summary = {
        "_meta": {
            "briefs": [brief_id(b) for b in sample],
            "n_briefs": len(sample),
            "reruns": RERUNS,
            "tasks": SENTENCE_TASKS,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "partial": partial,
            "total_cost_usd": client.total_cost_usd,
            "total_calls": client.call_count,
        }
    }
    for model, by_cell in results.items():
        sigmas = [c["sigma"] for c in by_cell.values() if "sigma" in c]
        summary[model] = {
            "per_cell": by_cell,
        }
        if sigmas:
            summary[model]["aggregate"] = {
                "mean_sigma": round(mean(sigmas), 4),
                "max_sigma": round(max(sigmas), 4),
                "min_sigma": round(min(sigmas), 4),
                "n_cells": len(sigmas),
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    n_cells_possible = len(sample) * len(SENTENCE_TASKS)
    print("=" * 70)
    print(f"σ summary  ({len(sample)} briefs × {len(SENTENCE_TASKS)} tasks, "
          f"{RERUNS} reruns per cell)")
    print("=" * 70)
    for model, blk in summary.items():
        if model.startswith("_"):
            continue
        agg = blk.get("aggregate")
        if not agg:
            print(f"  {model:10s}  no data")
            continue
        print(
            f"  {model:10s}  mean σ = {agg['mean_sigma']:.4f}   "
            f"max σ = {agg['max_sigma']:.4f}   "
            f"({agg['n_cells']}/{n_cells_possible} cells)"
        )
    print("=" * 70)

    # Threshold recommendation: pool σ across briefs AND models. We report the
    # mean-of-means (typical spread) and the max (conservative). The noise floor
    # pasted into config should be ~2× a representative σ, NOT the single-brief
    # max, so it reflects the whole sample rather than one outlier cell.
    mean_sigmas = [
        blk.get("aggregate", {}).get("mean_sigma", 0)
        for k, blk in summary.items()
        if not k.startswith("_") and "aggregate" in blk
    ]
    all_max_sigmas = [
        blk.get("aggregate", {}).get("max_sigma", 0)
        for k, blk in summary.items()
        if not k.startswith("_") and "aggregate" in blk
    ]
    if mean_sigmas:
        sigma_typical = max(mean_sigmas)        # representative across models
        sigma_worst = max(all_max_sigmas)       # conservative
        print(f"\nNoise floor candidates (pooled over {len(sample)} briefs):")
        print(f"  typical σ (max of per-model mean σ): {sigma_typical:.4f}")
        print(f"  worst-cell σ:                        {sigma_worst:.4f}")
        print(f"  -> recommended NOISE_FLOOR_COSINE (2× typical σ): {2 * sigma_typical:.4f}")
        print(f"     paste this into src/config.py if it differs from the current value.")
        print(f"  practically equivalent (2σ):   cosine diff ≤ {2 * sigma_typical:.4f}")
        print(f"  significantly better   (5σ):   cosine diff ≥ {5 * sigma_typical:.4f}")
        print(f"  Reminder: σ is only a tie-band. Use the paired test for "
              f"'is A really better than B'.")

    print(f"\nCost spent:     ${client.total_cost_usd:.4f}")
    print(f"Calls made:     {client.call_count}")
    print(f"Full dump:      {OUT_PATH}")
    print(f"Embedding cache: {cfg.OUTPUTS_DIR / 'embedding_cache.jsonl'}")


if __name__ == "__main__":
    main()
