"""
Empirically measure the within-config noise floor sigma, used to set the
"practically equivalent" and "significantly better" decision thresholds.

Method:
  1 brief x 8 sentence tasks x N reruns x M cheap models.
  For each (task, model), embed every output and compute cosine vs ground
  truth. sigma_task_model = stdev of the N cosine values. Report mean / max
  sigma per model.

Default: Plonts brief, 5 reruns, [haiku, gpt5mini].

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


NOISE_BRIEF = "Plonts"
RERUNS = 5
DEFAULT_MODELS = ["haiku", "gpt5mini"]
OUT_PATH = cfg.OUTPUTS_DIR / f"noise_floor_{datetime.now():%Y%m%d_%H%M%S}.json"


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
    print(f"Measuring σ on brief='{NOISE_BRIEF}', {RERUNS} reruns × {len(SENTENCE_TASKS)} tasks")
    print(f"Budget cap: ${cfg.BUDGET_CAP['noise_floor']}\n")

    briefs = load_briefs()
    brief = next((b for b in briefs if brief_id(b) == NOISE_BRIEF), None)
    if brief is None:
        raise SystemExit(f"Brief '{NOISE_BRIEF}' not found in briefs.yml")

    templates = load_prompts()
    client = LLMClient.from_env(budget_cap_usd=cfg.BUDGET_CAP["noise_floor"])
    embedder = EmbeddingClient(api_key=cfg.OPENAI_API_KEY)

    print("Probing model accessibility...")
    models = probe_accessible(client, DEFAULT_MODELS)
    if not models:
        raise SystemExit("No accessible models. Fix verify_models first.")
    print()

    results: dict[str, dict] = {}

    for model in models:
        results[model] = {}
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
                    brief_id=NOISE_BRIEF,
                    task=task,
                    config_id="noise_floor",
                    run_id=run,
                )
                if r.status == Status.BUDGET_EXCEEDED:
                    print(f"\nBUDGET EXCEEDED at {model} {task} run={run}. Stop.")
                    _finalize(results, client, partial=True)
                    return
                if r.status not in DONE_STATUSES:
                    print(f"  [skip] {model} {task} run={run}: {r.status} {(r.error or '')[:80]}")
                    continue
                out_emb = embedder.embed(r.raw_response)
                cos = _cosine(out_emb, gt_emb)
                cosines.append(cos)
                outputs.append(r.raw_response[:120])
                print(f"  {model:10s} {task:18s} run={run} cos={cos:.4f}  out={r.raw_response[:60]!r}")

            if len(cosines) >= 2:
                results[model][task] = {
                    "n_runs": len(cosines),
                    "cosines": [round(c, 4) for c in cosines],
                    "outputs": outputs,
                    "mean_cosine": round(mean(cosines), 4),
                    "sigma": round(stdev(cosines), 4),
                }
            else:
                results[model][task] = {"n_runs": len(cosines), "error": "insufficient data"}
            print()

    _finalize(results, client, partial=False)


def _finalize(results: dict, client: LLMClient, partial: bool):
    """Aggregate σ, write JSON, print summary."""
    summary = {
        "_meta": {
            "brief": NOISE_BRIEF,
            "reruns": RERUNS,
            "tasks": SENTENCE_TASKS,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "partial": partial,
            "total_cost_usd": client.total_cost_usd,
            "total_calls": client.call_count,
        }
    }
    for model, by_task in results.items():
        sigmas = [t["sigma"] for t in by_task.values() if "sigma" in t]
        summary[model] = {
            "per_task": by_task,
        }
        if sigmas:
            summary[model]["aggregate"] = {
                "mean_sigma": round(mean(sigmas), 4),
                "max_sigma": round(max(sigmas), 4),
                "min_sigma": round(min(sigmas), 4),
                "n_tasks": len(sigmas),
            }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 70)
    print(f"σ summary  (brief={NOISE_BRIEF}, {RERUNS} reruns per task)")
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
            f"({agg['n_tasks']}/{len(SENTENCE_TASKS)} tasks)"
        )
    print("=" * 70)

    # Threshold recommendation: use the larger σ across models (conservative)
    all_max_sigmas = [
        blk.get("aggregate", {}).get("max_sigma", 0)
        for k, blk in summary.items()
        if not k.startswith("_") and "aggregate" in blk
    ]
    if all_max_sigmas:
        sigma_used = max(all_max_sigmas)
        print(f"\nRecommended thresholds (using max sigma across models = {sigma_used:.4f}):")
        print(f"  practically equivalent (2σ):   cosine diff ≤ {2 * sigma_used:.4f}")
        print(f"  significantly better   (5σ):   cosine diff ≥ {5 * sigma_used:.4f}")
        print(f"  hard floor              :   cosine diff ≥ 0.05")

    print(f"\nCost spent:     ${client.total_cost_usd:.4f}")
    print(f"Calls made:     {client.call_count}")
    print(f"Full dump:      {OUT_PATH}")
    print(f"Embedding cache: {cfg.OUTPUTS_DIR / 'embedding_cache.jsonl'}")


if __name__ == "__main__":
    main()
