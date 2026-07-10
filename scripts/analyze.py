"""
Analysis pipeline.
Pattern adapted from 毕设/250038039/experiments/analyze_final_results.py
(matplotlib subplots + auto-label bars + groupby/agg/rename pattern).

Reads:
  outputs/results.jsonl   (raw call results — produced by run_experiment.py)
  outputs/scored.csv      (per-call metrics — produced by the scoring pipeline below)

Writes:
  outputs/scored.csv               (if running --score)
  outputs/summary_by_config.csv    (mean / std cosine per task × config_id × model)
  outputs/cost_summary.csv         (cost per stage / model / phase)
  outputs/plots/*.png              (bar charts + heatmap)

Usage:
  python -m scripts.analyze --counts     # quick: status / token / cost rollup from JSONL
  python -m scripts.analyze --score      # run evaluators on raw outputs -> scored.csv
  python -m scripts.analyze --summarize  # build summary_by_config.csv + plots
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src import config as cfg
from src.utils import read_jsonl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_JSONL = cfg.OUTPUTS_DIR / "results.jsonl"
SCORED_CSV = cfg.OUTPUTS_DIR / "scored.csv"
SUMMARY_CSV = cfg.OUTPUTS_DIR / "summary_by_config.csv"
COST_CSV = cfg.OUTPUTS_DIR / "cost_summary.csv"
PLOTS_DIR = cfg.OUTPUTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DONE_STATUSES = {"ok", "ok_length_violation"}


# ---------- counts / cost rollup (works on JSONL alone, no scoring needed) ----------

def cmd_counts():
    """Quick status / token / cost rollup. Use this anytime to sanity-check."""
    records = read_jsonl(RESULTS_JSONL)
    if not records:
        print(f"No data in {RESULTS_JSONL}")
        return

    df = pd.DataFrame(records)
    print(f"\n=== {len(df)} total records ===\n")

    # Status
    print("Status:")
    print(df["status"].value_counts().to_string())

    # Cost / tokens
    total_cost = df["cost_usd"].sum() if "cost_usd" in df else 0
    print(f"\nTotal cost: ${total_cost:.4f}")
    if "input_tokens" in df:
        print(f"Input tokens:      {int(df['input_tokens'].sum()):,}")
        print(f"Output tokens:     {int(df['output_tokens'].sum()):,}")
    if "reasoning_tokens" in df:
        r = int(df["reasoning_tokens"].sum())
        o = int(df["output_tokens"].sum()) or 1
        print(f"Reasoning tokens:  {r:,}  ({100*r/o:.1f}% of output)")

    # Per model
    if "model_key" in df:
        print("\nBy model:")
        by_model = df.groupby("model_key").agg(
            calls=("status", "count"),
            ok=("status", lambda s: s.isin(DONE_STATUSES).sum()),
            cost=("cost_usd", "sum"),
            avg_in=("input_tokens", "mean"),
            avg_out=("output_tokens", "mean"),
        ).round({"cost": 4, "avg_in": 0, "avg_out": 0})
        print(by_model.to_string())

    # Cost summary CSV
    if "model_key" in df:
        cost_summary = df.groupby("model_key").agg(
            calls=("status", "count"),
            ok=("status", lambda s: s.isin(DONE_STATUSES).sum()),
            cost_usd=("cost_usd", "sum"),
        ).reset_index()
        cost_summary.to_csv(COST_CSV, index=False)
        print(f"\n-> {COST_CSV}")


# ---------- summarize by config (needs scored.csv) ----------

def cmd_summarize():
    """Build per-(task, config_id, model) cosine summary + plots."""
    if not SCORED_CSV.exists():
        raise SystemExit(
            f"Missing {SCORED_CSV}. Run scoring first ("
            f"python -m scripts.analyze --score)."
        )
    scored = pd.read_csv(SCORED_CSV)

    # Pattern from 毕设/analyze_final_results.py: groupby + agg + rename
    summary = scored.groupby(["task", "config_id", "model_key"]).agg(
        n=("cosine", "count"),
        mean_cosine=("cosine", "mean"),
        std_cosine=("cosine", "std"),
        mean_rouge_l=("rouge_l", "mean"),
        length_ok_pct=("length_compliant", lambda s: 100 * s.mean()),
        worst_cosine=("cosine", "min"),  # worst-case score across briefs
    ).round(4).reset_index()
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"-> {SUMMARY_CSV} ({len(summary)} rows)")

    plot_per_task_best(summary)
    plot_task_x_field_heatmap(summary)


# ---------- plotting ----------

def plot_per_task_best(summary: pd.DataFrame) -> None:
    """For each task, bar-chart of mean cosine across configs (model fixed = best cheap)."""
    import matplotlib.pyplot as plt

    tasks = sorted(summary["task"].unique())
    if not tasks:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for ax, task in zip(axes, tasks):
        sub = summary[summary["task"] == task].sort_values("mean_cosine", ascending=False).head(10)
        bars = ax.bar(
            range(len(sub)),
            sub["mean_cosine"],
            yerr=sub["std_cosine"],
            color="#3498db",
            alpha=0.75,
            edgecolor="black",
        )
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["config_id"], rotation=45, ha="right", fontsize=7)
        ax.set_title(task, fontsize=10)
        ax.set_ylabel("mean cosine")
        ax.grid(axis="y", alpha=0.3)
        # Pattern from 毕设: auto-label bars
        for bar, val in zip(bars, sub["mean_cosine"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

    for ax in axes[len(tasks):]:
        ax.axis("off")

    plt.tight_layout()
    out = PLOTS_DIR / "per_task_top_configs.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


def plot_task_x_field_heatmap(summary: pd.DataFrame) -> None:
    """Tab 2 source: task (rows) × config_id (cols) heatmap of mean cosine."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Pick the cheaper model for the main matrix; medium-model results go in a side panel.
    cheap_models = [k for k, v in cfg.MODELS.items() if v["tier"] == "cheap"]
    sub = summary[summary["model_key"].isin(cheap_models)]
    if sub.empty:
        return

    pivot = sub.groupby(["task", "config_id"])["mean_cosine"].mean().unstack()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.7), 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Task × Config — mean cosine (cheap models)")

    # Cell value labels
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(im, ax=ax, label="cosine")
    plt.tight_layout()
    out = PLOTS_DIR / "task_x_config_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"-> {out}")


# ---------- scoring ----------

def cmd_score():
    """
    Score every ok / ok_length_violation row in results.jsonl.
    Writes outputs/scored.csv.

    Failed rows (status != ok*) appear in scored.csv with metric columns blank
    Failures should NOT be coerced to a 0 score — they're excluded, not "very low".
    """
    from src.evaluators import (
        EmbeddingClient,
        parse_keywords,
        score_keywords,
        score_sentence,
    )
    from src.llm_client import DONE_STATUSES
    from src.prompt_builder import (
        KEYWORD_TASK,
        SENTENCE_TASKS,
        brief_id,
        load_briefs,
    )

    if not RESULTS_JSONL.exists():
        raise SystemExit(f"Missing {RESULTS_JSONL}. Run the experiment first.")

    records = read_jsonl(RESULTS_JSONL)
    print(f"Loaded {len(records)} records from {RESULTS_JSONL}")

    briefs = {brief_id(b): b for b in load_briefs()}
    print(f"Indexed {len(briefs)} briefs")

    embedder = EmbeddingClient(api_key=cfg.OPENAI_API_KEY)
    print(f"Embedding cache: {embedder.cache_stats()}")

    # --- Batch pre-warm the embedding cache ---
    # The original loop called embedder.embed() twice per sentence row (pred + GT).
    # At 9,806 rows × 2 calls × ~250ms HTTP latency = ~30 min serial. We pre-fetch
    # every unique text once via embed_batch (100/chunk = single HTTP call per
    # chunk) so the main loop just hits the local cache. Cache stays at
    # outputs/embedding_cache.jsonl — repeated runs cost $0.
    unique_texts: set[str] = set()
    for r in records:
        if r.get("status") not in DONE_STATUSES:
            continue
        if r.get("task") not in SENTENCE_TASKS:
            continue
        brief = briefs.get(r.get("brief_id"))
        if brief is None:
            continue
        pred = (r.get("raw_response") or "").strip()
        if pred:
            unique_texts.add(pred)
        gt = (brief.get(r.get("task"), "") or "").strip()
        if gt:
            unique_texts.add(gt)

    to_fetch = [t for t in unique_texts if embedder._cache_key(t) not in embedder._cache]
    print(f"Pre-warm: {len(unique_texts)} unique texts, {len(to_fetch)} not yet cached.")
    if to_fetch:
        BATCH = 100
        for i in range(0, len(to_fetch), BATCH):
            chunk = to_fetch[i:i + BATCH]
            embedder.embed_batch(chunk)
            done = min(i + BATCH, len(to_fetch))
            print(f"  pre-warm [{done}/{len(to_fetch)}] ({100*done/len(to_fetch):.0f}%)")
    print(f"Pre-warm done. Cache now: {embedder.cache_stats()['cached_items']} items.")

    out_rows = []
    counts = {"scored_sentence": 0, "scored_keyword": 0, "skipped_status": 0,
              "skipped_empty": 0, "errors": 0}

    PROGRESS_EVERY = 1000
    for row_idx, r in enumerate(records):
        if row_idx and row_idx % PROGRESS_EVERY == 0:
            print(f"  scoring [{row_idx}/{len(records)}] {counts}")
        # Common row scaffold (metrics blank by default)
        row = {
            "brief_id": r.get("brief_id"),
            "task": r.get("task"),
            "config_id": r.get("config_id"),
            "model_key": r.get("model_key"),
            "run_id": r.get("run_id"),
            "status": r.get("status"),
            "finish_reason": r.get("finish_reason"),
            "input_tokens": r.get("input_tokens", 0),
            "output_tokens": r.get("output_tokens", 0),
            "reasoning_tokens": r.get("reasoning_tokens", 0),
            "cost_usd": r.get("cost_usd", 0),
            "prediction": r.get("raw_response") or "",
            "prediction_parsed": "",
            "ground_truth": "",
            "cosine": None,
            "rouge_l": None,
            "word_count": None,
            "length_compliant": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "pred_count": None,
            "gt_count": None,
            "parse_errors": "",
        }

        if r.get("status") not in DONE_STATUSES:
            counts["skipped_status"] += 1
            out_rows.append(row)
            continue

        brief = briefs.get(r.get("brief_id"))
        if brief is None:
            row["parse_errors"] = f"brief not found: {r.get('brief_id')}"
            counts["errors"] += 1
            out_rows.append(row)
            continue

        task = r.get("task")
        prediction = (r.get("raw_response") or "").strip()

        if not prediction:
            row["parse_errors"] = "empty prediction"
            counts["skipped_empty"] += 1
            out_rows.append(row)
            continue

        try:
            if task in SENTENCE_TASKS:
                gt = brief.get(task, "") or ""
                row["ground_truth"] = gt
                if not gt.strip():
                    row["parse_errors"] = "missing GT"
                    counts["errors"] += 1
                else:
                    s = score_sentence(prediction, gt, task, embedder)
                    row["cosine"] = round(s.cosine, 6) if s.cosine is not None else None
                    row["rouge_l"] = round(s.rouge_l, 6)
                    row["word_count"] = s.word_count
                    row["length_compliant"] = s.length_compliant
                    counts["scored_sentence"] += 1
            elif task == KEYWORD_TASK:
                gt = brief.get("keywords", []) or []
                row["ground_truth"] = json.dumps(gt, ensure_ascii=False)
                terms, perrs = parse_keywords(prediction)
                row["prediction_parsed"] = json.dumps(terms, ensure_ascii=False)
                if perrs:
                    row["parse_errors"] = "; ".join(perrs)
                if not terms:
                    counts["errors"] += 1
                else:
                    s = score_keywords(terms, gt)
                    row["precision"] = round(s.precision, 6)
                    row["recall"] = round(s.recall, 6)
                    row["f1"] = round(s.f1, 6)
                    row["pred_count"] = s.pred_count
                    row["gt_count"] = s.gt_count
                    counts["scored_keyword"] += 1
            else:
                row["parse_errors"] = f"unknown task: {task}"
                counts["errors"] += 1
        except Exception as e:
            row["parse_errors"] = f"{type(e).__name__}: {e}"[:200]
            counts["errors"] += 1

        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    df.to_csv(SCORED_CSV, index=False, encoding="utf-8")

    print(f"\nScoring complete. Counts: {counts}")
    print(f"-> {SCORED_CSV}")

    # PASS / FAIL summary against Phase 0 gating criteria
    sent = df[df["cosine"].notna()]
    kw = df[df["f1"].notna()]
    all_done = df[df["status"].isin(["ok", "ok_length_violation"])]

    print("\n=== Quick stats ===")
    if len(sent) > 0:
        print(f"Sentence tasks scored: {len(sent)}")
        print(f"  mean cosine:           {sent['cosine'].mean():.4f}")
        print(f"  mean rouge_l:          {sent['rouge_l'].mean():.4f}")
        print(f"  length compliant rate: {sent['length_compliant'].mean()*100:.1f}%")
    if len(kw) > 0:
        print(f"Keyword task scored: {len(kw)}")
        print(f"  mean F1:        {kw['f1'].mean():.4f}")
        print(f"  mean precision: {kw['precision'].mean():.4f}")
        print(f"  mean recall:    {kw['recall'].mean():.4f}")
        kw_attempts = df[df["task"] == "keywords"]
        kw_parse_fail = kw_attempts[kw_attempts["prediction_parsed"] == "[]"]
        print(f"  keyword parse-fail rate: {len(kw_parse_fail)}/{len(kw_attempts)}")

    print(f"\nAPI parse success rate: {len(all_done)}/{len(df)} = {len(all_done)/len(df)*100:.1f}%")
    print("  Phase 0 PASS/FAIL gates:")
    print(f"    API parse success > 95%:   {'PASS' if len(all_done)/len(df)*100 > 95 else 'FAIL'}")
    if len(sent) > 0:
        comp = sent['length_compliant'].mean() * 100
        print(f"    Length compliance > 90%:   {'PASS' if comp > 90 else 'FAIL'}  ({comp:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true", help="Quick rollup from JSONL")
    ap.add_argument("--score", action="store_true", help="Run evaluators -> scored.csv")
    ap.add_argument("--summarize", action="store_true", help="Build summary_by_config.csv + plots")
    args = ap.parse_args()

    if args.counts:
        cmd_counts()
    if args.score:
        cmd_score()
    if args.summarize:
        cmd_summarize()
    if not any([args.counts, args.score, args.summarize]):
        ap.print_help()


if __name__ == "__main__":
    main()
