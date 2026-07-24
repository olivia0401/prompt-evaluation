"""Single-task statistical check on the latest promptfoo eval.

promptfoo reports scores; it does not decide whether a gap is real. This reads
the most recent eval from promptfoo's DB and applies the noise floor + paired
sign test (same rule as ../src/config.py) to the top-2 recipes.
"""
import json
import os
import sqlite3
import sys
from collections import defaultdict
from scipy.stats import binomtest, wilcoxon

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOISE, ALPHA = 0.036, 0.05

db = os.path.expanduser("~/.promptfoo/promptfoo.db")
con = sqlite3.connect(db)
con.row_factory = sqlite3.Row
eval_id = con.execute("SELECT id FROM evals ORDER BY created_at DESC LIMIT 1").fetchone()["id"]

by_recipe = defaultdict(dict)
for r in con.execute(
        "SELECT prompt, test_idx, score FROM eval_results "
        "WHERE eval_id=? AND score IS NOT NULL", (eval_id,)):
    label = json.loads(r["prompt"]).get("label", "?")
    by_recipe[label][r["test_idx"]] = r["score"]

means = {k: sum(v.values()) / len(v) for k, v in by_recipe.items()}
ranked = sorted(means, key=lambda k: -means[k])
win, runner = ranked[0], ranked[1]

print(f"eval: {eval_id}\n")
for k in ranked:
    print(f"  {means[k]:.3f}  {k}")

briefs = sorted(set(by_recipe[win]) & set(by_recipe[runner]))
a = [by_recipe[win][b] for b in briefs]
b = [by_recipe[runner][b] for b in briefs]
diffs = [x - y for x, y in zip(a, b)]
wins = sum(d > 0 for d in diffs)
losses = sum(d < 0 for d in diffs)
gap = means[win] - means[runner]
sign_p = binomtest(min(wins, losses), wins + losses, 0.5).pvalue if wins + losses else 1.0
wil_p = wilcoxon(a, b).pvalue if len(briefs) >= 6 and any(diffs) else float("nan")

print(f"\n{win} vs {runner}")
print(f"  gap {gap:.3f} (noise floor {NOISE}) | {wins}W/{losses}L | "
      f"sign p={sign_p:.3f} | wilcoxon p={wil_p:.3f}")
if gap < NOISE or sign_p >= ALPHA:
    print("  -> tie: nominal winner but not statistically separable; pick cheaper")
else:
    print(f"  -> {win} is a real win")
