"""Phase-4: are premium models worth it? Compare best cheap vs best premium
score per task; a gap below the noise floor means premium is not worth it.
"""
import json
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOISE = 0.036
TIER = {"gpt-5-mini": "cheap", "haiku": "cheap",
        "gpt-5-2025": "medium", "sonnet": "medium",
        "gpt-5.5": "premium", "opus": "premium"}

TASKS = ["concept", "position", "emotion", "function",
         "benefit", "category", "feature", "context"]


def tier(model_id):
    return next((t for k, t in TIER.items() if k in model_id), "?")


print(f"{'task':<9}{'cheap':>8}{'medium':>8}{'premium':>9}{'prem-cheap':>12}  premium worth it?")
print("-" * 66)

worth = 0
for task in TASKS:
    rows = json.load(open(f"results/phase4-{task}.json", encoding="utf-8"))["results"]["results"]
    best = defaultdict(float)
    for r in rows:
        if r["score"] is not None:
            best[tier(r["provider"]["id"])] = max(best[tier(r["provider"]["id"])], r["score"])
    gap = best["premium"] - best["cheap"]
    yes = gap >= NOISE
    worth += yes
    print(f"{task:<9}{best['cheap']:>8.3f}{best['medium']:>8.3f}{best['premium']:>9.3f}"
          f"{gap:>+12.3f}  {'yes' if yes else 'no (tie -> cheaper)'}")

print("-" * 66)
print(f"\npremium beats cheap on {worth}/8 tasks (gap >= noise floor {NOISE})")
