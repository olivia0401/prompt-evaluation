"""Stage-A summary: pick each task's winner under the tie->cheapest rule and
compare against the custom pipeline's published conclusions.

Two cheap-model scores per (recipe, brief) are averaged. A recipe within the
noise floor of the top score joins the tie pool; the recommendation is the
cheapest (fewest-field) recipe in that pool — the same decision rule as
../src/config.py.
"""
import json
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOISE = 0.036
SEMANTIC = ["product", "differentiators", "audience", "brand_strategy", "personality"]

# custom pipeline's winning field set per task (from Results/*.xlsx)
ORIG = {
    "concept":  {"brand_strategy"},
    "position": {"audience", "differentiators"},
    "emotion":  {"brand_strategy", "differentiators"},
    "function": {"product"},
    "benefit":  {"audience", "differentiators"},
    "category": {"product"},
    "feature":  {"product"},
    "context":  {"audience"},
}


def fields_of(label):
    if "full" in label:
        return set(SEMANTIC)
    if "baseline" in label:
        return set()
    return set(label.split("+"))


print(f"{'task':<9}{'promptfoo (tie->cheapest)':<26}{'verdict':<15}{'pool':<6}| custom winner | match")
print("-" * 92)

tie_count = agree = 0
for task, orig_fields in ORIG.items():
    data = json.load(open(f"results/{task}.json", encoding="utf-8"))["results"]
    labels = [p.get("label", f"p{i}") for i, p in enumerate(data["prompts"])]

    cell = defaultdict(list)
    for r in data["results"]:
        cell[(r["promptIdx"], r["testIdx"])].append(r["score"])
    by = defaultdict(dict)
    for (pi, ti), scores in cell.items():
        by[labels[pi]][ti] = sum(scores) / len(scores)

    means = {lab: sum(v.values()) / len(v) for lab, v in by.items()}
    ranked = sorted(means, key=lambda k: -means[k])
    top = means[ranked[0]]
    pool = [l for l in ranked if top - means[l] < NOISE and "baseline" not in l]
    winner = min(pool, key=lambda l: (len(fields_of(l)), -means[l]))
    tie = len(pool) > 1
    tie_count += tie
    match = fields_of(winner) == orig_fields
    agree += match

    orig_name = "+".join(sorted(orig_fields)) or "baseline"
    print(f"{task:<9}{winner:<26}{'tie->cheapest' if tie else 'clear win':<15}"
          f"{len(pool):<6}| {orig_name:<13} | {'yes' if match else '~'}")

print("-" * 92)
print(f"\ncustom:     8/8 tasks tied -> choose by cost")
print(f"promptfoo:  {tie_count}/8 tasks tied")
print(f"winner field-set exact match: {agree}/8")
