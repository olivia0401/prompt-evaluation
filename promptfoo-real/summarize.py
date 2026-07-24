"""
汇总完整版(17配方 x 2便宜模型)全部 8 个任务,和原来 13146 行版本并排对比。
每个配方在每个 brief 上取两个模型的平均分,再按你的 noise floor + 配对检验判断。
"""
import json, sys
from collections import defaultdict
from scipy.stats import binomtest

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOISE, ALPHA = 0.036, 0.05
SEMANTIC = ["product", "differentiators", "audience", "brand_strategy", "personality"]

# 原版(gpt5mini,142配方)每个任务的赢家字段集合 + 判定
ORIG_FIELDS = {
 "concept":  ({"brand_strategy"},               "Tie: choose cheaper"),
 "position": ({"audience", "differentiators"},  "Tie: choose cheaper"),
 "emotion":  ({"brand_strategy", "differentiators"}, "Tie: choose cheaper"),
 "function": ({"product"},                      "Tie: choose cheaper"),
 "benefit":  ({"audience", "differentiators"},  "Tie: choose cheaper"),
 "category": ({"product"},                      "Tie: choose cheaper"),
 "feature":  ({"product"},                      "Tie: choose cheaper"),
 "context":  ({"audience"},                     "Tie: choose cheaper"),
}

def fields_of(label):
    if "full" in label:     return set(SEMANTIC)
    if "baseline" in label: return set()
    if label.startswith("single-"): return {label[len("single-"):]}
    if label.startswith("pair-"):   return set(label[len("pair-"):].split("+"))
    return set()

TASKS = list(ORIG_FIELDS)
print(f"{'任务':<9}{'promptfoo推荐(打平选省)':<30}{'判定':<10}{'池':<5} | 原版赢家 | 一致")
print("-" * 96)

tie_count = agree = 0
for task in TASKS:
    d = json.load(open(f"results/{task}.json", encoding="utf-8"))
    R = d["results"]
    labels = [p.get("label", f"p{i}") for i, p in enumerate(R["prompts"])]
    # 每配方每brief:两模型平均
    cell = defaultdict(list)
    for r in R["results"]:
        cell[(r["promptIdx"], r["testIdx"])].append(r["score"])
    by = defaultdict(dict)   # label -> {brief: mean_score}
    for (pi, ti), scores in cell.items():
        by[labels[pi]][ti] = sum(scores) / len(scores)

    means = {lab: sum(v.values())/len(v) for lab, v in by.items()}
    ranked = sorted(means, key=lambda k: -means[k])
    top = means[ranked[0]]
    # 你的决策规则:和最高分打平(gap<noise floor)的配方组成"打平池",
    # 在池里选最省的(字段最少);baseline(空)不算可用配方。
    tie_pool = [l for l in ranked if (top - means[l]) < NOISE and "baseline" not in l]
    w = min(tie_pool, key=lambda l: (len(fields_of(l)), -means[l]))   # 最省的
    run = ranked[1] if ranked[1] != w else ranked[0]
    briefs = sorted(set(by[ranked[0]]) & set(by[ranked[1]]))
    diffs = [by[ranked[0]][b] - by[ranked[1]][b] for b in briefs]
    wins = sum(1 for x in diffs if x > 0)
    losses = sum(1 for x in diffs if x < 0)
    gap = top - means[ranked[1]]
    p = binomtest(min(wins, losses), wins+losses, 0.5).pvalue if (wins+losses) else 1.0
    tie = len(tie_pool) > 1
    if tie: tie_count += 1

    orig_set, orig_call = ORIG_FIELDS[task]
    same = fields_of(w) == orig_set
    if same: agree += 1

    orig_name = "+".join(sorted(orig_set)) if orig_set else "baseline"
    print(f"{task:<9}{w:<30}{'打平→选省' if tie else '真赢':<10}{'池'+str(len(tie_pool)):<5} | "
          f"{orig_name:<24} {'✓' if same else '≈'}")

print("-" * 96)
print(f"\n元结论对比:")
print(f"  原版:      8/8 任务 'Tie: choose cheaper'(全打平,按成本选)")
print(f"  promptfoo: {tie_count}/8 任务 top1 vs top2 打平")
print(f"  赢家配方(字段集合)完全一致: {agree}/8")
