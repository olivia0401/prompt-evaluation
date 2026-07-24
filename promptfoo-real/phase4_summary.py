"""
Phase 4 分析:贵模型到底值不值?
每个任务的推荐配方,6 个模型各跑 3 brief,比较便宜 vs 贵模型的平均分。
差距 < noise floor(0.036)= 贵模型不值,选便宜的。
"""
import json, sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NOISE = 0.036
TIER = {  # 模型 → 价位
 "gpt-5-mini": "cheap", "haiku": "cheap",
 "gpt-5-2025": "medium", "sonnet": "medium",
 "gpt-5.5": "premium", "opus": "premium",
}
def tier_of(mid):
    for k, t in TIER.items():
        if k in mid: return t
    return "?"

TASKS = ["concept","position","emotion","function","benefit","category","feature","context"]
print(f"{'任务':<9}{'便宜最佳':>9}{'中档最佳':>9}{'贵最佳':>9}{'贵-便宜':>9}  贵模型值吗")
print("-" * 68)

worth = 0
for task in TASKS:
    d = json.load(open(f"results/phase4-{task}.json", encoding="utf-8"))
    by_tier = defaultdict(list)
    for r in d["results"]["results"]:
        mid = r["provider"]["id"]
        if r["score"] is not None:
            by_tier[tier_of(mid)].append(r["score"])
    best = {t: (max(v) if v else 0) for t, v in by_tier.items()}
    cheap, med, prem = best.get("cheap",0), best.get("medium",0), best.get("premium",0)
    gap = prem - cheap
    yes = gap >= NOISE
    if yes: worth += 1
    print(f"{task:<9}{cheap:>9.3f}{med:>9.3f}{prem:>9.3f}{gap:>+9.3f}  "
          f"{'值(gap≥噪声)' if yes else '不值(打平→省)'}")

print("-" * 68)
print(f"\n贵模型在 {worth}/8 个任务上显著优于便宜模型(gap ≥ noise floor 0.036)")
print("原版 Phase 4 结论:premium 基本不值,只 feature 任务有微弱信号")
