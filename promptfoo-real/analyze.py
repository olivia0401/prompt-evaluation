"""
第 3 步:用你的统计判断 promptfoo 的分数。

promptfoo 只给平均分,不告诉你"差距是真的还是噪声"。这一步接上你项目里
已有的判断规则(src/config.py 的 NOISE_FLOOR_COSINE=0.036 和 audit_data.py
D4 的配对符号检验),回答:top-1 配方是真的赢,还是和 top-2 打平?

只读 promptfoo 的结果数据库,不调 API、不花钱。
"""
import os, json, sqlite3, sys
from collections import defaultdict
from scipy.stats import binomtest, wilcoxon

# Windows 终端默认 GBK 编码不了 emoji,强制 UTF-8(你项目里 analyze/audit 也这么做)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# —— 你项目里定义好的规则,直接沿用 ——
NOISE_FLOOR = 0.036   # src/config.py:207  差距小于它 = 平手
ALPHA       = 0.05    # src/config.py PAIRED_TEST_ALPHA

db = os.path.expanduser("~/.promptfoo/promptfoo.db")
con = sqlite3.connect(db); con.row_factory = sqlite3.Row
ev = con.execute("SELECT id FROM evals ORDER BY created_at DESC LIMIT 1").fetchone()["id"]

# 每个配方在每个 brief 上的分数:  recipe -> {brief_idx: score}
by_recipe = defaultdict(dict)
for r in con.execute(
    "SELECT prompt, test_idx, score FROM eval_results WHERE eval_id=? AND score IS NOT NULL", (ev,)):
    label = json.loads(r["prompt"]).get("label", "?")
    by_recipe[label][r["test_idx"]] = r["score"]

means = {k: sum(v.values())/len(v) for k, v in by_recipe.items()}
ranked = sorted(means, key=lambda k: -means[k])
win, runner = ranked[0], ranked[1]

print(f"eval: {ev}\n")
print("配方平均分(高→低):")
for k in ranked:
    print(f"  {means[k]:.3f}  {k}")

# —— top-1 vs top-2:同样的 brief 上逐个配对 ——
briefs = sorted(set(by_recipe[win]) & set(by_recipe[runner]))
a = [by_recipe[win][b]    for b in briefs]
b = [by_recipe[runner][b] for b in briefs]
diffs = [x - y for x, y in zip(a, b)]
wins   = sum(1 for d in diffs if d > 0)
losses = sum(1 for d in diffs if d < 0)
gap = means[win] - means[runner]

sign_p = binomtest(min(wins, losses), wins + losses, 0.5).pvalue if (wins + losses) else 1.0
wil_p  = wilcoxon(a, b).pvalue if len(briefs) >= 6 and any(diffs) else float("nan")

print(f"\n对决:  {win}   vs   {runner}")
print(f"  平均分差 gap        = {gap:.3f}   (noise floor = {NOISE_FLOOR})")
print(f"  逐 brief 胜/负       = {wins} 赢 / {losses} 负  (共 {len(briefs)} 个 brief)")
print(f"  符号检验 p          = {sign_p:.3f}   (alpha = {ALPHA})")
print(f"  Wilcoxon p          = {wil_p:.3f}")

within_noise = gap < NOISE_FLOOR
not_sig      = sign_p >= ALPHA
print("\n结论:")
if within_noise or not_sig:
    why = []
    if within_noise: why.append(f"gap {gap:.3f} < noise floor {NOISE_FLOOR}")
    if not_sig:      why.append(f"符号检验 p {sign_p:.2f} ≥ {ALPHA}")
    print(f"  ⚖️  平手（{' 且 '.join(why)}）")
    print(f"  → 平均分说 '{win}' 赢,但统计说分不出来。")
    print(f"  → 按你的决策规则(平手时选更省的):选 {win}")
    print(f"     不是因为它更准,而是打平之下它字段更少、prompt 更短、更便宜。")
else:
    print(f"  ✅  {win} 是真的赢(gap ≥ noise floor 且符号检验显著)")
