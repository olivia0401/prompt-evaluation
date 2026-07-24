"""
生成完整版 config:17 个配方(单字段 + 字段组合 + 整份 + 空白)× 2 个便宜模型。
对应你原来实验的 Stage A(便宜模型广筛全配方)。
贵模型验证(Phase 4)之后单独对赢家跑,不在这里全交叉,避免烧钱。
"""
import itertools
from pathlib import Path
HERE = Path(__file__).resolve().parent

# 每个任务:标准答案字段, 中性化指令
TASKS = {
 "concept":  ("concept_relevant",
    "identify the single most important idea — the one that, if removed, would "
    "make the brand unrecognizable — and distill it into one sentence, 10-20 words"),
 "position": ("position_relevant",
    "summarize how the company seeks to be positioned against its competitors — "
    "in terms of price, proposition, or promise — in one sentence, 15-25 words"),
 "emotion":  ("emotion_relevant",
    "distill the feelings or emotions the company seeks to spur in its audience "
    "into one sentence — describe feelings, not features — 15-25 words"),
 "function": ("function_relevant",
    "summarize what the product or service is, what it does and how it works — "
    "its function, not its benefits — in one sentence, 15-25 words"),
 "benefit":  ("benefit_relevant",
    "distill the tangible and intangible benefits of using the product or "
    "service into one sentence, 15-25 words"),
 "category": ("category_relevant",
    "summarize the category the product or service operates within, in one "
    "sentence, 15-20 words"),
 "feature":  ("feature_relevant",
    "capture the most important and consequential features of the product or "
    "service in one sentence, 15-25 words"),
 "context":  ("context_relevant",
    "summarize the social, physical and environmental context(s) in which the "
    "product or service exists, in one sentence, 15-25 words"),
}

# 5 个语义字段 → 自动生成 17 个配方
SEMANTIC = ["product", "differentiators", "audience", "brand_strategy", "personality"]

def field_block(fields):
    return "\n      ".join(f"{f.replace('_',' ').title()}: {{{{{f}}}}}" for f in fields)

FIELDS = [("1-full-brief", SEMANTIC)]                       # 整份简报
for i, f in enumerate(SEMANTIC):                            # 5 个单字段
    FIELDS.append((f"single-{f}", [f]))
for a, b in itertools.combinations(SEMANTIC, 2):           # 10 个字段对
    FIELDS.append((f"pair-{a}+{b}", [a, b]))
FIELDS.append(("baseline-no-context", None))               # 空白基线
# = 1 + 5 + 10 + 1 = 17 个配方

# Stage A:2 个便宜模型(你 plan 里的 cheap tier)
PROVIDERS = ["openai:gpt-5-mini-2025-08-07",
             "anthropic:claude-haiku-4-5-20251001"]

for task, (gt_field, instr) in TASKS.items():
    lines = [f'description: "{task.title()} — 完整版(17配方 x 2便宜模型)"', "", "prompts:"]
    for label, fields in FIELDS:
        if fields is None:
            raw = (f"You are a brand strategy consultant. Based on general brand "
                   f"knowledge, {instr}. Write only the sentence, no preamble, no quotes.")
        else:
            raw = (f"You are a brand strategy consultant. From the company "
                   f"information below, {instr}. Write only the sentence, no "
                   f"preamble, no quotes.\n\n      {field_block(fields)}")
        lines.append(f'  - label: "{label}"')
        lines.append(f"    raw: |")
        for ln in raw.split("\n"):
            lines.append(f"      {ln}")
    lines.append("")
    lines.append("providers:")
    for p in PROVIDERS:
        lines.append(f"  - {p}")
    lines += ["", "defaultTest:", "  assert:", "    - type: similar",
              f'      value: "{{{{{gt_field}}}}}"', "      threshold: 0.7", "",
              "tests: file://tests.yaml", ""]
    (HERE / f"{task}.yaml").write_text("\n".join(lines), encoding="utf-8")
    print(f"生成 {task}.yaml  (17 配方 x 2 模型)")

print(f"\n完成。每个任务 17 配方 x 2 模型 x 23 brief = {17*2*23} 次调用")
print(f"8 个任务合计约 {17*2*23*8} 次(Stage A 规模,和你原来 ~6532 次一致)")
