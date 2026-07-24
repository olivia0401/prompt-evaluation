"""
Phase 4:对每个任务的推荐配方,用全部 6 个模型 × 3 个 brief 跑,
看贵模型(Opus/GPT-5.5/Sonnet/GPT-5)是否显著优于便宜模型。
对应你 plan 的 Phase 4(premium ladder,cap £1)。
"""
import yaml
from pathlib import Path
HERE = Path(__file__).resolve().parent

# Stage A 统计出的每任务推荐配方(单字段)
REC = {
 "concept": "brand_strategy", "position": "product", "emotion": "brand_strategy",
 "function": "product", "benefit": "product", "category": "product",
 "feature": "product", "context": "audience",
}
INSTR = {
 "concept":  ("concept_relevant",  "identify the single most important idea and distill it into one sentence, 10-20 words"),
 "position": ("position_relevant", "summarize how the company is positioned against competitors — price, proposition, or promise — in one sentence, 15-25 words"),
 "emotion":  ("emotion_relevant",  "distill the feelings or emotions the company seeks to spur, in one sentence, 15-25 words"),
 "function": ("function_relevant", "summarize what the product is, does and how it works — its function — in one sentence, 15-25 words"),
 "benefit":  ("benefit_relevant",  "distill the tangible and intangible benefits, in one sentence, 15-25 words"),
 "category": ("category_relevant", "summarize the category it operates within, in one sentence, 15-20 words"),
 "feature":  ("feature_relevant",  "capture the most important features, in one sentence, 15-25 words"),
 "context":  ("context_relevant",  "summarize the social, physical and environmental context, in one sentence, 15-25 words"),
}
MODELS = ["openai:gpt-5-mini-2025-08-07", "openai:gpt-5-2025-08-07", "openai:gpt-5.5",
          "anthropic:claude-haiku-4-5-20251001", "anthropic:claude-sonnet-4-6",
          "anthropic:claude-opus-4-7"]

# 3 个 curated brief(前 3 个,类别多样)
tests = yaml.safe_load((HERE / "tests.yaml").read_text(encoding="utf-8"))[:3]
(HERE / "tests3.yaml").write_text(yaml.safe_dump(tests, allow_unicode=True, sort_keys=False), encoding="utf-8")

for task, field in REC.items():
    gt, instr = INSTR[task]
    cap = field.replace("_", " ").title()
    raw = (f"You are a brand strategy consultant. From the company information "
           f"below, {instr}. Write only the sentence, no preamble, no quotes.\n\n"
           f"      {cap}: {{{{{field}}}}}")
    lines = [f'description: "Phase4 {task} — 推荐配方 x 6模型 x 3brief"', "",
             "prompts:", f'  - label: "{field}"', "    raw: |"]
    for ln in raw.split("\n"):
        lines.append(f"      {ln}")
    lines.append("")
    lines.append("providers:")
    for m in MODELS:
        lines.append(f"  - {m}")
    lines += ["", "defaultTest:", "  assert:", "    - type: similar",
              f'      value: "{{{{{gt}}}}}"', "      threshold: 0.7", "",
              "tests: file://tests3.yaml", ""]
    (HERE / f"phase4-{task}.yaml").write_text("\n".join(lines), encoding="utf-8")
    print(f"生成 phase4-{task}.yaml")
print(f"\nPhase4: 8任务 x 1配方 x 6模型 x 3brief = {8*6*3} 次(含贵模型)")
