"""Generate Phase-4 configs: each task's winning recipe x 6 models x 3 briefs."""
import yaml
from pathlib import Path

OUT = Path(__file__).resolve().parent / "configs"
OUT.mkdir(exist_ok=True)

# winning single-field recipe per task (from summarize.py, Stage-A)
WINNER = {
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

for task, field in WINNER.items():
    gt, instr = INSTR[task]
    cap = field.replace("_", " ").title()
    raw = (f"You are a brand strategy consultant. From the company information "
           f"below, {instr}. Write only the sentence, no preamble, no quotes.\n\n"
           f"{cap}: {{{{{field}}}}}")
    cfg = {
        "description": f"phase4 {task} — winner recipe x 6 models x 3 briefs",
        "prompts": [{"label": field, "raw": raw}],
        "providers": MODELS,
        "defaultTest": {"assert": [{"type": "similar", "value": f"{{{{{gt}}}}}", "threshold": 0.7}]},
        "tests": "file://../tests3.yaml",
    }
    (OUT / f"phase4-{task}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=4000), "utf-8")

print(f"Wrote {len(WINNER)} Phase-4 configs to configs/")
