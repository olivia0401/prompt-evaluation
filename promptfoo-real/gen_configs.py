"""Generate the 8 Stage-A task configs (17 recipes x 2 cheap models each)."""
import itertools
import yaml
from pathlib import Path

OUT = Path(__file__).resolve().parent / "configs"
OUT.mkdir(exist_ok=True)

# task -> (ground-truth field, task instruction)
TASKS = {
    "concept":  ("concept_relevant",  "identify the single most important idea — the one that, if removed, would make the brand unrecognizable — and distill it into one sentence, 10-20 words"),
    "position": ("position_relevant", "summarize how the company seeks to be positioned against its competitors — in terms of price, proposition, or promise — in one sentence, 15-25 words"),
    "emotion":  ("emotion_relevant",  "distill the feelings or emotions the company seeks to spur in its audience into one sentence — describe feelings, not features — 15-25 words"),
    "function": ("function_relevant", "summarize what the product or service is, what it does and how it works — its function, not its benefits — in one sentence, 15-25 words"),
    "benefit":  ("benefit_relevant",  "distill the tangible and intangible benefits of using the product or service into one sentence, 15-25 words"),
    "category": ("category_relevant", "summarize the category the product or service operates within, in one sentence, 15-20 words"),
    "feature":  ("feature_relevant",  "capture the most important and consequential features of the product or service in one sentence, 15-25 words"),
    "context":  ("context_relevant",  "summarize the social, physical and environmental context(s) in which the product or service exists, in one sentence, 15-25 words"),
}

SEMANTIC = ["product", "differentiators", "audience", "brand_strategy", "personality"]
MODELS = ["openai:gpt-5-mini-2025-08-07", "anthropic:claude-haiku-4-5-20251001"]

# 17 recipes: full brief, 5 single fields, 10 field pairs, no-context baseline
RECIPES = ([("full-brief", SEMANTIC)]
           + [(f, [f]) for f in SEMANTIC]
           + [(f"{a}+{b}", [a, b]) for a, b in itertools.combinations(SEMANTIC, 2)]
           + [("baseline", None)])


def prompt(instr, fields):
    if fields is None:
        return (f"You are a brand strategy consultant. Based on general brand "
                f"knowledge, {instr}. Write only the sentence, no preamble, no quotes.")
    block = "\n".join(f"{f.replace('_', ' ').title()}: {{{{{f}}}}}" for f in fields)
    return (f"You are a brand strategy consultant. From the company information "
            f"below, {instr}. Write only the sentence, no preamble, no quotes.\n\n{block}")


for task, (gt, instr) in TASKS.items():
    cfg = {
        "description": f"{task} — recipe screening (17 recipes x 2 cheap models)",
        "prompts": [{"label": label, "raw": prompt(instr, fields)} for label, fields in RECIPES],
        "providers": MODELS,
        "defaultTest": {"assert": [{"type": "similar", "value": f"{{{{{gt}}}}}", "threshold": 0.7}]},
        "tests": "file://../tests.yaml",
    }
    (OUT / f"{task}.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=4000), "utf-8")

print(f"Wrote {len(TASKS)} configs to configs/")
