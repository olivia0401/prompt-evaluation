"""Build promptfoo test cases (tests.yaml) from the local briefs.yml."""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["current_name", "product", "differentiators", "audience",
          "brand_strategy", "personality"]
GROUND_TRUTH = ["concept_relevant", "position_relevant", "emotion_relevant",
                "function_relevant", "benefit_relevant", "category_relevant",
                "feature_relevant", "context_relevant"]


def clean(v):
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return " ".join(str(v or "").split())


briefs = yaml.safe_load((ROOT.parent / "briefs.yml").read_text("utf-8"))
tests = [{"vars": {k: clean(b.get(k)) for k in FIELDS + GROUND_TRUTH}} for b in briefs]

(ROOT / "tests.yaml").write_text(
    yaml.safe_dump(tests, allow_unicode=True, sort_keys=False), "utf-8")
# tests3.yaml: first 3 briefs, used by the Phase-4 premium-model configs
(ROOT / "tests3.yaml").write_text(
    yaml.safe_dump(tests[:3], allow_unicode=True, sort_keys=False), "utf-8")
print(f"Wrote tests.yaml ({len(tests)} briefs) and tests3.yaml (3)")
