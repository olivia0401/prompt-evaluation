"""
把你真实的 briefs.yml 转成 promptfoo 能读的 tests.yaml。

这一步 = 你原来手写的"加载 brief 数据"那部分。跑一次生成 tests.yaml,
之后 promptfoo 直接读它。不花钱、不调 API,只是数据格式转换。
"""
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
briefs = yaml.safe_load((HERE.parent / "briefs.yml").read_text(encoding="utf-8"))

def clean(v):
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    return " ".join(str(v or "").split())   # 压掉多余换行/空格

tests = []
for b in briefs:
    tests.append({"vars": {
        "current_name":     clean(b.get("current_name")),
        "product":          clean(b.get("product")),
        "differentiators":  clean(b.get("differentiators")),
        "audience":         clean(b.get("audience")),
        "brand_strategy":   clean(b.get("brand_strategy")),
        "personality":      clean(b.get("personality")),
        # 8 个任务的标准答案都带上,后面别的任务能复用同一个 tests.yaml
        "concept_relevant":   clean(b.get("concept_relevant")),
        "position_relevant":  clean(b.get("position_relevant")),
        "emotion_relevant":   clean(b.get("emotion_relevant")),
        "function_relevant":  clean(b.get("function_relevant")),
        "benefit_relevant":   clean(b.get("benefit_relevant")),
        "category_relevant":  clean(b.get("category_relevant")),
        "feature_relevant":   clean(b.get("feature_relevant")),
        "context_relevant":   clean(b.get("context_relevant")),
    }})

out = HERE / "tests.yaml"
out.write_text(yaml.safe_dump(tests, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(f"生成 {out.name}: {len(tests)} 个 brief")
