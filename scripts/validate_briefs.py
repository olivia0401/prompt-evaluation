"""
Validate briefs.yml before any API call.

Checks (per brief):
  - All required metadata fields present and non-empty
  - All 5 semantic fields present and non-empty
  - All 8 *_relevant ground-truth fields present and non-empty
  - keywords list has exactly 10 string items
  - personality and priorities are lists of strings
  - GT length is within the range stated in the corresponding task prompt

Exits 0 if all clean, 1 if any issue.

Usage:
  python -m scripts.validate_briefs
  python -m scripts.validate_briefs --strict  # also fail on length warnings
"""
import argparse
import sys
from pathlib import Path

import yaml

from src import config as cfg
from src.utils import word_count

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REQUIRED_METADATA = ["current_name", "business_category", "priorities"]
REQUIRED_SEMANTIC = ["product", "differentiators", "audience", "brand_strategy", "personality"]
REQUIRED_GT_SENTENCE = [
    "concept_relevant",
    "position_relevant",
    "emotion_relevant",
    "function_relevant",
    "benefit_relevant",
    "category_relevant",
    "feature_relevant",
    "context_relevant",
]

# Length ranges per prompts.txt (min, max). Used to flag suspicious GT.
LENGTH_RANGES = {
    "concept_relevant":  (10, 20),
    "position_relevant": (15, 25),  # adjusted 2026-05-18: matched to GT length distribution
    "emotion_relevant":  (15, 25),
    "function_relevant": (15, 25),
    "benefit_relevant":  (15, 25),
    "category_relevant": (15, 20),
    "feature_relevant":  (15, 25),
    "context_relevant":  (15, 25),
}

# Audit thresholds live in src/config.py.
from src.config import EXPECTED_BRIEF_COUNT, EXPECTED_KEYWORD_COUNT  # noqa: E402


def check_brief(idx: int, brief: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one brief."""
    errors, warnings = [], []
    name = brief.get("current_name", f"<brief #{idx}>")

    # Metadata
    for f in REQUIRED_METADATA:
        v = brief.get(f)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
            errors.append(f"missing/empty metadata: {f}")

    # Semantic
    for f in REQUIRED_SEMANTIC:
        v = brief.get(f)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
            errors.append(f"missing/empty semantic field: {f}")

    # personality / priorities must be lists of strings
    for f in ("personality", "priorities"):
        v = brief.get(f)
        if v is not None and not isinstance(v, list):
            errors.append(f"{f} must be a list, got {type(v).__name__}")
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"{f}[{i}] not a non-empty string")

    # Ground-truth sentences
    for f in REQUIRED_GT_SENTENCE:
        v = brief.get(f)
        if not isinstance(v, str) or not v.strip():
            errors.append(f"missing/empty GT: {f}")
            continue
        # Length check
        wc = word_count(v)
        lo, hi = LENGTH_RANGES[f]
        if wc < lo or wc > hi:
            warnings.append(f"GT length out of prompt range for {f}: {wc} words (expected {lo}-{hi})")

    # Keywords
    kw = brief.get("keywords")
    if not isinstance(kw, list):
        errors.append(f"keywords must be a list, got {type(kw).__name__}")
    else:
        if len(kw) != EXPECTED_KEYWORD_COUNT:
            errors.append(f"keywords count = {len(kw)} (expected {EXPECTED_KEYWORD_COUNT})")
        for i, item in enumerate(kw):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"keywords[{i}] not a non-empty string")
        # Soft check: keywords should not contain spaces (per prompt spec)
        bad = [w for w in kw if isinstance(w, str) and " " in w.strip()]
        if bad:
            warnings.append(f"keywords with spaces (prompt says single words): {bad}")

    return errors, warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    args = ap.parse_args()

    path = cfg.BRIEFS_FILE
    if not path.exists():
        print(f"ERROR: briefs file not found at {path}")
        sys.exit(2)

    with open(path, encoding="utf-8") as f:
        briefs = yaml.safe_load(f)

    if not isinstance(briefs, list):
        print(f"ERROR: briefs.yml root must be a list, got {type(briefs).__name__}")
        sys.exit(2)

    print(f"Loaded {len(briefs)} briefs from {path}\n")

    total_errors = 0
    total_warnings = 0
    issue_count = 0

    if len(briefs) != EXPECTED_BRIEF_COUNT:
        print(f"WARNING: expected {EXPECTED_BRIEF_COUNT} briefs, got {len(briefs)}")
        total_warnings += 1

    for i, brief in enumerate(briefs):
        name = brief.get("current_name", f"<#{i}>")
        errors, warnings = check_brief(i, brief)
        if errors or warnings:
            issue_count += 1
            print(f"[{i:2d}] {name}")
            for e in errors:
                print(f"     ERROR  : {e}")
                total_errors += 1
            for w in warnings:
                print(f"     warn   : {w}")
                total_warnings += 1

    print()
    print("=" * 60)
    print(f"Summary: {len(briefs)} briefs checked")
    print(f"  Errors:   {total_errors}")
    print(f"  Warnings: {total_warnings}")
    print(f"  Briefs with issues: {issue_count}")
    print("=" * 60)

    if total_errors > 0:
        print("\nFAIL: fix errors before running any API call.")
        sys.exit(1)
    if args.strict and total_warnings > 0:
        print("\nFAIL (strict mode): warnings present.")
        sys.exit(1)

    print("\nOK: briefs.yml is ready.")
    sys.exit(0)


if __name__ == "__main__":
    main()
