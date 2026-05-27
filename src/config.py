"""
Paths, API keys, model registry, budget caps.

MODELS dict keys ("haiku" / "gpt5mini" / "sonnet" / "gpt5" / "opus47" /
"gpt55") are the short keys used everywhere (CSV, JSONL, config_id).
Renaming them invalidates existing JSONL/CSV resume keys.

Tiers:
  cheap   : Stage A broad screening (haiku, gpt5mini)
  medium  : Stage B stability reruns + Sonnet judge (sonnet, gpt5)
  premium : Phase 4 final-config validation ONLY (opus47, gpt55)
            — Project constraint: ≤£1 for premium re-run, top configs only.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Target Google Sheets to replace on each build_xlsx run ---
# Set to None to skip upload (and keep the local Results/*.xlsx file instead).
# When set, build_xlsx generates the workbook in a temp file, uploads it to
# Drive as a replacement for this Sheets file's contents (preserving formatting,
# heatmaps, embedded images via the Drive xlsx→Sheets conversion), then
# deletes the temp file. The deliverable URL stays the same across builds.
#
# IMPORTANT: this requires BOTH Google Drive API AND Google Sheets API enabled
# in the Cloud project — Drive does the xlsx→Sheets conversion internally and
# fails silently with 403 "insufficientFilePermissions" if Sheets API is off.
# Set via env var GOOGLE_SHEETS_ID, or paste your own Sheets ID here.
RESULTS_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID")

# Chinese-edition workbook target. Used when `python -m scripts.build_xlsx
# --lang zh` runs. Same upload mechanism, separate Sheet so EN and ZH never
# overwrite each other. Set via env var GOOGLE_SHEETS_ID_ZH.
RESULTS_SHEETS_ID_ZH = os.getenv("GOOGLE_SHEETS_ID_ZH")

# --- API keys (set in .env, not committed) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Paths ---
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = PROJECT_ROOT / "Results"
BRIEFS_FILE = PROJECT_ROOT / "briefs.yml"
PROMPTS_FILE = PROJECT_ROOT / "prompts.txt"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# NOTE: OneDrive will lock files mid-write. If this project lives under
# OneDrive, right-click `outputs/` -> "Always keep on this device" AND
# pause sync for the folder, OR symlink outputs/ to a non-OneDrive path.

# --- Determinism ---
SEED = 42  # OpenAI only; Anthropic has no seed parameter

# --- Network ---
HTTP_TIMEOUT = 60.0

# --- Concurrency (per-model threading.Semaphore initial values) ---
# Wired through LLMClient.from_env -> __init__ as per-model semaphores.
# Stage A (2026-05-19) showed Haiku=15 caused 77% rate_limited; Anthropic
# Tier-1 RPM cannot sustain it. 5 is the highest value where Haiku stayed
# stable under sustained load with sdk_max_retries=8.
CONCURRENCY = {
    "haiku": 5,
    "gpt5mini": 15,
    "sonnet": 5,
    "gpt5": 5,
    "opus47": 3,   # premium — keep slow, low TPM tier
    "gpt55": 3,
}

# --- Model registry ---
# After running scripts/verify_models.py, replace each `id` with the
# verified value and append `# verified YYYY-MM-DD`.
MODELS = {
    "haiku": {
        "provider": "anthropic",
        "id": "claude-haiku-4-5-20251001",  # TODO verify
        "tier": "cheap",
        "params": {"temperature": 0, "max_tokens": 1000},
        "price_per_1m": {"input": 1.00, "output": 5.00},
        "supports_seed": False,
        "is_reasoning_model": False,
    },
    "gpt5mini": {
        "provider": "openai",
        "id": "gpt-5-mini-2025-08-07",  # TODO verify
        "tier": "cheap",
        "params": {
            "reasoning_effort": "minimal",
            "seed": SEED,
            "max_completion_tokens": 1000,
            # NOTE: no temperature — reasoning models reject it
        },
        "price_per_1m": {"input": 0.25, "output": 2.00},
        "supports_seed": True,
        "is_reasoning_model": True,
    },
    "sonnet": {
        "provider": "anthropic",
        "id": "claude-sonnet-4-6",  # TODO verify — date looks future-dated
        "tier": "medium",
        "params": {"temperature": 0, "max_tokens": 1000},
        "price_per_1m": {"input": 3.00, "output": 15.00},
        "supports_seed": False,
        "is_reasoning_model": False,
    },
    "gpt5": {
        "provider": "openai",
        "id": "gpt-5-2025-08-07",  # TODO verify
        "tier": "medium",
        "params": {
            "reasoning_effort": "minimal",
            "seed": SEED,
            "max_completion_tokens": 1000,
        },
        "price_per_1m": {"input": 1.25, "output": 10.00},
        "supports_seed": True,
        "is_reasoning_model": True,
    },
    # ---- Premium tier — Phase 4 only, on top-1 config per task. ----
    # IDs and prices below are best-guess; verify with scripts/verify_models.py
    # before running Phase 4. If the ID doesn't list, verify_models prints
    # similar IDs from the provider so you can re-pick.
    "opus47": {
        "provider": "anthropic",
        "id": "claude-opus-4-7",  # verified 2026-05-19
        "tier": "premium",
        # NOTE: Anthropic deprecated `temperature` on Opus 4.7 — passing it
        # returns 400 "temperature is deprecated for this model". Omit it.
        "params": {"max_tokens": 1000},
        "price_per_1m": {"input": 15.00, "output": 75.00},  # TODO verify pricing
        "supports_seed": False,
        "is_reasoning_model": False,
    },
    "gpt55": {
        "provider": "openai",
        "id": "gpt-5.5",  # verified 2026-05-19
        "tier": "premium",
        # NOTE: GPT-5.5 dropped support for reasoning_effort='minimal'.
        # Valid values: none|low|medium|high|xhigh. 'none' is cheapest and
        # matches the spirit of our 'no extra reasoning tokens' choice.
        "params": {
            "reasoning_effort": "none",
            "seed": SEED,
            "max_completion_tokens": 1000,
        },
        "price_per_1m": {"input": 2.50, "output": 20.00},  # TODO verify pricing
        "supports_seed": True,
        "is_reasoning_model": True,
    },
}

# --- Embedding ---
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_FALLBACK = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_PRICE_PER_1M = 0.13  # text-embedding-3-large

# --- USD budget caps per stage. Trips BudgetExceededError when exceeded. ---
# Project total budget: £50 (~$63 USD), ideally less. Premium re-run: ≤£1.
# Sum of caps below stays well under £50 to leave headroom for re-runs.
BUDGET_CAP = {
    "phase_0":         1.0,    # Pilot on 3 briefs, cheap models
    "phase_1":         8.0,    # Stage A — Phase 1 single-field screen
    "phase_2":         8.0,    # Stage A — Phase 2 semantic pairs
    "phase_3":         2.0,    # Stage A — Phase 3 keyword versions
    "stage_b":         5.0,    # Stage B — stability reruns + optional Sonnet judge
    "phase_4_premium": 1.25,   # Phase 4 — Opus 4.7 + GPT-5.5 on top configs only (≈£1)
    "verify":          0.5,
    "noise_floor":     0.5,
}
# Total of above: $26.25 — well within £50 (~$63) with comfortable headroom.

# Hard ceiling: nothing in the experiment may push past this without an
# explicit override. Mirrors the project's "≤£50 total" rule.
TOTAL_BUDGET_CEILING_USD = 63.0

# Per-model cap. No single model_key may accumulate more than this in USD.
# Catches runaway spend on one provider (e.g., a stuck reasoning model burning
# tokens) without blocking other models. Independent from the per-stage caps.
PER_MODEL_BUDGET_CAP_USD = 8.0
