"""
Prompt assembly + config matrix generation.

Two responsibilities:
  1. Parse prompts.txt into 8 sentence-task templates + 4 keyword versions.
  2. Inject brief fields into a template to produce the final prompt string.
  3. Enumerate all (task, prompt_version, field_subset) configs for a stage.

Public surface:
  - load_prompts()                  -> {task: PromptTemplate}
  - load_briefs()                   -> list[dict]
  - build_prompt(task, fields, brief, version='A') -> str
  - is_compatible(task, fields)     -> bool
  - list_configs_for_phase(phase)   -> list[Config]
  - Config dataclass

The keyword task has 4 prompt versions (A/B/C/D); other tasks always use version 'A'.

All semantic-field combinations are run regardless of nominal prompt-content
fit; the data shows which combinations work and which don't.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable, Optional

import yaml

from src import config as cfg

# ---------- Task / field definitions ----------

SENTENCE_TASKS = [
    "concept_relevant",
    "position_relevant",
    "emotion_relevant",
    "function_relevant",
    "benefit_relevant",
    "category_relevant",
    "feature_relevant",
    "context_relevant",
]
KEYWORD_TASK = "keywords"
ALL_TASKS = SENTENCE_TASKS + [KEYWORD_TASK]

# Semantic brief fields tested individually and in combinations.
SEMANTIC_FIELDS = ["product", "differentiators", "audience", "brand_strategy", "personality"]
# Metadata fields, used only in a few hypothesis-driven combos.
#
# Excluded by design — uninformative for the extraction tasks:
#   - "current_name"  — existing brand name being replaced; uninformative for new-name generation
#   - "priorities"    — meta-list of which fields matter; not content itself
# These are excluded from BOTH single-field tests and the full_brief baseline
# so we don't waste budget on fields known to be uninformative.
METADATA_FIELDS = ["business_category"]

# Special config sentinels
FULL_BRIEF = "_full_brief"
PROMPT_IMPLIED = "_prompt_implied"
TRIVIAL = "_trivial"

# Phase 0 Pilot: 3 hand-picked briefs covering category diversity.
PHASE0_BRIEF_NAMES = [
    "Plonts",              # consumer food (plant-based)
    "Data Fabric",         # B2B tech / enterprise data
    "Board of Innovation", # B2B service / consulting
]

# Phase 2 targeted metadata combos (hypothesis-driven).
TARGETED_METADATA_COMBOS: list[tuple[str, list[str]]] = [
    # (task, fields)
    ("category_relevant", ["business_category"]),
    ("function_relevant", ["business_category", "product"]),
    ("context_relevant", ["audience"]),  # audience is semantic but combo is targeted
]

# Incompatibility: task prompt wording references a specific kind
# of content (e.g., "company's brand strategy below" expects brand_strategy
# or full brief; injecting only `audience` makes the prompt nonsensical).
# Single-field configs are flagged here. Pairs are compatible if at least
# one field is compatible.
COMPATIBLE_SINGLE_FIELDS: dict[str, set[str]] = {
    # Sentence tasks where prompt explicitly says "brand strategy below":
    "concept_relevant":  {"brand_strategy"},
    "position_relevant": {"brand_strategy"},
    "emotion_relevant":  {"brand_strategy"},
    # Sentence tasks where prompt says "product or service description":
    "function_relevant": {"product", "differentiators"},
    "benefit_relevant":  {"product", "differentiators"},
    "category_relevant": {"product", "business_category"},
    "feature_relevant":  {"product", "differentiators"},
    "context_relevant":  {"product", "audience"},
}


@dataclass(frozen=True)
class PromptTemplate:
    task: str
    version: str            # 'A' for sentence tasks; A/B/C/D for keyword
    instruction: str        # the prompt text up to (but not including) the brief
    appends_brief: bool     # whether [INSERT BRIEF] or similar marker is at end


@dataclass(frozen=True)
class Config:
    """A single experimental configuration. Model and run_id are separate."""
    task: str
    fields: tuple[str, ...]      # sorted; empty tuple = baseline (see version)
    prompt_version: str = "A"    # only used by keyword task

    @property
    def config_id(self) -> str:
        # Frozen format — never change.
        if not self.fields:
            return f"{self.prompt_version}:_baseline"
        joined = "+".join(self.fields)
        return f"{self.prompt_version}:{joined}"


# ---------- prompts.txt loader ----------

_TASK_NAME_FROM_HEADER = {
    "concept-relevant":  "concept_relevant",
    "position-relevant": "position_relevant",
    "emotion-relevant":  "emotion_relevant",
    "function-relevant": "function_relevant",
    "benefit-relevant":  "benefit_relevant",
    "category-relevant": "category_relevant",
    "feature-relevant":  "feature_relevant",
    "context-relevant":  "context_relevant",
}


def load_prompts(path: Optional[Path] = None) -> dict[str, PromptTemplate]:
    """
    Parse prompts.txt into a dict {task_or_'keywords:A': PromptTemplate}.

    Format of prompts.txt:
      "Sub-clue: Concept-Relevant" header, then "LLM Prompt: ..." text.
      The keyword extraction prompt comes last (different header).
    """
    if path is None:
        path = cfg.PROMPTS_FILE
    raw = Path(path).read_text(encoding="utf-8")
    out: dict[str, PromptTemplate] = {}

    # Split sentence tasks first
    chunks = re.split(r"(?im)^\s*Sub-clue:\s*(.+?)\s*$", raw)
    # chunks[0] is preamble (often empty), then alternating (name, body)
    for i in range(1, len(chunks), 2):
        header = chunks[i].strip().lower()
        body = chunks[i + 1] if i + 1 < len(chunks) else ""
        task = _TASK_NAME_FROM_HEADER.get(header)
        if task is None:
            continue
        # Extract the LLM Prompt body
        m = re.search(r"(?is)LLM\s*Prompt\s*:\s*(.+?)(?=\n\s*Sub-clue:|\n\s*Keyword|\Z)", body)
        if not m:
            continue
        instruction = m.group(1).strip()
        out[task] = PromptTemplate(task=task, version="A", instruction=instruction, appends_brief=True)

    # Keyword task — find every "keyword ... brief" section.
    # Header forms supported:
    #   "Keyword extraxction brief:"                            → Version A (legacy, has typo)
    #   "Keyword extraction brief (Version B — Reduced ...)"    → Version B
    #   "Keyword extraction brief (Version C ...)"              → Version C
    #   "Keyword extraction brief (Version D ...)"              → Version D
    lines = raw.splitlines()
    keyword_headers: list[tuple[int, str]] = []  # (line_idx, version_letter)
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if not stripped.startswith("keyword"):
            continue
        if "brief" not in stripped and not stripped.endswith(":"):
            continue
        m = re.search(r"version\s+([abcd])", stripped)
        version = m.group(1).upper() if m else "A"
        keyword_headers.append((i, version))

    for idx, (header_row, version) in enumerate(keyword_headers):
        body_start = header_row + 1
        body_end = keyword_headers[idx + 1][0] if idx + 1 < len(keyword_headers) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip()
        instruction = re.sub(
            r"\[INSERT BRIEF\].*$", "", body, flags=re.IGNORECASE | re.DOTALL
        ).strip()
        out[f"{KEYWORD_TASK}:{version}"] = PromptTemplate(
            task=KEYWORD_TASK, version=version, instruction=instruction, appends_brief=True
        )

    return out


# ---------- briefs.yml loader ----------

def load_briefs(path: Optional[Path] = None) -> list[dict]:
    if path is None:
        path = cfg.BRIEFS_FILE
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def brief_id(brief: dict) -> str:
    """Stable identifier for a brief — uses current_name."""
    return str(brief.get("current_name", "")).strip()


# ---------- Compatibility ----------

def is_compatible(task: str, fields: Iterable[str]) -> bool:
    """
    Incompatibility policy: prompt headers are not modified, so
    fields whose semantics conflict with the prompt's content reference are
    skipped.

    Rule: a pair config is compatible if at least one field is in
    COMPATIBLE_SINGLE_FIELDS[task]. A single field config is compatible
    only if it's listed. Baselines (no fields) and full_brief are always
    compatible. Keyword task accepts any combination.
    """
    fields = tuple(fields)
    if task == KEYWORD_TASK:
        return True
    if not fields:
        return True
    allowed = COMPATIBLE_SINGLE_FIELDS.get(task, set(SEMANTIC_FIELDS))
    return any(f in allowed for f in fields)


# ---------- Field rendering ----------

def _render_field(name: str, value) -> str:
    """Render one brief field for injection. Lists become bullet text."""
    if value is None:
        return ""
    if isinstance(value, list):
        body = "\n".join(f"- {v}" for v in value)
    else:
        body = str(value).strip()
    label = name.replace("_", " ").title()
    return f"{label}:\n{body}"


def _render_brief_subset(brief: dict, fields: Iterable[str]) -> str:
    parts = []
    for f in fields:
        v = brief.get(f)
        if v is None or (isinstance(v, str) and not v.strip()) or (isinstance(v, list) and not v):
            continue
        parts.append(_render_field(f, v))
    return "\n\n".join(parts)


def _render_full_brief(brief: dict) -> str:
    """All informative fields, in a stable canonical order.

    current_name and priorities are intentionally excluded — uninformative for the
    flagged them as uninformative for this generation task. See METADATA_FIELDS.
    """
    canonical = ["business_category"] + SEMANTIC_FIELDS
    return _render_brief_subset(brief, canonical)


# ---------- Build prompt ----------

def build_prompt(
    task: str,
    fields: Iterable[str] | str,
    brief: dict,
    *,
    prompt_version: str = "A",
    templates: Optional[dict[str, PromptTemplate]] = None,
) -> str:
    """
    Assemble the final prompt for one (task, fields, brief, version) cell.

    `fields` can be:
      - a list/tuple of field names → inject those fields
      - FULL_BRIEF → inject the whole brief
      - PROMPT_IMPLIED → inject nothing (instruction only)
      - TRIVIAL → reserved; not callable via API (handled outside)
    """
    if templates is None:
        templates = load_prompts()

    key = f"{task}:{prompt_version}" if task == KEYWORD_TASK else task
    tmpl = templates.get(key)
    if tmpl is None:
        raise KeyError(f"No template for task={task} version={prompt_version}")

    if fields == TRIVIAL:
        raise ValueError("TRIVIAL baseline does not produce a prompt; handle outside build_prompt.")

    instruction = tmpl.instruction

    if fields == PROMPT_IMPLIED:
        brief_text = ""
    elif fields == FULL_BRIEF:
        brief_text = _render_full_brief(brief)
    else:
        brief_text = _render_brief_subset(brief, fields)

    if not brief_text:
        # Prompt-implied baseline: instruction only.
        return instruction.strip()

    return f"{instruction.strip()}\n\nBRIEF:\n{brief_text}"


# ---------- Config matrix per phase ----------

def list_phase0_configs() -> list[Config]:
    """
    Phase 0 — Pilot:
      - 8 sentence tasks x 2 configs (full brief + prompt-implied baseline)
      - keyword task x 1 prompt version (A)
    """
    out: list[Config] = []
    for task in SENTENCE_TASKS:
        out.append(Config(task=task, fields=(FULL_BRIEF,)))
        out.append(Config(task=task, fields=(PROMPT_IMPLIED,)))
    out.append(Config(task=KEYWORD_TASK, fields=(FULL_BRIEF,), prompt_version="A"))
    return out


def list_phase1_configs(filter_incompatible: bool = False) -> list[Config]:
    """
    Phase 1 — Single semantic-field evaluation on the 8 sentence tasks.
    Includes the full-brief baseline and prompt-implied baseline for all
    23 briefs (Phase 0 only ran 3 briefs; we need all 23 as anchors).

    filter_incompatible defaults to False: the experiment's core question is
    which fields work for which tasks, so pre-filtering "obvious mismatches"
    would pre-judge the answer. is_compatible() remains available for
    downstream tagging.
    """
    out: list[Config] = []
    for task in SENTENCE_TASKS:
        out.append(Config(task=task, fields=(FULL_BRIEF,)))
        out.append(Config(task=task, fields=(PROMPT_IMPLIED,)))
        for f in SEMANTIC_FIELDS:
            if filter_incompatible and not is_compatible(task, (f,)):
                continue
            out.append(Config(task=task, fields=(f,)))
    return out


def list_phase2_configs(filter_incompatible: bool = False) -> list[Config]:
    """
    Phase 2 — Semantic+semantic pairs (10 combinations) + targeted metadata.
    Phase 1 / 0 configs are NOT repeated here.

    filter_incompatible defaults to False (see list_phase1_configs).
    """
    out: list[Config] = []
    pairs = list(combinations(SEMANTIC_FIELDS, 2))  # exactly 10 pairs
    for task in SENTENCE_TASKS:
        for pair in pairs:
            fields = tuple(sorted(pair))
            if filter_incompatible and not is_compatible(task, fields):
                continue
            out.append(Config(task=task, fields=fields))
    # Targeted metadata combos
    for task, fields in TARGETED_METADATA_COMBOS:
        out.append(Config(task=task, fields=tuple(sorted(fields))))
    return out


def list_phase3_configs() -> list[Config]:
    """
    Phase 3 - Keyword prompt compression. 4 versions x 23 briefs x 2 models.
    Only the keyword task; sentence tasks not affected.
    """
    return [
        Config(task=KEYWORD_TASK, fields=(FULL_BRIEF,), prompt_version=v)
        for v in ("A", "B", "C", "D")
    ]


def list_configs_for_stage(stage: str) -> list[Config]:
    """
    Return the union of configs for a stage. Used by run_experiment.build_todo().
    Stage A = Phases 0 + 1 + 2 + 3 (de-duplicated).
    """
    seen: set[str] = set()
    out: list[Config] = []

    def add_unique(cfgs: list[Config]):
        for c in cfgs:
            key = (c.task, c.config_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)

    if stage == "phase0":
        add_unique(list_phase0_configs())
    elif stage in ("stage_a", "phase_1_2_3"):
        add_unique(list_phase1_configs())
        add_unique(list_phase2_configs())
        add_unique(list_phase3_configs())
    elif stage == "phase1":
        add_unique(list_phase1_configs())
    elif stage == "phase2":
        add_unique(list_phase2_configs())
    elif stage == "phase3":
        add_unique(list_phase3_configs())
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return out
