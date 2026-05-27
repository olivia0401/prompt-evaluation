"""Offline tests for scripts.ai_judge_absolute. No API calls."""
import importlib.util
import json
from pathlib import Path

import pytest


def _import_judge():
    here = Path(__file__).resolve().parent.parent / "scripts" / "ai_judge_absolute.py"
    spec = importlib.util.spec_from_file_location("ai_judge_absolute", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- _parse_score ----

@pytest.mark.parametrize("text,expected", [
    ("4", 4),
    ("5", 5),
    ("1", 1),
    ("4.", 4),
    ("4/5", 4),
    ("\"4\"", 4),
    ("Score: 3", 3),
    (" 2 ", 2),
])
def test_parse_score_extracts_first_digit(text, expected):
    judge = _import_judge()
    assert judge._parse_score(text) == expected


@pytest.mark.parametrize("text", [
    None,
    "",
    "I cannot rate this output.",
    "0",     # out of range
    "9",     # out of range
    "abc",
])
def test_parse_score_returns_none_for_unparseable(text):
    judge = _import_judge()
    assert judge._parse_score(text) is None


def test_parse_score_ignores_long_monologue():
    """If the judge wrote a paragraph, treat as parse-fail rather than guess."""
    judge = _import_judge()
    long_text = ("The output is reasonably good and broadly matches the ground truth, "
                 "but misses some details. I would say it deserves a 4.")
    # The "4" is past the first ~20 chars — defensive design treats this as parse-fail.
    assert judge._parse_score(long_text) is None


# ---- _done_keys / resume logic ----

def test_done_keys_returns_empty_for_missing_file(tmp_path):
    judge = _import_judge()
    assert judge._done_keys(tmp_path / "nope.jsonl") == set()


def test_done_keys_indexes_resume_tuples(tmp_path):
    judge = _import_judge()
    p = tmp_path / "ai_judge_absolute.jsonl"
    rows = [
        {"brief_id": "Plonts", "task": "concept_relevant",
         "config_id": "A:personality", "model_key": "haiku", "run_id": 1,
         "score": 4},
        {"brief_id": "Plonts", "task": "emotion_relevant",
         "config_id": "A:_full_brief", "model_key": "gpt5mini", "run_id": 1,
         "score": 3},
        # A row with score=None must be ignored — it should be re-judged.
        {"brief_id": "Plonts", "task": "feature_relevant",
         "config_id": "A:product", "model_key": "haiku", "run_id": 1,
         "score": None},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    done = judge._done_keys(p)
    assert ("Plonts", "concept_relevant", "A:personality", "haiku", 1) in done
    assert ("Plonts", "emotion_relevant", "A:_full_brief", "gpt5mini", 1) in done
    # Parse-failed row should NOT be marked done — it must be retried.
    assert ("Plonts", "feature_relevant", "A:product", "haiku", 1) not in done
    assert len(done) == 2


def test_done_keys_skips_blank_or_malformed_lines(tmp_path):
    judge = _import_judge()
    p = tmp_path / "ai_judge_absolute.jsonl"
    p.write_text(
        "\n"  # blank
        "not json at all\n"
        + json.dumps({"brief_id": "X", "task": "t", "config_id": "c",
                      "model_key": "m", "run_id": 1, "score": 5}) + "\n",
        encoding="utf-8",
    )
    done = judge._done_keys(p)
    assert done == {("X", "t", "c", "m", 1)}


# ---- prompt template sanity ----

def test_prompt_template_contains_required_placeholders():
    judge = _import_judge()
    t = judge.ABSOLUTE_TEMPLATE
    assert "{task}" in t
    assert "{ground_truth}" in t
    assert "{ai_output}" in t
    assert "1, 2, 3, 4, or 5" in t  # the scale must be unambiguous
