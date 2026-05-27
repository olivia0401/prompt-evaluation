"""Unit tests for src/utils.py — word_count and config_id rules."""
from src.utils import word_count, make_config_id, normalize_keyword


def test_word_count_basic():
    assert word_count("hello world") == 2


def test_word_count_empty():
    assert word_count("") == 0
    assert word_count("   ") == 0


def test_word_count_multiple_spaces():
    assert word_count("  hello   world  ") == 2


def test_word_count_hyphenated_counts_as_one():
    # Hyphenated terms count as a single word (split() preserves the hyphen).
    assert word_count("co-located fully") == 2


def test_word_count_punctuation_attached():
    # split() doesn't split on punctuation by itself.
    assert word_count("hello, world!") == 2


def test_word_count_newlines():
    assert word_count("hello\nworld") == 2


def test_word_count_real_gt_example():
    # From briefs.yml: Plonts concept_relevant
    s = "Promoting the harmonious coexistence of people and the planet through a spiritual and down-to-earth persona."
    assert word_count(s) == 15


def test_make_config_id_sorts_fields():
    assert make_config_id(["product", "audience"]) == "audience+product"
    assert make_config_id(["audience", "product"]) == "audience+product"


def test_make_config_id_empty():
    assert make_config_id([]) == "_baseline"


def test_make_config_id_single():
    assert make_config_id(["product"]) == "product"


def test_normalize_keyword_pre_stem():
    # utils.normalize_keyword only does lowercase + strip hyphens (no stemming).
    # Porter stemming lives in evaluators.normalize_keyword.
    assert normalize_keyword("Non-Dairy") == "nondairy"
    assert normalize_keyword("  FOOD  ") == "food"
