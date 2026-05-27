"""
Unit tests for src/evaluators.py.

No API access required — uses a fake embedder for cosine tests.
"""
import math

import pytest

from src.evaluators import (
    EmbeddingClient,
    LENGTH_RANGES,
    KeywordScore,
    SentenceScore,
    _cosine,
    normalize_keyword,
    parse_keywords,
    score_keywords,
    score_sentence,
)


# ---------- Porter / normalization ----------

class TestNormalizeKeyword:
    def test_lowercase(self):
        assert normalize_keyword("FOOD") == normalize_keyword("food")

    def test_hyphen_collapse(self):
        # "non-dairy" and "nondairy" should normalize to the same token.
        assert normalize_keyword("non-dairy") == normalize_keyword("nondairy")

    def test_punctuation_stripped(self):
        assert normalize_keyword("food.") == normalize_keyword("food")
        assert normalize_keyword("'food'") == normalize_keyword("food")

    def test_porter_singular_plural(self):
        # Porter collapses plain -s plurals.
        assert normalize_keyword("plant") == normalize_keyword("plants")

    def test_porter_does_not_collapse_unrelated(self):
        # sanity: two clearly different words don't collide
        assert normalize_keyword("food") != normalize_keyword("forest")

    def test_empty_input(self):
        assert normalize_keyword("") == ""
        assert normalize_keyword("   ") == ""
        assert normalize_keyword("!!!") == ""

    def test_non_string(self):
        assert normalize_keyword(None) == ""  # type: ignore[arg-type]
        assert normalize_keyword(123) == ""   # type: ignore[arg-type]


# ---------- Keyword scoring ----------

class TestScoreKeywords:
    def test_perfect_match(self):
        gt = ["food", "sustainable", "hippie", "plant", "cheese",
              "harmony", "spiritual", "protein", "cultured", "non-dairy"]
        pred = list(gt)
        s = score_keywords(pred, gt)
        assert s.f1 == pytest.approx(1.0)
        assert s.precision == pytest.approx(1.0)
        assert s.recall == pytest.approx(1.0)

    def test_normalization_lets_plural_match(self):
        gt = ["plant"]
        pred = ["plants"]  # plural — should still match after stemming
        s = score_keywords(pred, gt)
        assert s.f1 == 1.0

    def test_hyphen_variants_match(self):
        gt = ["non-dairy"]
        pred = ["nondairy"]
        s = score_keywords(pred, gt)
        assert s.f1 == 1.0

    def test_no_overlap(self):
        s = score_keywords(["foo", "bar"], ["baz", "qux"])
        assert s.precision == 0.0
        assert s.recall == 0.0
        assert s.f1 == 0.0
        assert s.true_positives == 0

    def test_partial_match(self):
        gt = ["a", "b", "c", "d"]
        pred = ["a", "b", "x", "y"]
        s = score_keywords(pred, gt)
        assert s.precision == 0.5  # 2 of 4 predicted are correct
        assert s.recall == 0.5     # 2 of 4 GT recovered
        assert s.f1 == 0.5

    def test_empty_prediction(self):
        s = score_keywords([], ["a", "b"])
        assert s.precision == 0.0
        assert s.recall == 0.0

    def test_dedup(self):
        # Duplicate predictions ("food" and "foods" both stem to same)
        # should not inflate precision via dedup.
        s = score_keywords(["food", "foods", "food"], ["food", "plant"])
        # pred normalizes to {"food"} = 1 unique
        # gt normalizes to {"food", "plant"} = 2 unique
        # tp = 1
        assert s.pred_count == 1
        assert s.gt_count == 2
        assert s.precision == 1.0
        assert s.recall == 0.5


# ---------- Keyword output parsing ----------

class TestParseKeywords:
    def test_clean_numbered_list(self):
        raw = """Step 1: (longlist)
- food
- plant

Step 2: Final 10:
1. food
2. plant
3. sustainable
4. hippie
5. cheese
6. cultured
7. protein
8. harmony
9. spiritual
10. non-dairy
"""
        terms, errors = parse_keywords(raw)
        assert len(terms) == 10
        assert errors == []
        assert terms[0] == "food"
        assert terms[-1] == "non-dairy"

    def test_short_list_flags_error(self):
        raw = "1. food\n2. plant\n3. sustainable"
        terms, errors = parse_keywords(raw)
        assert len(terms) == 3
        assert any("expected 10" in e for e in errors)

    def test_eleven_terms_truncates_to_last_10(self):
        # Without a Step 2 marker, >10 numbered terms suggests Step 1 was also
        # numbered. Parser takes the last 10 and flags the truncation.
        raw = "\n".join(f"{i}. word{i}" for i in range(1, 12))
        terms, errors = parse_keywords(raw)
        assert len(terms) == 10
        assert terms == [f"word{i}" for i in range(2, 12)]
        assert any("truncated" in e for e in errors)

    def test_under_ten_keeps_all_and_flags(self):
        raw = "1. food\n2. plant\n3. sustainable"
        terms, errors = parse_keywords(raw)
        assert len(terms) == 3
        assert any("expected 10" in e for e in errors)

    def test_step2_marker_isolates_correct_list(self):
        # If "Step 2" marker is present, take only numbered lines after it.
        raw = """Step 1: longlist
1. raw
2. brainstorm

Step 2: Final 10
1. food
2. plant
3. sustainable
4. hippie
5. cheese
6. cultured
7. protein
8. harmony
9. spiritual
10. non-dairy
"""
        terms, _ = parse_keywords(raw)
        assert len(terms) == 10
        assert terms[0] == "food"

    def test_markdown_emphasis_stripped(self):
        raw = "1. **food**\n2. *plant*\n3. \"sustainable\""
        terms, _ = parse_keywords(raw)
        assert terms == ["food", "plant", "sustainable"]

    def test_trailing_punctuation_stripped(self):
        raw = "1. food,\n2. plant.\n3. sustainable;"
        terms, _ = parse_keywords(raw)
        assert terms == ["food", "plant", "sustainable"]

    def test_parens_stripped(self):
        raw = "1. adventure (broad concept)\n2. trail"
        terms, _ = parse_keywords(raw)
        assert terms == ["adventure", "trail"]

    def test_multiword_ignored(self):
        raw = "1. food chain\n2. plant"
        terms, errors = parse_keywords(raw)
        # "food chain" rejected; "plant" kept
        assert "plant" in terms
        assert any("multi-word" in e for e in errors)

    def test_hyphenated_multiword_kept(self):
        raw = "1. always-on\n2. plant"
        terms, _ = parse_keywords(raw)
        assert "always-on" in terms

    def test_two_sequences_picks_longest(self):
        # Step 1 has its own numbering; Step 2 has the actual 10
        raw = """Step 1:
1. raw
2. brainstorm
3. ideas

Step 2:
1. food
2. plant
3. sustainable
4. hippie
5. cheese
6. cultured
7. protein
8. harmony
9. spiritual
10. non-dairy
"""
        terms, errors = parse_keywords(raw)
        # Both runs are valid; longest wins (the 10-term one)
        assert len(terms) == 10
        assert "non-dairy" in terms

    def test_bullets_fallback(self):
        raw = "- food\n- plant\n- sustainable"
        terms, errors = parse_keywords(raw)
        assert terms == ["food", "plant", "sustainable"]
        assert any("bullets" in e for e in errors)

    def test_empty_output(self):
        terms, errors = parse_keywords("")
        assert terms == []
        assert errors

    def test_no_structure_at_all(self):
        terms, errors = parse_keywords("This is just prose with no list at all.")
        assert terms == []
        assert errors


# ---------- Sentence scoring (with fake embedder) ----------

class _FakeEmbedder:
    """Embedder that returns deterministic vectors for testing cosine logic."""

    def __init__(self, mapping: dict[str, list[float]] = None):
        self._map = mapping or {}

    def embed(self, text: str) -> list[float]:
        t = text.strip()
        if t in self._map:
            return self._map[t]
        # Default: hash to 3-dim deterministic vector
        h = hash(t)
        return [
            ((h % 1000) / 1000),
            (((h >> 10) % 1000) / 1000),
            (((h >> 20) % 1000) / 1000),
        ]


class TestScoreSentence:
    def test_identical_strings_cosine_one(self):
        emb = _FakeEmbedder({"hello world": [1.0, 0.0, 0.0]})
        s = score_sentence("hello world", "hello world", "concept_relevant", emb)
        assert s.cosine == pytest.approx(1.0)
        assert s.rouge_l == pytest.approx(1.0)

    def test_orthogonal_vectors_cosine_zero(self):
        emb = _FakeEmbedder({
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
        })
        s = score_sentence("a", "b", "concept_relevant", emb)
        assert s.cosine == pytest.approx(0.0)

    def test_empty_prediction(self):
        emb = _FakeEmbedder()
        s = score_sentence("", "something here", "concept_relevant", emb)
        assert s.cosine is None
        assert s.length_compliant is False
        assert s.word_count == 0

    def test_length_compliance_in_range(self):
        emb = _FakeEmbedder()
        # concept_relevant expects 10-20 words
        text = " ".join(["word"] * 15)
        s = score_sentence(text, "ground truth here", "concept_relevant", emb)
        assert s.length_compliant is True
        assert s.word_count == 15

    def test_length_compliance_out_of_range_short(self):
        emb = _FakeEmbedder()
        s = score_sentence("only five words here please now", "ground truth", "concept_relevant", emb)
        # 6 words, but concept_relevant requires 10-20
        assert s.length_compliant is False

    def test_length_compliance_out_of_range_long(self):
        emb = _FakeEmbedder()
        text = " ".join(["word"] * 25)
        s = score_sentence(text, "gt", "category_relevant", emb)
        # category_relevant requires 15-20
        assert s.length_compliant is False

    def test_unknown_task_no_length_constraint(self):
        emb = _FakeEmbedder()
        s = score_sentence("a b c", "gt", "made_up_task", emb)
        # Falls through to (1, 10**6) — anything passes
        assert s.length_compliant is True


# ---------- Cosine math sanity ----------

class TestCosine:
    def test_identical(self):
        assert _cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_opposite(self):
        assert _cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_orthogonal(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        # Don't NaN — return 0 by convention.
        assert _cosine([0, 0, 0], [1, 2, 3]) == 0.0


# ---------- Length ranges sanity ----------

def test_length_ranges_cover_all_8_tasks():
    expected = {
        "concept_relevant", "position_relevant", "emotion_relevant",
        "function_relevant", "benefit_relevant", "category_relevant",
        "feature_relevant", "context_relevant",
    }
    assert set(LENGTH_RANGES.keys()) == expected
