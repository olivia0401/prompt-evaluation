"""Unit tests for src/prompt_builder.py — verifies templates, injection, and config matrix."""
import pytest

from src.prompt_builder import (
    ALL_TASKS,
    Config,
    FULL_BRIEF,
    KEYWORD_TASK,
    METADATA_FIELDS,
    PROMPT_IMPLIED,
    SEMANTIC_FIELDS,
    SENTENCE_TASKS,
    build_prompt,
    is_compatible,
    list_configs_for_stage,
    list_phase0_configs,
    list_phase1_configs,
    list_phase2_configs,
    list_phase3_configs,
    load_briefs,
    load_prompts,
)


# ---------- load_prompts ----------

class TestLoadPrompts:
    def test_returns_all_8_sentence_tasks(self):
        templates = load_prompts()
        for task in SENTENCE_TASKS:
            assert task in templates, f"missing template: {task}"

    def test_returns_4_keyword_versions(self):
        templates = load_prompts()
        for v in ("A", "B", "C", "D"):
            assert f"{KEYWORD_TASK}:{v}" in templates

    def test_instruction_is_non_empty(self):
        templates = load_prompts()
        for t in SENTENCE_TASKS:
            assert templates[t].instruction.strip()
        assert templates[f"{KEYWORD_TASK}:A"].instruction.strip()

    def test_instructions_have_expected_keywords(self):
        # Sanity check that we parsed the right blocks.
        templates = load_prompts()
        assert "brand strategy" in templates["concept_relevant"].instruction.lower()
        assert "product or service" in templates["function_relevant"].instruction.lower()


# ---------- load_briefs ----------

class TestLoadBriefs:
    def test_returns_23_briefs(self):
        briefs = load_briefs()
        assert len(briefs) == 23

    def test_briefs_have_required_fields(self):
        briefs = load_briefs()
        for b in briefs:
            assert "current_name" in b
            assert "product" in b


# ---------- build_prompt ----------

class TestBuildPrompt:
    def _sample_brief(self):
        return {
            "current_name": "TestBrand",
            "business_category": "Test category",
            "product": "Product description here",
            "differentiators": "Diff text",
            "audience": "Audience text",
            "brand_strategy": "Strategy text",
            "personality": ["Lively", "Bold"],
            "priorities": ["product", "audience"],
            "keywords": ["a"] * 10,
        }

    def test_prompt_implied_has_no_brief(self):
        b = self._sample_brief()
        p = build_prompt("concept_relevant", PROMPT_IMPLIED, b)
        assert "BRIEF:" not in p
        assert "TestBrand" not in p

    def test_full_brief_includes_all_semantic_fields(self):
        b = self._sample_brief()
        p = build_prompt("concept_relevant", FULL_BRIEF, b)
        assert "BRIEF:" in p
        assert "Product description here" in p
        assert "Strategy text" in p

    def test_single_field_only_includes_that_field(self):
        b = self._sample_brief()
        p = build_prompt("function_relevant", ("product",), b)
        assert "Product description here" in p
        assert "Strategy text" not in p
        assert "Audience text" not in p

    def test_pair_includes_both_fields(self):
        b = self._sample_brief()
        p = build_prompt("function_relevant", ("product", "differentiators"), b)
        assert "Product description here" in p
        assert "Diff text" in p

    def test_personality_list_rendered_as_bullets(self):
        b = self._sample_brief()
        p = build_prompt("concept_relevant", ("personality",), b)
        assert "- Lively" in p
        assert "- Bold" in p

    def test_keyword_task_versions_differ(self):
        b = self._sample_brief()
        pA = build_prompt(KEYWORD_TASK, FULL_BRIEF, b, prompt_version="A")
        pB = build_prompt(KEYWORD_TASK, FULL_BRIEF, b, prompt_version="B")
        pC = build_prompt(KEYWORD_TASK, FULL_BRIEF, b, prompt_version="C")
        pD = build_prompt(KEYWORD_TASK, FULL_BRIEF, b, prompt_version="D")
        # All 4 prompts must differ (Phase 3 compression versions).
        prompts = [pA, pB, pC, pD]
        assert len(set(prompts)) == 4, "Versions A-D should all be distinct"
        # Each successive version should be no longer than the previous.
        assert len(pA) >= len(pB) >= len(pC) >= len(pD)
        # D should be much shorter than A (target ~10% length).
        assert len(pD) < 0.4 * len(pA)

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            build_prompt("nonexistent_task", FULL_BRIEF, self._sample_brief())


# ---------- is_compatible ----------

class TestIsCompatible:
    def test_baseline_always_compatible(self):
        for t in SENTENCE_TASKS:
            assert is_compatible(t, ())

    def test_keyword_accepts_anything(self):
        assert is_compatible(KEYWORD_TASK, ("audience",))
        assert is_compatible(KEYWORD_TASK, ())

    def test_function_relevant_accepts_product(self):
        assert is_compatible("function_relevant", ("product",))

    def test_function_relevant_rejects_audience_alone(self):
        # function prompt says "product or service description" — audience alone contradicts
        assert not is_compatible("function_relevant", ("audience",))

    def test_pair_compatible_if_any_field_compatible(self):
        # audience alone fails for function_relevant, but (audience, product) passes
        assert is_compatible("function_relevant", ("audience", "product"))

    def test_concept_relevant_accepts_brand_strategy(self):
        assert is_compatible("concept_relevant", ("brand_strategy",))

    def test_concept_relevant_rejects_audience(self):
        assert not is_compatible("concept_relevant", ("audience",))


# ---------- Config dataclass ----------

class TestConfig:
    def test_config_id_baseline(self):
        c = Config(task="concept_relevant", fields=())
        assert c.config_id == "A:_baseline"

    def test_config_id_single_field(self):
        c = Config(task="function_relevant", fields=("product",))
        assert c.config_id == "A:product"

    def test_config_id_pair(self):
        c = Config(task="function_relevant", fields=("audience", "product"))
        assert c.config_id == "A:audience+product"

    def test_config_id_keyword_versioned(self):
        c = Config(task=KEYWORD_TASK, fields=(FULL_BRIEF,), prompt_version="C")
        assert c.config_id == "C:_full_brief"


# ---------- Phase config lists ----------

class TestPhase0Configs:
    def test_phase0_count(self):
        # 8 sentence tasks × 2 configs + 1 keyword config = 17 configs
        configs = list_phase0_configs()
        assert len(configs) == 17

    def test_phase0_includes_keyword(self):
        configs = list_phase0_configs()
        assert any(c.task == KEYWORD_TASK for c in configs)

    def test_phase0_call_count_for_3_briefs_2_models(self):
        # Plan claims 102 calls. Let's verify: 17 configs * 3 briefs * 2 models = 102.
        configs = list_phase0_configs()
        assert len(configs) * 3 * 2 == 102


class TestPhase1Configs:
    def test_phase1_default_runs_all_configs(self):
        # 8 tasks × (5 single + 2 baselines) = 56. Default is filter_incompatible=False
        # so we run everything — let the data show low cosine for poor matches.
        configs = list_phase1_configs()
        assert len(configs) == 8 * 7  # 56

    def test_phase1_filter_flag_still_works(self):
        # is_compatible() is still available; opting in to filter still drops configs.
        filtered = list_phase1_configs(filter_incompatible=True)
        unfiltered = list_phase1_configs(filter_incompatible=False)
        assert len(filtered) < len(unfiltered)


class TestPhase2Configs:
    def test_phase2_includes_targeted_metadata(self):
        configs = list_phase2_configs()
        assert any("business_category" in c.fields for c in configs)


class TestPhase3Configs:
    def test_phase3_has_4_versions_keyword_only(self):
        configs = list_phase3_configs()
        assert len(configs) == 4
        assert all(c.task == KEYWORD_TASK for c in configs)
        versions = {c.prompt_version for c in configs}
        assert versions == {"A", "B", "C", "D"}


class TestStageDedup:
    def test_stage_a_has_no_duplicate_config_ids(self):
        configs = list_configs_for_stage("stage_a")
        seen = set()
        for c in configs:
            key = (c.task, c.config_id)
            assert key not in seen, f"duplicate: {key}"
            seen.add(key)
