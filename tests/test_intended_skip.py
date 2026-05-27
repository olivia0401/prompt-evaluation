"""Offline test for the targeted-metadata "n/a by design" cell lookup."""
from scripts.build_xlsx import _intended_skip_cells


def test_business_category_alone_skipped_outside_category_task():
    """A:business_category was only enumerated for category_relevant.
    All other 7 sentence tasks must be in the skip set."""
    skip = _intended_skip_cells()
    bc_skips = {t for (t, cfg) in skip if cfg == "A:business_category"}
    assert "category_relevant" not in bc_skips
    assert bc_skips == {
        "concept_relevant", "position_relevant", "emotion_relevant",
        "function_relevant", "benefit_relevant", "feature_relevant",
        "context_relevant",
    }


def test_business_category_plus_product_skipped_outside_function_task():
    skip = _intended_skip_cells()
    bcp_skips = {t for (t, cfg) in skip
                 if cfg == "A:business_category+product"}
    assert "function_relevant" not in bcp_skips
    assert bcp_skips == {
        "concept_relevant", "position_relevant", "emotion_relevant",
        "benefit_relevant", "category_relevant", "feature_relevant",
        "context_relevant",
    }


def test_semantic_only_targeted_combos_dont_produce_skips():
    """('context_relevant', ['audience']) is in TARGETED_METADATA_COMBOS, but
    'audience' is a semantic field (also enumerated as a single-field config
    across all tasks). It must NOT create n/a cells for the other tasks —
    those tasks DO have audience data from Phase 1 single-field enumeration."""
    skip = _intended_skip_cells()
    cfgs_in_skip = {cfg for (_, cfg) in skip}
    assert "A:audience" not in cfgs_in_skip


def test_total_skip_count_matches_design():
    """2 metadata-only configs × 7 other tasks = 14 skipped cells."""
    skip = _intended_skip_cells()
    assert len(skip) == 14


def test_skip_cells_use_full_config_id_format():
    """Heatmap pivot columns use 'A:<fields>' format — skip set must match
    that, not the raw fields representation."""
    skip = _intended_skip_cells()
    for _, cfg_id in skip:
        assert cfg_id.startswith("A:"), f"unexpected config_id format: {cfg_id}"
