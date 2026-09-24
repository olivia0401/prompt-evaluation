"""The annotation pipeline: queue -> annotations -> report -> gate.

Covers the honesty properties specifically, because they are the ones that fail
silently: a metric that vanishes without being declared, an exclusion with no
reason, and a sample-size floor firing for a metric that does not apply.
"""
from __future__ import annotations

import json

import pytest

from scripts.build_quality_report import build_report
from service.quality_gate import check_report


@pytest.fixture()
def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema": "golden-manifest/v1"}', encoding="utf-8")
    return path


def _example(idx: int, cited: list[str], supported: list[str],
             quality: dict[str, int]) -> dict:
    return {
        "id": f"pc-{idx:03d}@tester",
        "stratum": "scam/danger",
        "claims": [{"claim_id": "verdict", "evidence_ids": cited}],
        "gold_claim_evidence": {"verdict": supported},
        "evidence": [{"evidence_id": e, "source_quality": quality[e]} for e in cited],
    }


NOT_APPLICABLE = {
    "entity_resolution_f1": "no entity mentions in this corpus",
    "source_acceptable_rate": "the engine predicts no source quality",
    "judge_weighted_kappa": "no LLM judge scores this corpus",
}


def _payload(examples: list[dict], **overrides) -> dict:
    return {"dataset_version": "test-golden", "not_applicable": NOT_APPLICABLE,
            "examples": examples, **overrides}


def test_declared_exclusions_keep_the_report_buildable(manifest):
    report = build_report(_payload([
        _example(0, ["s0", "s1"], ["s0"], {"s0": 5, "s1": 2}),
        _example(1, ["s0"], ["s0"], {"s0": 4}),
    ]), manifest, "v1")

    assert set(report["metrics"]) == {
        "groundedness", "citation_completeness", "unsupported_claim_rate"}
    assert set(report["not_applicable"]) == set(NOT_APPLICABLE)
    # Excluded metrics carry a reason through to the report, not just a flag.
    assert all(report["not_applicable"].values())


def test_an_undeclared_missing_judge_is_an_error_not_a_silent_skip(manifest):
    payload = _payload([_example(0, ["s0"], ["s0"], {"s0": 5})], not_applicable={})
    with pytest.raises(ValueError, match="declare it in not_applicable"):
        build_report(payload, manifest, "v1")


def test_an_exclusion_without_a_reason_is_rejected(manifest):
    payload = _payload([_example(0, ["s0"], ["s0"], {"s0": 5})],
                       not_applicable={"judge_weighted_kappa": "   "})
    with pytest.raises(ValueError, match="no reason"):
        build_report(payload, manifest, "v1")


def test_an_uncited_verdict_is_ungrounded_even_when_it_is_correct(manifest):
    """The 16 real queue items that cite nothing. Correctness is a separate axis."""
    report = build_report(_payload([
        _example(0, ["s0"], ["s0"], {"s0": 5}),
        {**_example(1, [], [], {}), "claims": [{"claim_id": "verdict", "evidence_ids": []}]},
    ]), manifest, "v1")
    assert report["metrics"]["citation_completeness"] == 0.5
    assert report["metrics"]["groundedness"] == 0.5
    assert report["metrics"]["unsupported_claim_rate"] == 0.5


def test_a_claim_resting_only_on_weak_signals_is_not_grounded(manifest):
    """Cited, gold-accepted, and still below the source-quality floor."""
    report = build_report(_payload([
        _example(0, ["s0"], ["s0"], {"s0": 1}),   # accepted but quality 1
    ]), manifest, "v1")
    assert report["metrics"]["citation_completeness"] == 1.0
    assert report["metrics"]["groundedness"] == 0.0


def test_gate_does_not_demand_judge_ratings_for_a_corpus_with_no_judge(manifest):
    """A floor guarding a not-applicable metric must not fire.

    Before this, a dataset with no judge failed on 'judge_ratings=0 < 30' — the
    gate going red for a reason unrelated to quality, which is exactly how a
    team learns to stop reading it.
    """
    report = build_report(_payload(
        [_example(i, ["s0", "s1"], ["s0", "s1"], {"s0": 5, "s1": 4}) for i in range(20)]
    ), manifest, "v1")
    report["provenance"]["golden_manifest_sha256"] = "abc"

    _passed, errors = check_report(report)
    assert not any("judge_ratings" in e for e in errors)
    # It still fails on the claim count, which IS a real reason.
    assert any("insufficient sample: claims" in e for e in errors)


def test_gate_passes_a_well_grounded_corpus_with_declared_exclusions(manifest):
    report = build_report(_payload(
        [_example(i, ["s0", "s1"], ["s0", "s1"], {"s0": 5, "s1": 4}) for i in range(40)]
    ), manifest, "v1")
    report["provenance"]["golden_manifest_sha256"] = "abc"

    passed, errors = check_report(report)
    assert passed, errors
    assert report["denominators"]["claims"] == 40


def test_the_exported_queue_is_real_and_well_formed():
    """Guards the committed queue itself: schema, strata, and the uncited count."""
    from src import config as cfg

    queue_path = cfg.PROJECT_ROOT / "data" / "golden" / "queue.jsonl" \
        if hasattr(cfg, "PROJECT_ROOT") else None
    if queue_path is None or not queue_path.exists():
        pytest.skip("queue.jsonl not present (export it from parent-check)")

    lines = [json.loads(l) for l in queue_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    header, items = lines[0], lines[1:]
    assert header["schema"] == "annotation-queue/v1"
    assert len(items) == header["items"]
    for item in items:
        assert item["claim"]["claim_id"] == "verdict"
        assert isinstance(item["evidence"], list)
        assert "/" in item["stratum"]
    uncited = sum(1 for i in items if not i["evidence"])
    assert uncited > 0, "a queue with no uncited verdicts cannot exercise groundedness"
