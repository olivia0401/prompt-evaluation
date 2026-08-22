from src.golden_manifest import build_manifest, verify_manifest
from src.quality_evaluators import (
    Claim,
    Evidence,
    calibrate_judge,
    score_entity_resolution,
    score_evidence_chain,
    score_source_quality,
)


def test_evidence_chain_distinguishes_missing_and_wrong_citations():
    claims = [
        Claim("c1", ("s1",)),
        Claim("c2", ()),
        Claim("c3", ("s2",)),
    ]
    evidence = {
        "s1": Evidence("s1", 5),
        "s2": Evidence("s2", 2),
    }
    result = score_evidence_chain(claims, {"c1": {"s1"}, "c2": {"s1"}, "c3": {"s3"}}, evidence)
    assert result["citation_completeness"] == 2 / 3
    assert result["groundedness"] == 1 / 3
    assert result["unsupported_claim_rate"] == 2 / 3


def test_entity_resolution_uses_cluster_metrics():
    predicted = {"m1": "e1", "m2": "e1", "m3": "e2"}
    gold = {"m1": "g1", "m2": "g2", "m3": "g2"}
    result = score_entity_resolution(predicted, gold)
    assert result["mentions"] == 3
    assert 0 < result["f1"] < 1
    assert result["accuracy"] == 0.0


def test_source_quality_and_judge_calibration_are_ordinal():
    source = score_source_quality({"s1": 5, "s2": 3}, {"s1": 4, "s2": 3})
    assert source["mae"] == 0.5
    assert source["acceptable_rate"] == 1.0
    judge = calibrate_judge([1, 3, 5], [1, 4, 4])
    assert judge["n"] == 3
    assert judge["within_one_rate"] == 1.0
    assert 0 <= judge["weighted_kappa"] <= 1


def test_golden_manifest_detects_drift(tmp_path):
    data = tmp_path / "golden.jsonl"
    data.write_text('{"id":"x"}\n', encoding="utf-8")
    manifest = build_manifest(tmp_path, [data], "2026.08.22")
    assert verify_manifest(tmp_path, manifest) == []
    data.write_text('{"id":"changed"}\n', encoding="utf-8")
    assert verify_manifest(tmp_path, manifest) == ["changed: golden.jsonl"]
