from service.quality_gate import check_report


def _report(**metrics):
    return {
        "schema": "quality-report/v1",
        "dataset_version": "golden-2026.08.22",
        "metrics": {
            "groundedness": 0.95,
            "citation_completeness": 0.98,
            "unsupported_claim_rate": 0.02,
            "entity_resolution_f1": 0.94,
            "judge_weighted_kappa": 0.72,
            "source_acceptable_rate": 0.93,
            **metrics,
        },
        "denominators": {
            "claims": 120, "mentions": 340, "sources": 95, "judge_ratings": 40,
        },
        "provenance": {
            "evaluator_version": "quality-evaluators-1",
            "golden_manifest_sha256": "abc123",
        },
    }


def test_quality_gate_passes_with_provenance_and_good_metrics():
    passed, errors = check_report(_report())
    assert passed
    assert errors == []


def test_quality_gate_blocks_regression():
    passed, errors = check_report(_report(groundedness=0.80))
    assert not passed
    assert any("groundedness" in error for error in errors)


def test_quality_gate_requires_provenance():
    report = _report()
    report["provenance"] = {}
    passed, errors = check_report(report)
    assert not passed
    assert any("provenance" in error for error in errors)


def test_quality_gate_blocks_on_insufficient_sample():
    """Good-looking metrics on a tiny denominator must not pass.

    This is the "green tick is lying" case: every threshold is cleared, but on
    9 claims and 4 ratings none of those numbers carries information. The gate
    has to distinguish "not enough data to judge" from "judged and passed".
    """
    report = _report()
    report["denominators"] = {"claims": 9, "mentions": 12, "sources": 7,
                              "judge_ratings": 4}
    passed, errors = check_report(report)
    assert not passed
    assert any("insufficient sample: claims" in e for e in errors)
    assert any("insufficient sample: judge_ratings" in e for e in errors)


def test_quality_gate_requires_denominators():
    """A report that omits its denominators cannot be verified, so it fails."""
    report = _report()
    report.pop("denominators")
    passed, errors = check_report(report)
    assert not passed
    assert any("missing denominator" in e for e in errors)
