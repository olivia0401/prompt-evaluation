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
