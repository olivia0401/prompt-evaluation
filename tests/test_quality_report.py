import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_quality_report.py"
    spec = importlib.util.spec_from_file_location("build_quality_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_quality_report_has_provenance_and_all_metrics(tmp_path):
    module = _module()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    payload = {
        "dataset_version": "golden-test",
        "examples": [{
            "claims": [{"claim_id": "c1", "evidence_ids": ["s1"]}],
            "gold_claim_evidence": {"c1": ["s1"]},
            "evidence": [{"evidence_id": "s1", "source_quality": 5}],
            "entities": {"predicted": {"m1": "e1"}, "gold": {"m1": "e1"}},
            "source_quality": {"predicted": {"s1": 5}, "gold": {"s1": 5}},
            "judge": {"human": [4, 5], "judge": [4, 5]},
        }],
    }
    report = module.build_report(payload, manifest, "test-evaluator")
    assert report["schema"] == "quality-report/v1"
    assert report["dataset_version"] == "golden-test"
    assert report["metrics"]["groundedness"] == 1.0
    assert report["metrics"]["entity_resolution_f1"] == 1.0
    assert report["provenance"]["evaluator_version"] == "test-evaluator"
    assert len(report["provenance"]["golden_manifest_sha256"]) == 64
