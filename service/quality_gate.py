"""Offline quality gate for evidence-grounded AI evaluation reports.

The gate consumes a JSON report produced by an evaluation run.  It intentionally
does not generate metrics itself: the report must carry the dataset version and
the evaluator provenance, making a green build auditable.

Usage::

    python -m service.quality_gate quality_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_THRESHOLDS = {
    "groundedness_floor": 0.90,
    "citation_completeness_floor": 0.95,
    "unsupported_claim_rate_ceiling": 0.05,
    "entity_resolution_f1_floor": 0.90,
    "judge_weighted_kappa_floor": 0.60,
    "source_acceptable_rate_floor": 0.90,
}


def check_report(report: dict, thresholds: dict | None = None) -> tuple[bool, list[str]]:
    """Validate schema, provenance and configured quality thresholds."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    errors: list[str] = []
    required = ["schema", "dataset_version", "metrics", "provenance"]
    for field in required:
        if field not in report:
            errors.append(f"missing required field: {field}")
    if errors:
        return False, errors
    if report["schema"] != "quality-report/v1":
        errors.append("unsupported schema: expected quality-report/v1")
    provenance = report["provenance"]
    for field in ("evaluator_version", "golden_manifest_sha256"):
        if not provenance.get(field):
            errors.append(f"missing provenance: {field}")

    metrics = report["metrics"]
    checks = [
        ("groundedness", "groundedness_floor", lambda v, t: v >= t),
        ("citation_completeness", "citation_completeness_floor", lambda v, t: v >= t),
        ("unsupported_claim_rate", "unsupported_claim_rate_ceiling", lambda v, t: v <= t),
        ("entity_resolution_f1", "entity_resolution_f1_floor", lambda v, t: v >= t),
        ("judge_weighted_kappa", "judge_weighted_kappa_floor", lambda v, t: v >= t),
        ("source_acceptable_rate", "source_acceptable_rate_floor", lambda v, t: v >= t),
    ]
    for metric, threshold, predicate in checks:
        if metric not in metrics:
            errors.append(f"missing metric: {metric}")
            continue
        value = float(metrics[metric])
        if not predicate(value, limits[threshold]):
            errors.append(f"{metric}={value:.4f} violates {threshold}={limits[threshold]:.4f}")
    return not errors, errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check a quality-report/v1 JSON file.")
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    passed, messages = check_report(report)
    print("QUALITY GATE: PASS" if passed else "QUALITY GATE: FAIL")
    for message in messages:
        print(f"- {message}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
