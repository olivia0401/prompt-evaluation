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
    # Sample-size floors. A metric computed on 3 claims can clear any threshold
    # by luck, so a green tick on a tiny denominator is not evidence — it is an
    # absence of evidence wearing the same colour. The 30-rating floor for the
    # judge matches the plan's kappa rule (§2 of the evaluation plan).
    "min_claims": 30,
    "min_judge_ratings": 30,
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

    # A metric may be excluded only by an explicit, reasoned declaration. This
    # is the difference between "we decided this does not apply here" and "it
    # quietly went missing" — and the gate must never confuse the two.
    not_applicable = report.get("not_applicable") or {}
    for name, reason in not_applicable.items():
        if not str(reason).strip():
            errors.append(f"{name} declared not-applicable without a reason")

    # Sample-size floors before threshold checks: report which is which clearly,
    # because "not enough data to judge" and "judged and failed" are different
    # conversations with whoever is trying to ship.
    #
    # A floor is skipped when the metric it guards is declared not-applicable.
    # Demanding 30 judge ratings for a dataset that has no judge is the gate
    # failing for a reason unrelated to quality — which is how gates get muted.
    denominators = report.get("denominators") or {}
    guarded_by = {"claims": None, "judge_ratings": "judge_weighted_kappa"}
    for field, threshold in (("claims", "min_claims"),
                             ("judge_ratings", "min_judge_ratings")):
        guarded_metric = guarded_by.get(field)
        if guarded_metric and guarded_metric in not_applicable:
            continue
        limit = limits.get(threshold)
        if limit is None:
            continue
        actual = denominators.get(field)
        if actual is None:
            errors.append(f"missing denominator: {field} (cannot verify {threshold})")
        elif int(actual) < int(limit):
            errors.append(
                f"insufficient sample: {field}={actual} < {threshold}={limit} "
                f"— metrics are not evidence at this size"
            )

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
        if metric in not_applicable:
            continue
        if metric not in metrics:
            errors.append(
                f"missing metric: {metric} (declare it in not_applicable with a "
                f"reason if this dataset genuinely cannot produce it)"
            )
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
    # Print exclusions on a PASS too. A green tick that quietly covers fewer
    # metrics than the reader assumes is the exact failure this gate is for.
    for name, reason in sorted((report.get("not_applicable") or {}).items()):
        print(f"~ not evaluated: {name} - {reason}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
