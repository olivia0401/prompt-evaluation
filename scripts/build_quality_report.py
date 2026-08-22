"""Build a quality-report/v1 from annotated evidence-evaluation examples.

Input JSON shape::

    {
      "dataset_version": "golden-2026.08.22",
      "examples": [{
        "claims": [{"claim_id": "c1", "evidence_ids": ["s1"]}],
        "gold_claim_evidence": {"c1": ["s1"]},
        "evidence": [{"evidence_id": "s1", "source_quality": 5}],
        "entities": {"predicted": {"m1": "e1"}, "gold": {"m1": "e1"}},
        "source_quality": {"predicted": {"s1": 5}, "gold": {"s1": 4}},
        "judge": {"human": [4], "judge": [5]}
      }]
    }

Fields that are not relevant to a particular example may be omitted, but the
quality gate will require all aggregate metrics before allowing a release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from src.golden_manifest import sha256_file
from src.quality_evaluators import (
    Claim,
    Evidence,
    calibrate_judge,
    score_entity_resolution,
    score_evidence_chain,
    score_source_quality,
)


def _average(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        raise ValueError(f"no values available for metric: {key}")
    return mean(values)


def build_report(payload: dict, manifest_path: Path, evaluator_version: str) -> dict:
    examples = payload.get("examples") or []
    if not examples:
        raise ValueError("input must contain at least one annotated example")
    evidence_rows, entity_rows, source_rows, judge_rows = [], [], [], []
    for example in examples:
        claims = [Claim(c["claim_id"], tuple(c.get("evidence_ids", [])), bool(c.get("factual", True)))
                  for c in example.get("claims", [])]
        evidence = {
            e["evidence_id"]: Evidence(e["evidence_id"], int(e["source_quality"]))
            for e in example.get("evidence", [])
        }
        if claims:
            evidence_rows.append(score_evidence_chain(
                claims, example.get("gold_claim_evidence", {}), evidence
            ))
        entities = example.get("entities")
        if entities:
            entity_rows.append(score_entity_resolution(entities["predicted"], entities["gold"]))
        source = example.get("source_quality")
        if source:
            source_rows.append(score_source_quality(source["predicted"], source["gold"]))
        judge = example.get("judge")
        if judge:
            judge_rows.append(calibrate_judge(judge["human"], judge["judge"]))

    metrics = {
        "groundedness": _average(evidence_rows, "groundedness"),
        "citation_completeness": _average(evidence_rows, "citation_completeness"),
        "unsupported_claim_rate": _average(evidence_rows, "unsupported_claim_rate"),
        "entity_resolution_f1": _average(entity_rows, "f1"),
        "source_acceptable_rate": _average(source_rows, "acceptable_rate"),
        "judge_weighted_kappa": _average(judge_rows, "weighted_kappa"),
    }
    return {
        "schema": "quality-report/v1",
        "dataset_version": payload.get("dataset_version", "unknown"),
        "metrics": metrics,
        "provenance": {
            "evaluator_version": evaluator_version,
            "golden_manifest_sha256": sha256_file(manifest_path),
            "examples": len(examples),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-version", default="quality-evaluators-1")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = build_report(payload, args.manifest, args.evaluator_version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} for {report['provenance']['examples']} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
