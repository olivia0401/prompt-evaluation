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
    """Unweighted (macro) mean over examples — reported for visibility only."""
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        raise ValueError(f"no values available for metric: {key}")
    return mean(values)


def _pooled(rows: list[dict], key: str, weight_key: str) -> float:
    """Denominator-weighted (micro) rate — the number the gate acts on.

    A plain mean over examples silently gives an example with 1 claim the same
    say as one with 40. For rates whose denominator varies per example, that is
    not "the rate on this dataset", it is an average of incomparable fractions.
    Both are reported; the gate uses this one.
    """
    num = den = 0.0
    for row in rows:
        if key not in row or weight_key not in row:
            continue
        w = float(row[weight_key])
        num += float(row[key]) * w
        den += w
    if den == 0:
        raise ValueError(f"no weighted values available for metric: {key}")
    return num / den


def build_report(payload: dict, manifest_path: Path, evaluator_version: str) -> dict:
    examples = payload.get("examples") or []
    if not examples:
        raise ValueError("input must contain at least one annotated example")
    evidence_rows, entity_rows, source_rows = [], [], []
    # Judge ratings are POOLED, never averaged per example. Cohen's kappa is a
    # chance-corrected agreement statistic over a contingency table: it is not
    # defined on 1-2 ratings (a single agreeing pair returns kappa=1.0 because
    # expected agreement is also 1.0), and a mean of per-example kappas is not
    # the kappa of the dataset. Collect every rating, compute one kappa.
    judge_human: list[int] = []
    judge_model: list[int] = []
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
            human, model = list(judge["human"]), list(judge["judge"])
            if len(human) != len(model):
                raise ValueError(
                    f"example {example.get('id', '?')}: {len(human)} human ratings "
                    f"vs {len(model)} judge ratings"
                )
            judge_human.extend(int(x) for x in human)
            judge_model.extend(int(x) for x in model)

    # A metric with no data is either an oversight or a decision. Declaring it
    # in the payload's `not_applicable` map (with a reason) makes it a decision;
    # leaving it silently absent makes the gate fail, which is the right default.
    not_applicable = dict(payload.get("not_applicable") or {})
    for name, reason in not_applicable.items():
        if not str(reason).strip():
            raise ValueError(
                f"not_applicable['{name}'] has no reason. An exclusion without a "
                f"stated reason is indistinguishable from an omission."
            )

    judge_stats = None
    if judge_human:
        judge_stats = calibrate_judge(judge_human, judge_model)
    elif "judge_weighted_kappa" not in not_applicable:
        raise ValueError(
            "no judge/human ratings available for calibration. If this dataset "
            "genuinely has no judge, declare it in not_applicable with a reason."
        )

    candidates = {
        "groundedness": (evidence_rows, "groundedness", "claims"),
        "citation_completeness": (evidence_rows, "citation_completeness", "claims"),
        "unsupported_claim_rate": (evidence_rows, "unsupported_claim_rate", "claims"),
        "entity_resolution_f1": (entity_rows, "f1", "mentions"),
        "source_acceptable_rate": (source_rows, "acceptable_rate", "sources"),
    }
    metrics = {}
    for name, (rows, key, weight) in candidates.items():
        if name in not_applicable:
            continue
        metrics[name] = _pooled(rows, key, weight)
    if judge_stats is not None:
        metrics["judge_weighted_kappa"] = judge_stats["weighted_kappa"]
    # Macro means kept alongside, never gated on: a large gap between macro and
    # micro means the per-example denominators are lopsided, which is itself
    # worth seeing before anyone quotes the headline number.
    macro = {
        name: _average(rows, key)
        for name, (rows, key, _weight) in candidates.items()
        if name not in not_applicable
    }
    return {
        "schema": "quality-report/v1",
        "dataset_version": payload.get("dataset_version", "unknown"),
        "metrics": metrics,
        "macro_metrics": macro,
        "not_applicable": not_applicable,
        "denominators": {
            "claims": int(sum(r["claims"] for r in evidence_rows)),
            "mentions": int(sum(r["mentions"] for r in entity_rows)),
            "sources": int(sum(r["sources"] for r in source_rows)),
            "judge_ratings": len(judge_human),
        },
        "judge_calibration": judge_stats,
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
