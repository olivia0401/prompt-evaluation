"""Aggregate repeated quality reports into a stability report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev


def aggregate(paths: list[Path]) -> dict:
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    versions = {r.get("dataset_version") for r in reports}
    if len(versions) != 1:
        raise ValueError("all reports must use the same dataset_version")
    metric_names = sorted(set().union(*(r.get("metrics", {}).keys() for r in reports)))
    metrics = {}
    for name in metric_names:
        values = [float(r["metrics"][name]) for r in reports if name in r.get("metrics", {})]
        if len(values) != len(reports):
            raise ValueError(f"metric {name!r} is missing from at least one report")
        metrics[name] = {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0,
                         "min": min(values), "max": max(values), "n": len(values)}
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return {
        "schema": "quality-stability-report/v1",
        "dataset_version": reports[0]["dataset_version"],
        "runs": len(reports),
        "metrics": metrics,
        "provenance": {"source_reports_sha256": digest.hexdigest()},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = aggregate(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote stability report for {result['runs']} runs -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
