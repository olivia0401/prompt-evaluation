"""Create or verify a versioned manifest for golden evaluation files.

Examples:
    python -m scripts.build_golden_manifest --root data/golden --output data/golden/manifest.json --version 2026.08.22
    python -m scripts.build_golden_manifest --verify data/golden/manifest.json --root data/golden
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.golden_manifest import build_manifest, verify_manifest, write_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--version", default="unversioned")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)

    if args.verify:
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        errors = verify_manifest(args.root, manifest)
        for error in errors:
            print(f"FAIL: {error}")
        if not errors:
            print(f"PASS: {manifest.get('version', 'unknown')}")
        return 1 if errors else 0

    if not args.output:
        parser.error("--output is required when --verify is not used")
    files = [p for p in args.root.rglob("*") if p.is_file() and p != args.output]
    manifest = build_manifest(args.root, files, args.version)
    write_manifest(args.output, manifest)
    print(f"Wrote {len(files)} file entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
