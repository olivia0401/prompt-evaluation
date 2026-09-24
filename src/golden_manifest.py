"""Version and integrity helpers for golden evaluation datasets."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(root: Path, files: list[Path], version: str) -> dict:
    """Build a stable manifest; paths are stored relative to ``root``."""
    entries = []
    for path in sorted(files, key=lambda p: str(p)):
        resolved = path.resolve()
        if not resolved.is_file() or root.resolve() not in resolved.parents:
            raise ValueError(f"file is outside manifest root: {path}")
        entries.append({
            "path": str(resolved.relative_to(root.resolve())).replace("\\", "/"),
            "sha256": sha256_file(resolved),
            "bytes": resolved.stat().st_size,
        })
    return {
        "schema": "golden-manifest/v1",
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": entries,
    }


def verify_manifest(root: Path, manifest: dict) -> list[str]:
    """Return human-readable drift errors; an empty list means no drift."""
    errors = []
    for entry in manifest.get("files", []):
        path = root / entry["path"]
        if not path.is_file():
            errors.append(f"missing: {entry['path']}")
            continue
        actual = sha256_file(path)
        if actual != entry.get("sha256"):
            errors.append(f"changed: {entry['path']}")
    return errors


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the manifest is itself hashed into every report, so its
    # bytes must not depend on the OS that wrote it.
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
