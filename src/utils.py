"""
Shared utilities: safe file writes, JSONL append, normalization.
"""
import json
import time
from pathlib import Path
from typing import Any, Union

import pandas as pd


def safe_to_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    mode: str,
    header: bool,
    max_retries: int = 6,
) -> None:
    """CSV write with PermissionError retry (handles Excel open on the file)."""
    delay = 0.5
    for attempt in range(max_retries):
        try:
            df.to_csv(path, mode=mode, header=header, index=False, encoding="utf-8")
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def append_jsonl(
    path: Union[str, Path],
    record: dict,
    max_retries: int = 6,
) -> None:
    """
    Primary storage: one JSON object per line, flushed each call.
    Retries on PermissionError (OneDrive lock).
    Use this for raw call results, not CSV — model outputs contain
    commas/quotes/newlines that break CSV in subtle ways.
    """
    delay = 0.5
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    for attempt in range(max_retries):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2


def read_jsonl(path: Union[str, Path]) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def word_count(text: str) -> int:
    """
    The single word-count rule for this entire experiment. Do not change.
    Whitespace-separated tokens after strip. Hyphenated terms count as 1.
    """
    return len(text.strip().split())


def normalize_keyword(kw: str) -> str:
    """
    Keyword normalization (pre-stemming): lowercase, strip, collapse hyphens.
    Porter stemming happens in evaluators.py.
    """
    return kw.strip().lower().replace("-", "")


def make_config_id(fields: list[str]) -> str:
    """
    Stable config_id from a field list. Sorting makes (a, b) and (b, a)
    map to the same id — never change this format, or resume breaks.
    """
    if not fields:
        return "_baseline"
    return "+".join(sorted(fields))
