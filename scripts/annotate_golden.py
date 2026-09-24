"""Evidence annotation harness — turns product output into a golden set.

    python -m scripts.annotate_golden --annotator olivia            # annotate
    python -m scripts.annotate_golden --annotator olivia --status   # progress
    python -m scripts.annotate_golden --agreement                   # 2-annotator kappa
    python -m scripts.annotate_golden --build                       # -> annotations.json

Input is an ``annotation-queue/v1`` file (``data/golden/queue.jsonl``), produced
by whatever system is under evaluation. Each item carries one claim and the
evidence that system *cited* for it. Annotation decides the thing no script can:

  1. Does each cited signal **genuinely support** the claim, or did it merely
     match? A keyword firing on a benign message is a citation without support —
     exactly the unsupported-claim pattern the metrics exist to catch.
  2. How strong is that signal as evidence, 1-5? Groundedness only counts
     evidence at or above the source-quality floor, so a claim resting entirely
     on weak signals is correctly scored as ungrounded even when it is right.
  3. Was the claim itself correct? Recorded separately, because "right for the
     wrong reason" is a distinct and more dangerous failure than "wrong".

Two annotators
--------------
Run it twice with different ``--annotator`` values, then ``--agreement`` to get
inter-annotator kappa on the support decision. Below about 0.6 the disagreement
is in the *instructions*, not the annotators, and the fix is to sharpen the
support criterion rather than to average two people guessing differently. A
single-annotator golden set is one person's opinion, and its metrics inherit
that without saying so.

``--build`` writes ``data/golden/annotations.json``. Metrics with no data in
this corpus are declared not-applicable **with a reason** rather than omitted;
the gate treats an undeclared missing metric as a failure, so silence can never
be mistaken for a pass.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src import config as cfg
from src.quality_evaluators import weighted_kappa

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GOLDEN_DIR = cfg.PROJECT_ROOT / "data" / "golden" if hasattr(cfg, "PROJECT_ROOT") \
    else Path(__file__).resolve().parents[1] / "data" / "golden"
QUEUE_PATH = GOLDEN_DIR / "queue.jsonl"
RAW_PATH = GOLDEN_DIR / "annotations_raw.jsonl"
BUILT_PATH = GOLDEN_DIR / "annotations.json"

# Metrics this corpus genuinely cannot support, and why. Declared, never omitted.
NOT_APPLICABLE = {
    "entity_resolution_f1":
        "this corpus has no entity mentions to resolve; entity resolution is "
        "evaluated on the investigative corpus, not on message triage",
    "source_acceptable_rate":
        "the system under evaluation emits no source-quality prediction of its "
        "own, so there is nothing to compare the annotator's rating against",
    "judge_weighted_kappa":
        "no LLM judge scores this corpus; judge calibration is measured "
        "separately via scripts.compute_kappa on the prompt-evaluation corpus",
}

SUPPORT_RUBRIC = """\
  For each cited signal, answer: does it genuinely support THIS verdict?

    y  Yes — a reader shown only this signal would reach the same verdict.
    n  No  — it matched, but it does not carry the verdict. A keyword that
           appears in perfectly ordinary messages is a citation, not support.

  Then rate the signal as evidence, 1-5:
    5  Decisive on its own (an unsolicited payment demand, a URL shortener
       in a message claiming to be from a bank).
    3  Meaningful in combination, weak alone.
    1  Near-noise; appears constantly in benign text."""


def _wrap(text: str, indent: str = "     ", width: int = 88) -> str:
    if not text:
        return indent + "(empty)"
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


# ------------------------------------------------------------------ queue / raw

def load_queue() -> list[dict]:
    if not QUEUE_PATH.exists():
        raise SystemExit(
            f"Missing {QUEUE_PATH}.\n"
            f"Export one from the system under evaluation, e.g.:\n"
            f'    cd ../parent-check && python export_eval_queue.py '
            f'--output "../prompt test/data/golden/queue.jsonl"'
        )
    items = []
    with open(QUEUE_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema"):          # header line
                continue
            items.append(row)
    return items


def load_raw() -> list[dict]:
    if not RAW_PATH.exists():
        return []
    rows = []
    with open(RAW_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def append_raw(row: dict) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RAW_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------- annotating

def _ask(prompt: str, valid: set[str]) -> str | None:
    while True:
        try:
            raw = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Interrupted — everything annotated so far is saved.")
            return None
        if raw in ("q", "quit"):
            return None
        if raw in valid:
            return raw
        print(f"  Expected one of {sorted(valid)} (or q to save and quit).")


def annotate_item(item: dict, position: int, total: int) -> dict | None:
    print("\n" + "=" * 92)
    print(f"  [{position}/{total}]  {item['item_id']}  ·  channel: {item['channel']}"
          f"  ·  stratum: {item['stratum']}")
    print("=" * 92)
    print("\n  MESSAGE")
    print(_wrap(item["text"]))
    print("\n  CLAIM UNDER EVALUATION")
    print(_wrap(item["claim"]["text"]))

    evidence = item.get("evidence") or []
    supported, qualities = [], {}

    if not evidence:
        print("\n  CITED EVIDENCE: none.")
        print("  A verdict with no citation cannot be grounded, whether or not it")
        print("  happens to be right. Nothing to mark here.")
    else:
        print(f"\n  CITED EVIDENCE ({len(evidence)} signal(s))")
        for ev in evidence:
            print(f"\n    [{ev['evidence_id']}]  {ev['text']}")
            answer = _ask("      supports this verdict? (y/n, ? = rubric, q = quit): ",
                          {"y", "n", "?"})
            if answer is None:
                return None
            if answer == "?":
                print("\n" + SUPPORT_RUBRIC + "\n")
                answer = _ask("      supports this verdict? (y/n, q = quit): ", {"y", "n"})
                if answer is None:
                    return None
            quality = _ask("      strength as evidence 1-5 (q = quit): ",
                           {"1", "2", "3", "4", "5"})
            if quality is None:
                return None
            qualities[ev["evidence_id"]] = int(quality)
            if answer == "y":
                supported.append(ev["evidence_id"])

    print()
    correct = _ask("  Was the verdict itself correct? (y/n, q = quit): ", {"y", "n"})
    if correct is None:
        return None

    return {
        "item_id": item["item_id"],
        "stratum": item["stratum"],
        "cited": [ev["evidence_id"] for ev in evidence],
        "supported": supported,
        "quality": qualities,
        "verdict_correct": correct == "y",
    }


def cmd_annotate(annotator: str) -> int:
    items = load_queue()
    done = {r["item_id"] for r in load_raw() if r["annotator"] == annotator}
    queue = [i for i in items if i["item_id"] not in done]

    if not queue:
        print(f"\n{annotator} has annotated all {len(items)} items.")
        return cmd_status()

    print("\n" + "=" * 92)
    print(f"  EVIDENCE ANNOTATION  ·  annotator: {annotator}  ·  {len(queue)} remaining")
    print("=" * 92)
    print("\n" + SUPPORT_RUBRIC)
    print("\n  Ctrl-C or q is safe: each item is written as you finish it.\n")

    saved = 0
    for i, item in enumerate(queue, start=1):
        result = annotate_item(item, i, len(queue))
        if result is None:
            break
        append_raw({**result, "annotator": annotator,
                    "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        saved += 1

    print(f"\n\nSaved {saved} annotation(s) -> {RAW_PATH}")
    return cmd_status()


def cmd_status() -> int:
    items = load_queue()
    raw = load_raw()
    by_annotator = Counter(r["annotator"] for r in raw)
    print(f"\nQueue: {len(items)} items")
    if not by_annotator:
        print("  No annotations yet.")
    for who, n in sorted(by_annotator.items()):
        print(f"  {who:<16} {n}/{len(items)}")
    if len(by_annotator) < 2:
        print("\n  Only one annotator so far. A second one over an overlapping subset")
        print("  turns 'my opinion' into a measured agreement — run:")
        print("      python -m scripts.annotate_golden --annotator <second-person>")
    return 0


# ------------------------------------------------------------------- agreement

def cmd_agreement() -> int:
    """Inter-annotator agreement on the per-signal support decision."""
    raw = load_raw()
    by_annotator: dict[str, dict[str, dict]] = {}
    for row in raw:
        by_annotator.setdefault(row["annotator"], {})[row["item_id"]] = row

    if len(by_annotator) < 2:
        print("\nNeed two annotators before agreement means anything.")
        return 1

    names = sorted(by_annotator)
    a, b = names[0], names[1]
    shared = sorted(set(by_annotator[a]) & set(by_annotator[b]))
    if not shared:
        print(f"\n{a} and {b} share no annotated items.")
        return 1

    # Support is binary; encode as 1/5 so the same ordinal kappa is reused
    # rather than introducing a second agreement statistic to keep straight.
    ra, rb = [], []
    for item_id in shared:
        ea, eb = by_annotator[a][item_id], by_annotator[b][item_id]
        for ev_id in ea["cited"]:
            ra.append(5 if ev_id in ea["supported"] else 1)
            rb.append(5 if ev_id in eb.get("supported", []) else 1)

    print(f"\nInter-annotator agreement ({a} vs {b})")
    print(f"  shared items    : {len(shared)}")
    print(f"  signal decisions: {len(ra)}")
    if len(ra) < 20:
        print("  Too few shared decisions to report a kappa (need ~20+).")
        return 1
    try:
        kappa = weighted_kappa(ra, rb, weights="linear")
    except ValueError as exc:
        print(f"  Kappa undefined: {exc}")
        return 1
    raw_agree = sum(x == y for x, y in zip(ra, rb)) / len(ra)
    print(f"  raw agreement   : {raw_agree:.1%}")
    print(f"  kappa (linear)  : {kappa:.3f}")
    if kappa < 0.6:
        print("\n  Below 0.6. The disagreement is almost certainly in the support")
        print("  criterion, not in the annotators — sharpen SUPPORT_RUBRIC and")
        print("  re-annotate rather than averaging two different definitions.")
    return 0


# ----------------------------------------------------------------------- build

def cmd_build(annotator: str | None, dataset_version: str) -> int:
    items = {i["item_id"]: i for i in load_queue()}
    raw = load_raw()
    if annotator:
        raw = [r for r in raw if r["annotator"] == annotator]
    if not raw:
        print("\nNo annotations to build from.")
        return 1

    # Last annotation per (annotator, item) wins; multiple annotators contribute
    # separate examples rather than being silently merged, so a disagreement
    # shows up as variance instead of disappearing into an average.
    latest: dict[tuple[str, str], dict] = {}
    for row in raw:
        latest[(row["annotator"], row["item_id"])] = row

    examples = []
    for (who, item_id), row in sorted(latest.items()):
        item = items.get(item_id)
        if item is None:
            continue
        examples.append({
            "id": f"{item_id}@{who}",
            "stratum": row["stratum"],
            "claims": [{"claim_id": "verdict", "evidence_ids": row["cited"]}],
            "gold_claim_evidence": {"verdict": row["supported"]},
            "evidence": [
                {"evidence_id": ev_id, "source_quality": row["quality"].get(ev_id, 1)}
                for ev_id in row["cited"]
            ],
            "verdict_correct": row["verdict_correct"],
        })

    payload = {
        "dataset_version": dataset_version,
        "source": "parent-check via annotation-queue/v1",
        "annotators": sorted({who for who, _ in latest}),
        "not_applicable": NOT_APPLICABLE,
        "examples": examples,
    }
    BUILT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUILT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    strata = Counter(e["stratum"] for e in examples)
    correct = sum(e["verdict_correct"] for e in examples)
    print(f"\nWrote {len(examples)} examples -> {BUILT_PATH}")
    print(f"  annotators : {payload['annotators']}")
    print(f"  strata     : {dict(sorted(strata.items()))}")
    print(f"  verdict correct: {correct}/{len(examples)}")
    print(f"  declared not-applicable: {sorted(NOT_APPLICABLE)}")
    print("\nNext:")
    print("  python -m scripts.build_golden_manifest --root data/golden "
          "--output data/golden/manifest.json --version <version>")
    print("  python -m scripts.build_quality_report --input data/golden/annotations.json "
          "--manifest data/golden/manifest.json --output outputs/quality_report.json")
    print("  python -m service.quality_gate outputs/quality_report.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotator", default=None, help="Your identifier")
    ap.add_argument("--status", action="store_true", help="Progress, then exit")
    ap.add_argument("--agreement", action="store_true", help="Inter-annotator kappa")
    ap.add_argument("--build", action="store_true", help="Write annotations.json")
    ap.add_argument("--dataset-version", default="golden-parent-check-2026.08.22")
    args = ap.parse_args(argv)

    if args.status:
        return cmd_status()
    if args.agreement:
        return cmd_agreement()
    if args.build:
        return cmd_build(args.annotator, args.dataset_version)
    if not args.annotator:
        ap.error("--annotator is required to annotate (or pass --status / "
                 "--agreement / --build)")
    return cmd_annotate(args.annotator)


if __name__ == "__main__":
    raise SystemExit(main())
