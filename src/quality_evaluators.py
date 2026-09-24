"""Auditable quality metrics for evidence-grounded AI outputs.

This module deliberately evaluates *annotated contracts* rather than pretending
that lexical overlap proves truth.  A production evaluation row should contain:

* predicted claims and the evidence IDs cited for each claim;
* a versioned gold annotation mapping claims to acceptable evidence;
* entity mentions and canonical entity IDs;
* human and judge scores when judge calibration is being measured.

The functions are deterministic, dependency-light and safe to run offline.
They are intended to be used by batch evaluation and CI gates.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isclose
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_quality: int  # 1 (poor) .. 5 (excellent), from an annotation rubric


@dataclass(frozen=True)
class Claim:
    claim_id: str
    evidence_ids: tuple[str, ...]
    factual: bool = True


def _safe_rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def score_evidence_chain(
    predicted_claims: Sequence[Claim],
    gold_claim_evidence: Mapping[str, Iterable[str]],
    evidence: Mapping[str, Evidence],
    *,
    min_source_quality: int = 3,
) -> dict[str, float | int]:
    """Score citation completeness, evidence precision and groundedness.

    A factual predicted claim is grounded only when it cites at least one gold-
    accepted evidence item that exists and meets the source-quality threshold.
    This requires human/researcher annotations in ``gold_claim_evidence``; it
    does not claim to independently establish truth.
    """
    gold = {k: set(v) for k, v in gold_claim_evidence.items()}
    factual = [c for c in predicted_claims if c.factual]
    cited = [c for c in factual if c.evidence_ids]
    grounded = 0
    correct_links = 0
    predicted_links = 0
    for claim in factual:
        accepted = gold.get(claim.claim_id, set())
        valid_ids = {
            eid for eid in claim.evidence_ids
            if eid in evidence and evidence[eid].source_quality >= min_source_quality
        }
        grounded_ids = valid_ids & accepted
        grounded += bool(grounded_ids)
        correct_links += len(grounded_ids)
        predicted_links += len(claim.evidence_ids)

    return {
        "claims": len(factual),
        "citation_completeness": _safe_rate(len(cited), len(factual)),
        "evidence_precision": _safe_rate(correct_links, predicted_links),
        "groundedness": _safe_rate(grounded, len(factual)),
        "unsupported_claim_rate": _safe_rate(len(factual) - grounded, len(factual)),
    }


def score_entity_resolution(
    predicted: Mapping[str, str],
    gold: Mapping[str, str],
) -> dict[str, float | int]:
    """Score mention-to-entity assignments using accuracy and B-cubed F1."""
    ids = sorted(set(predicted) & set(gold))
    if not ids:
        return {"mentions": 0, "accuracy": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0}

    pred_groups: dict[str, set[str]] = defaultdict(set)
    gold_groups: dict[str, set[str]] = defaultdict(set)
    for mention_id in ids:
        pred_groups[predicted[mention_id]].add(mention_id)
        gold_groups[gold[mention_id]].add(mention_id)

    precision = recall = 0.0
    for mention_id in ids:
        p_cluster = pred_groups[predicted[mention_id]]
        g_cluster = gold_groups[gold[mention_id]]
        overlap = len(p_cluster & g_cluster)
        precision += overlap / len(p_cluster)
        recall += overlap / len(g_cluster)
    precision /= len(ids)
    recall /= len(ids)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "mentions": len(ids),
        "accuracy": _safe_rate(sum(predicted[i] == gold[i] for i in ids), len(ids)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def score_source_quality(
    predicted: Mapping[str, int],
    gold: Mapping[str, int],
) -> dict[str, float | int]:
    """Compare source-quality ratings on a 1--5 rubric."""
    ids = sorted(set(predicted) & set(gold))
    if not ids:
        return {"sources": 0, "exact_rate": 0.0, "mae": 0.0,
                "acceptable_rate": 0.0}
    errors = [abs(predicted[i] - gold[i]) for i in ids]
    return {
        "sources": len(ids),
        "exact_rate": _safe_rate(sum(predicted[i] == gold[i] for i in ids), len(ids)),
        "mae": sum(errors) / len(errors),
        "acceptable_rate": _safe_rate(sum(e <= 1 for e in errors), len(errors)),
    }


def weighted_kappa(
    human: Sequence[int],
    judge: Sequence[int],
    max_score: int = 5,
    weights: str = "linear",
) -> float:
    """Cohen's weighted kappa for ordinal judge calibration.

    ``weights`` selects the disagreement penalty:

    * ``"linear"``    — the penalty grows with the size of the gap (default).
    * ``"quadratic"`` — the penalty grows with the SQUARE of the gap, so it is
      far more forgiving of adjacent disagreements and reports a noticeably
      higher number on the same data.

    Neither is more correct in the abstract, but they are not interchangeable,
    and a kappa quoted without its weighting scheme cannot be read. This is the
    only kappa implementation in the codebase for exactly that reason: two
    scripts must not report two different numbers under one name.

    The category set is fixed at ``1..max_score`` rather than inferred from the
    ratings observed. Weighted kappa is invariant to a *uniform* rescaling of
    the labels, so inferring the set is harmless while the observed values are
    evenly spaced — which is why the bug it causes survives casual testing. It
    bites when a value in the MIDDLE of the range goes unused: raters who avoid
    "3" leave {1, 2, 4, 5}, inferring re-indexes that to {1, 2, 3, 4}, and the
    2-to-4 disagreement is scored as one step instead of two. Measured on a
    {1, 2, 5} rating set, the inferred-category version reports linear kappa
    0.205 where the true spacing gives 0.128.
    """
    if len(human) != len(judge) or not human:
        raise ValueError("human and judge scores must be non-empty and equal length")
    if any(not 1 <= int(x) <= max_score for x in [*human, *judge]):
        raise ValueError(f"scores must be in 1..{max_score}")
    if weights not in ("linear", "quadratic"):
        raise ValueError("weights must be 'linear' or 'quadratic'")

    n = len(human)
    denom = max_score - 1
    power = 1 if weights == "linear" else 2

    def agreement(a: int, b: int) -> float:
        if not denom:
            return 1.0
        return 1.0 - (abs(a - b) / denom) ** power

    h_counts = Counter(int(x) for x in human)
    j_counts = Counter(int(x) for x in judge)
    observed = sum(agreement(int(h), int(j)) for h, j in zip(human, judge))
    expected = sum(
        agreement(h, j) * (h_counts[h] / n) * (j_counts[j] / n)
        for h in range(1, max_score + 1)
        for j in range(1, max_score + 1)
    )
    if isclose(1.0 - expected, 0.0):
        # Chance agreement is already total (one side used a single value for
        # everything). Kappa is 0/0 here: undefined, not perfect. Returning 1.0
        # would turn "this sample carries no information" into a headline score.
        raise ValueError(
            "kappa is undefined: expected agreement is 1.0, which means the "
            "ratings have no variance to correct for. Collect ratings that "
            "actually span the scale."
        )
    return (observed / n - expected) / (1.0 - expected)


def calibrate_judge(human: Sequence[int], judge: Sequence[int]) -> dict[str, float | int]:
    """Return calibration metrics suitable for a release report."""
    if len(human) != len(judge) or not human:
        raise ValueError("human and judge scores must be non-empty and equal length")
    errors = [abs(int(h) - int(j)) for h, j in zip(human, judge)]
    return {
        "n": len(human),
        "exact_agreement": _safe_rate(sum(h == j for h, j in zip(human, judge)), len(human)),
        "within_one_rate": _safe_rate(sum(e <= 1 for e in errors), len(errors)),
        "mae": sum(errors) / len(errors),
        "weighted_kappa": weighted_kappa(human, judge),
    }
