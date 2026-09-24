"""Tests for judge calibration — the kappa itself, and the harness around it."""
from __future__ import annotations

import json

import pytest

from src.quality_evaluators import weighted_kappa


# --------------------------------------------------------------- the statistic

def test_linear_and_quadratic_disagree_materially():
    """The reason a kappa must always be quoted with its weighting scheme.

    Same ratings, two schemes, and the answer crosses the 0.7 "trust the judge"
    line depending on which one you picked. Before this was unified there were
    two implementations in the repo — linear in src/, quadratic in scripts/ —
    both printing "Cohen's weighted kappa".
    """
    human = [5, 4, 4, 3, 2, 1, 5, 3]
    judge = [5, 4, 3, 3, 3, 1, 4, 2]
    linear = weighted_kappa(human, judge, weights="linear")
    quadratic = weighted_kappa(human, judge, weights="quadratic")
    assert linear < 0.70 < quadratic
    assert quadratic - linear > 0.15


def test_unevenly_spaced_ratings_keep_their_true_distances():
    """A rating value nobody used must not shrink the gap between the rest.

    Weighted kappa is invariant to a uniform rescaling of the labels, so
    inferring the category set from the observed data looks harmless — and is,
    while the observed values are evenly spaced. It breaks when a value in the
    MIDDLE goes unused. Here raters used only {1, 2, 5}; inferring re-indexes
    that to {1, 2, 3}, turning the 2-to-5 disagreement into a single step.
    """
    human = [1, 2, 5, 1, 2, 5, 1, 2, 5, 1]
    judge = [1, 2, 5, 2, 5, 1, 1, 5, 2, 2]
    # The same rank pattern, re-indexed onto contiguous labels — what the old
    # observed-category implementation effectively computed.
    reindexed_h = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    reindexed_j = [1, 2, 3, 2, 3, 1, 1, 3, 2, 2]

    true_spacing = weighted_kappa(human, judge, weights="linear")
    inferred = weighted_kappa(reindexed_h, reindexed_j, weights="linear")
    assert true_spacing == pytest.approx(0.128, abs=0.005)
    assert inferred == pytest.approx(0.205, abs=0.005)
    # Inferring the categories flatters the judge by 0.08 on this sample.
    assert inferred > true_spacing


def test_no_variance_is_undefined_not_perfect():
    """Everyone rating 4 is an absence of information, not perfect agreement."""
    with pytest.raises(ValueError, match="undefined"):
        weighted_kappa([4, 4, 4, 4], [4, 4, 4, 4])


def test_rejects_out_of_scale_and_ragged_input():
    with pytest.raises(ValueError):
        weighted_kappa([1, 2, 6], [1, 2, 3])
    with pytest.raises(ValueError):
        weighted_kappa([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        weighted_kappa([1, 2], [1, 2], weights="cubic")


# ------------------------------------------------------------------ the harness

@pytest.fixture()
def kappa_env(tmp_path, monkeypatch):
    """Point the harness at a temp workbook + temp ratings file."""
    openpyxl = pytest.importorskip("openpyxl")

    xlsx = tmp_path / "workbook.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Human Review"
    ws.cell(1, 1, "Human Review")
    headers = ["Task", "Recipe", "Model", "Ground Truth", "AI Output",
               "Auto cosine", "Sonnet 1-5", "Human 1-5"]
    for c, h in enumerate(headers, start=1):
        ws.cell(4, c, h)
    # 32 samples: judge one point generous on a third of them.
    for i in range(32):
        human_truth = [5, 4, 3, 2, 1][i % 5]
        judge = min(5, human_truth + (1 if i % 3 == 0 else 0))
        row = 5 + i
        ws.cell(row, 1, f"Task{i % 4}")
        ws.cell(row, 2, f"recipe{i % 3}")
        ws.cell(row, 3, "haiku" if i % 2 else "gpt5mini")
        ws.cell(row, 4, f"ground truth {i}")
        ws.cell(row, 5, f"ai output {i}")
        ws.cell(row, 6, 0.5)
        ws.cell(row, 7, judge)
        ws.cell(row, 8, "Pending")
    wb.save(xlsx)

    from scripts import compute_kappa, rate_samples

    ratings = tmp_path / "human_ratings.jsonl"
    for mod in (rate_samples, compute_kappa):
        monkeypatch.setattr(mod, "XLSX_PATH", xlsx, raising=False)
        monkeypatch.setattr(mod, "RATINGS_PATH", ratings, raising=False)
    monkeypatch.setattr(compute_kappa, "KAPPA_PATH", tmp_path / "kappa.json")
    return {"xlsx": xlsx, "ratings": ratings, "kappa": tmp_path / "kappa.json",
            "rate_samples": rate_samples, "compute_kappa": compute_kappa}


def test_harness_reads_samples_without_leaking_the_judge_score(kappa_env):
    """The blinding is structural: the judge's score never enters the sample."""
    samples = kappa_env["rate_samples"].load_samples()
    assert len(samples) == 32
    fields = set(samples[0])
    assert "ground_truth" in fields and "ai_output" in fields
    assert "judge" not in fields and "cosine" not in fields
    assert not any("Sonnet" in str(v) for v in samples[0].values())


def _write_ratings(env, n, *, rater="tester", pass_no=1, offset=0):
    """Rate the first n samples, recovering the 'true' value the fixture encoded."""
    samples = env["rate_samples"].load_samples()
    lines = []
    for i, s in enumerate(samples[:n]):
        # The fixture built samples in workbook order; recover the human value
        # from the sample's own index in that order.
        idx = int(s["ai_output"].rsplit(" ", 1)[1])
        truth = [5, 4, 3, 2, 1][idx % 5]
        rating = max(1, min(5, truth + offset))
        lines.append(json.dumps({
            "sample_id": s["sample_id"], "task": s["task"], "recipe": s["recipe"],
            "model": s["model"], "rater": rater, "pass": pass_no,
            "rating": rating, "note": "", "rated_at": "2026-08-22T00:00:00+00:00",
        }))
    env["ratings"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_kappa_refuses_below_thirty_pairs(kappa_env, capsys):
    _write_ratings(kappa_env, 12)
    assert kappa_env["compute_kappa"].main([]) == 1
    out = capsys.readouterr().out
    assert "Below the 30-pair floor" in out
    assert "kappa (linear)" not in out          # no headline number leaked
    assert not kappa_env["kappa"].exists()      # and nothing persisted


def test_kappa_reports_with_ci_and_error_direction(kappa_env, capsys):
    _write_ratings(kappa_env, 32)
    assert kappa_env["compute_kappa"].main([]) == 0
    out = capsys.readouterr().out
    assert "weighted kappa (linear)" in out
    assert "95% bootstrap" in out
    assert "judge more generous" in out

    payload = json.loads(kappa_env["kappa"].read_text(encoding="utf-8"))
    assert payload["n_pairs"] == 32
    assert payload["decision_basis"] == "ci_lower_bound"
    assert payload["ci95_linear"][0] <= payload["kappa_linear"] <= payload["ci95_linear"][1]
    # The fixture made the judge generous on a third of samples and never harsh.
    assert payload["error_profile"]["judge_harsh"] == 0.0
    assert payload["error_profile"]["mean_signed_error"] > 0


def test_decision_uses_the_ci_lower_bound_not_the_point_estimate(kappa_env):
    """A point estimate above the bar with an interval straddling it is not a pass."""
    _write_ratings(kappa_env, 32)
    kappa_env["compute_kappa"].main([])
    payload = json.loads(kappa_env["kappa"].read_text(encoding="utf-8"))
    lower = payload["ci95_linear"][0]
    expected = ("HIGH" if lower >= 0.70 else "MEDIUM" if lower >= 0.40 else "LOW")
    assert payload["decision"] == expected


def test_intra_rater_pass_two_is_reported_as_the_ceiling(kappa_env, capsys):
    _write_ratings(kappa_env, 32)
    existing = kappa_env["ratings"].read_text(encoding="utf-8")
    samples = kappa_env["rate_samples"].load_samples()
    extra = []
    for i, s in enumerate(samples[:10]):
        idx = int(s["ai_output"].rsplit(" ", 1)[1])
        truth = [5, 4, 3, 2, 1][idx % 5]
        # Disagree with yourself on two of ten — a realistic re-rate.
        rating = max(1, min(5, truth + (1 if i in (2, 7) else 0)))
        extra.append(json.dumps({
            "sample_id": s["sample_id"], "task": s["task"], "recipe": s["recipe"],
            "model": s["model"], "rater": "tester", "pass": 2,
            "rating": rating, "note": "", "rated_at": "2026-08-22T01:00:00+00:00",
        }))
    kappa_env["ratings"].write_text(existing + "\n".join(extra) + "\n", encoding="utf-8")

    kappa_env["compute_kappa"].main([])
    out = capsys.readouterr().out
    assert "intra-rater kappa" in out
    payload = json.loads(kappa_env["kappa"].read_text(encoding="utf-8"))
    assert payload["intra_rater_kappa_linear"] is not None
