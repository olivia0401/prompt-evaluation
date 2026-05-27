# Prompt Evaluation Framework

End-to-end automated evaluation pipeline for LLM prompt experiments. **One command per stage, two Drive artefacts auto-produced, every claim audited against the raw data.**

The current run (23 briefs × 8 sentence tasks × 142 recipes × 6 models + a keyword-compression sub-experiment) cost **$3.05** total and produces a four-tab business-facing workbook that any stakeholder can scan in under five minutes.

## Pipeline

```
You run one command
       │
       ▼
run_experiment.py --stage <name>
       │
       │  ① Read outputs/results.jsonl → skip completed calls (resume-safe)
       │  ② Build the todo list
       │  ③ Call LLM APIs
       │     ├── Per-model concurrency semaphore
       │     ├── SDK max_retries=8 + exponential backoff
       │     └── 3-tier budget gate (global / per-model / call-cap)
       │  ④ Each result appended to results.jsonl
       │
       ▼
5-step auto-build chain  (fail-soft, each step independent)
       │
       ├── analyze --score        → scored.csv
       ├── analyze --summarize    → summary CSV
       ├── build_engineering_note → Google Doc URL
       ├── build_xlsx             → Google Sheet URL (in-place replace)
       └── audit_data             → 31 checks across 9 groups
       │
       ▼
STOP gate prints 2 URLs + audit pass/fail + per-step retry commands
```

## Run

```powershell
python -m scripts.run_experiment --stage phase0     # pilot
python -m scripts.run_experiment --stage stage_a    # broad screen
python -m scripts.run_experiment --stage stage_b    # stability reruns
python -m scripts.run_experiment --stage phase4     # premium model ladder
```

Each stage ends with a hard STOP gate — Drive URLs printed, send for review, then proceed.

## Deliverable workbook (4 tabs)

Built by `python -m scripts.build_xlsx` (English) or `python -m scripts.build_xlsx --lang zh` (Chinese). Both versions are uploaded to separate Google Sheets so they never overwrite each other.

| Tab | Layer | Contents |
|---|---|---|
| **1. Executive Summary** | Business decision layer | Business question · Decision (4 bullets) · Key findings · Recommended recipes (8 rows) · Keyword compression summary · Production recommendation · Score explanation. **A reader who only opens Tab 1 still knows what to ship.** |
| **2. Final Production Recommendations** | Deployment layer | Score guide (3 bands) · Per-task table (Task / Recipe / Model / Avg score / Vs Full Brief / Decision / Why) · Model policy (6 models with rec + note) · Operational takeaway. **The deployment-ready table.** |
| **3. Analysis** | Methodology evidence layer | Maps 1:1 to the experiment workflow diagram: Part A sentence-task scoring (5 sub-tables) · Part B keyword scoring · Part C stability check · Part D reliability + cost. Every section closes with an "Operational implication" line. |
| **4. Human Review** | Trust / validation layer | Section A: 30 balanced samples (Task / Recipe / Model / Ground Truth / AI Output / Auto cosine / Sonnet 1-5 / Human 1-5) · Section B: 5-row agreement summary with the live κ value computed against the AI judge. |

Sheet names: `Executive Summary` / `Final Recommendations` / `Analysis` / `Human Review` (`执行摘要` / `生产推荐` / `数据分析` / `人工评审`).

A companion engineering-note Doc is regenerated alongside the workbook on every stage completion.

## Engineering highlights

| Feature | Why it matters |
|---|---|
| **Resume key** = `(brief_id, task, config_id, model_key, run_id)` | Crash-safe — rerun skips completed calls, no double-billing |
| **Per-model semaphore + SDK retry=8** | Solves provider rate-limit storms (one cheap model dropped 77% of calls at concurrency=15 → fixed; Haiku still hits 46% rate-limit but the retry layer catches every one) |
| **Embedding cache** (sha256-keyed) | Re-scoring is free; batch prewarm cut 30-min serial scoring to 1–2 min |
| **Catalog-driven error log** | Verification + runtime errors auto-grouped by known issue; uncategorised ones surface for follow-up |
| **Fail-soft auto-build chain** | One step crashing doesn't kill the others; copy-paste retry commands printed |
| **Automated data audit** | `scripts/audit_data.py` runs 31 checks across 9 groups — schema, completeness, reliability, statistical adequacy, scoring integrity, workbook ↔ data consistency, security, output quality, stability. See *Audit checks* below. |
| **Bilingual workbook** | Single source, `--lang zh` re-renders every cell in Chinese; CJK-aware row-height math so wrapped Chinese never truncates |
| **Editorial recipe overrides** | When two recipes tie within the 0.036 noise floor, an editorial pick locks in one consistent answer across every tab (e.g. Benefit = `audience + differentiators`) |
| **Live κ sanity check** | Sonnet ↔ cosine-bin Cohen's weighted κ computed at build time (currently 0.62, Moderate); separately tracked from Human ↔ Sonnet κ which awaits human ratings |
| **Status taxonomy** | Every call lands as `ok` / `parse_fail` / `rate_limited` / `truncated` / `budget_exceeded` etc. — zero silent failures |
| **146 offline tests** | Resume logic, mocked subprocess chain, semaphore wiring, budget gates, catalog matching, shortlist selection, audit failure handling |

## Audit checks — what `scripts/audit_data.py` validates

Run standalone:

```powershell
python -m scripts.audit_data            # human-readable PASS / FAIL / WARN report
python -m scripts.audit_data --json     # machine-readable
```

Exit code `0` = all PASS (warnings allowed), `1` = at least one FAIL.

The 31 checks are organised into nine groups; each check catches a specific class of bug we have actually hit. The current run scores **27 PASS · 4 WARN · 0 FAIL**.

### Group A — Schema / prerequisites
Confirms the upstream files we depend on actually exist and have the right shape before any downstream logic runs.

- **A1.** `outputs/results.jsonl` exists and is non-empty.
- **A2.** `briefs.yml` has the expected 23 briefs.
- **A3.** `outputs/scored.csv` exists (warn if not — workbook cannot be trusted without it).

### Group B — Data completeness
Catches "the experiment didn't actually finish" silently. Forces us to compare the realised matrix against the planned 23 × 142 × 2 = 6,532-cell Stage A grid.

- **B1.** All 23 briefs from `briefs.yml` appear in `results.jsonl` at least once.
- **B2.** All 9 tasks present (8 sentence tasks + the keyword task).
- **B3.** Per-cell (task × config × model) sample-size distribution and outliers — flags cells where a model only got partial coverage.
- **B4.** Stage A coverage % vs the planned 23 × 142 × 2 = 6,532 cells.
- **B5.** Stage B `run_id` coverage (if `scored.csv` contains any `run_id ∈ {2, 3}`).

### Group C — Per-call reliability
Catches provider failures that the retry layer didn't recover from, and silent data corruption.

- **C1.** Per-model OK-rate (eventual-success rate per unique resume key) ≥ 95%.
- **C2.** Reports leftover `rate_limited` / `api_error` / `timeout` rows so the operator knows whether to rerun.
- **C3.** No empty `prediction` on `status=ok` rows.
- **C4.** Every resume key eventually reached `ok` (or reports the orphans).

### Group D — Statistical adequacy
Confirms the conclusions we ship are statistically defensible.

- **D1.** Sentence cells named on Tab 1 have `n_briefs ≥ 2` (else the 95% CI is undefined).
- **D2.** Length-compliance ≥ 90% (otherwise the model's been ignoring the word-count rule).
- **D3.** "Winner vs second-best" cosine gap ≥ noise floor (0.036) — flags tasks where the top recipe is statistically a tie, so the workbook can say "Equivalent" instead of overclaiming.

### Group E — Scoring integrity
Catches `scored.csv` drift from `results.jsonl`.

- **E1.** Every `status=ok` row in `results.jsonl` has a matching row in `scored.csv`.
- **E2.** `cosine` / `rouge_l` / `f1` values are in `[0, 1]` (NaN allowed for skipped rows).
- **E3.** No "missing ground truth" or "brief not found" parse errors.

### Group F — Workbook ↔ data consistency
Cross-checks that the workbook isn't claiming more data than the experiment actually produced.

- **F1.** The workbook does not claim "23 briefs" if fewer were actually run.
- **F2.** The pilot-status banner in the workbook matches the actual call count and cost.

### Group G — Security
A pre-publication scan so we never accidentally ship secrets or PII.

- **G1.** Secret files (`.env`, `credentials.json`, `token.json`) are not tracked in git.
- **G2.** No hardcoded API keys in `src/` or `scripts/`.
- **G3.** No API-key leaks in `raw_response` (in `results.jsonl`).
- **G4.** No PII (email / phone / credit card) in `raw_response`.
- **G5.** `.gitignore` covers `.env`, `credentials.json`, `outputs/`, the embedding cache.
- **G6.** No `sk-…` / `sk-ant-…` tokens leaked into `scored.csv` predictions.

### Group H — Output quality
Catches qualitative degradation that cosine alone could miss — refusals, copy-paste, empty answers.

- **H1.** Refusal rate ≤ 5% — patterns like "I cannot", "As an AI", "I don't have access to…".
- **H2.** Echo / verbatim-copy rate ≤ 2% — prediction equals the input brief field.
- **H3.** Keyword task: ≥ 90% of OK rows return exactly 10 keywords.
- **H4.** Cosine spike at < 0.1 — refusal / off-topic floor ≤ 5%.

### Group I — Stability & determinism
Validates the headline claim that single-run scores are trustworthy.

- **I1.** Stage B cross-run cosine standard deviation: median < 0.05.
- **I2.** `gpt5mini` reruns more stable than `haiku` (the OpenAI `seed=42` parameter pays off; Anthropic has no seed equivalent).

## How the workbook maps to the methodology

Tab 3 (Analysis) is a 1-to-1 reflection of the experiment's scoring flow:

```
AI generates output (task × recipe × model × brief)
    │
    ├──→ Part A: Sentence-task scoring (cosine + ROUGE-L + length compliance)
    │       A1 Model summary
    │       A2 Premium vs cheap on same recipe
    │       A3 Brief vs No Brief
    │       A4 Field-level signal
    │       A5 Per-task production picks
    │
    ├──→ Part B: Keyword-task scoring (Precision / Recall / F1 with Porter stemming)
    │       B1 A / B / C / D compression results
    │
    ├──→ Part C: Stage B stability check (top-2 recipes × 3 reruns → std dev)
    │       C1 Cross-rerun cosine std dev
    │
    └──→ Part D: Reliability + operational cost
            D1 Per-model OK rate
            D2 Per-model spend
```

Tab 4 (Human Review) closes the methodology loop — the AI judge itself needs validation. 30 stratified samples × Sonnet 1-5 absolute ratings + Cohen's weighted κ against the eventual human ratings (currently pending). The κ-decision rule is:

```
κ ≥ 0.7   →  Trust AI judge for the remaining ~6,500 unrated calls
0.4 ≤ κ <0.7 → Collect 30 more human samples and re-evaluate
κ < 0.4   →  Reject AI judge; rewrite prompt or rely on human only
```

## Setup

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # add OPENAI_API_KEY, ANTHROPIC_API_KEY
python -m scripts.verify_models     # confirm model IDs are reachable
```

For Drive uploads: drop a Google Cloud OAuth desktop `credentials.json` in the project root; first run opens a browser for consent. Both Google **Drive API** AND **Sheets API** must be enabled in the same Cloud project — Drive does the xlsx→Sheets conversion internally and fails with a misleading 403 if Sheets API is off.

## Tests

```powershell
pytest                              # 146 offline tests
RUN_LIVE_TESTS=1 pytest             # + 1 live API smoke test (~$0.01)
```

## Repo layout

```
src/                 # Library code: LLM client, scoring, Drive upload, prompt builder
scripts/             # Entry points: run_experiment, build_xlsx, audit_data, verify_models,
                     #               analyze, ai_judge_absolute, pairwise_judge, compute_kappa
tests/               # 146 offline tests
briefs.example.yml   # Template — copy to briefs.yml (gitignored) and fill in real briefs
prompts.txt          # Task prompts (4 keyword versions + 8 sentence-task prompts)
.env.example         # Template for API keys
```

`outputs/` (raw + scored data), `Results/` (built xlsx), `briefs.yml` (client data), and the secret files (`.env`, `credentials.json`, `token.json`) are all gitignored.
