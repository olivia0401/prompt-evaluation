# Prompt Evaluation Framework

**[METHODS.md](METHODS.md) maps every component here to what the rest of the
field calls it** — which parts are a RAGAS / DeepEval / promptfoo equivalent,
which parts those tools do not give you (measured noise floors, paired
significance, sample-size floors), and one known statistical gap left open
rather than hidden.

Automated pipeline for comparing LLM prompt recipes and models on a fixed set of
client briefs. One command per stage; every run is resume-safe, budget-capped,
and audited against the raw data. The current run is a **paired experiment over
23 briefs** — large API-call counts improve within-brief comparisons but do not
create thousands of independent samples.

## Pipeline

```
run_experiment.py --stage <name>     # call the LLM APIs (resume-safe, budget-gated)
   └─ analyze --score                # cosine / ROUGE-L / F1 → scored.csv
   └─ build_xlsx                      # 4-tab workbook (local .xlsx + optional Google Sheet)
   └─ audit_data                      # 32 data-integrity checks (PASS/WARN/FAIL)
```

## Run

```powershell
python -m scripts.run_experiment --stage phase0     # pilot (3 briefs)
python -m scripts.run_experiment --stage stage_a    # broad screen (23 briefs × ~142 recipes × 2 cheap models)
python -m scripts.run_experiment --stage stage_b    # stability reruns (top-2 recipes/task)
python -m scripts.run_experiment --stage phase4     # premium-model ladder (3 curated briefs)
```

Each stage ends at a STOP gate that prints the deliverable URLs and the audit result.

## What it tests

- **23 client briefs** — the true independent sample size.
- **9 tasks** — 8 one-sentence extraction tasks + 1 keyword task (extract 10 terms).
- **~142 recipes** — which brief fields to feed the prompt (full brief, no brief,
  single fields, field pairs) + 4 keyword-prompt versions (A/B/C/D).
- **6 models across 3 cost tiers** — cheap / medium / premium, spanning OpenAI and
  Anthropic. Exact model IDs are pinned in `src/config.py` and confirmed with
  `scripts/verify_models.py` before any cost claim is published.

## Scoring & statistics

- **Sentence tasks**: embedding cosine vs ground truth (primary) + ROUGE-L + length compliance.
  Embeddings use OpenAI `text-embedding-3-large` when `OPENAI_API_KEY` is set; with no key
  the client auto-falls back to local `sentence-transformers/all-mpnet-base-v2` so scoring
  still runs offline (scores stay comparable within a run, not across backends).
- **Keyword task**: precision / recall / F1 on Porter-stemmed term sets.
- **Noise floor** (`NOISE_FLOOR_COSINE`): 2σ of within-cell rerun cosine, re-measured
  across several briefs via `measure_noise.py`. Differences below it are ties.
- **Paired Wilcoxon signed-rank**: the real "is A better than B" test across the same
  briefs — a higher mean alone is never reported as a win.
- **Leave-one-brief-out**: recomputes each task winner after dropping one brief at a time.
- **Cohen's weighted κ**: AI-judge (Sonnet) vs human ratings, collected blind via
  `scripts/rate_samples.py` and reported under both linear and quadratic
  weighting with a bootstrap CI. Below 30 pairs no headline κ is printed at all.

## Deliverable workbook (4 tabs)

`python -m scripts.build_xlsx` (`--lang zh` for Chinese). Bold values in Tab 1 are
computed live from the data; normal-weight text is fixed editorial wording.

| Tab | Purpose |
|---|---|
| 1. Executive Summary | Decision, key findings, recommended recipes (computed), keyword compression |
| 2. Final Recommendations | Per-task winner table with median Δ, win/tie/loss, paired p, leave-one-out |
| 3. Analysis | Field-contribution heatmap, keyword compression, stability, reliability/cost |
| 4. Human Review | 30 stratified samples + Sonnet/Human ratings → live Human↔Sonnet κ |

## Interpretation guardrails

- Sample size is **23 paired briefs** — treat findings as directional, not universal.
- On the current data most task winners are **statistical ties** → pick by cost.
- **Cost conclusions are provisional** until model IDs and prices are verified
  (`verify_models`, then set `PRICES_VERIFIED = True` in `src/config.py`).
- Phase 4 (premium) runs on **3 briefs only** — directional until run across all 23.

## Audit

```powershell
python -m scripts.audit_data          # PASS / WARN / FAIL report (exit 0 = no FAIL)
```

32 checks across schema, completeness, reliability, statistical adequacy (incl. the
paired-significance check), scoring integrity, workbook↔data consistency, security,
output quality, and stability.

## Setup

```powershell
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                  # add OPENAI_API_KEY, ANTHROPIC_API_KEY
python -m scripts.verify_models       # confirm model IDs/prices before publishing cost claims
```

Google Sheets upload (optional): set `GOOGLE_SHEETS_ID` in `.env`, enable **both** the
Drive API and Sheets API in one Google Cloud project, and drop an OAuth desktop
`credentials.json` in the project root (first run opens a browser for consent).

## Service (API + queue + DB + dashboard)

The batch engine above also runs as a long-running service — submit eval runs
over HTTP, execute them on a worker, persist runs + results, and watch a
dashboard. Zero-infra by default (SQLite + inline jobs); scales to Postgres +
Redis + an RQ worker via `docker compose up`.

```powershell
pip install -r requirements-service.txt
uvicorn service.api:app --reload      # http://localhost:8000/docs
streamlit run service/dashboard.py    # http://localhost:8501
# or the full async stack:
docker compose up --build
```

`service/ci_gate.py` is the eval regression gate (blocks a release when a task's
score drops below the committed baseline). Two things it deliberately does:

- **Scores the recipe the baseline names**, not the run's current best. Taking
  `max` over ~142 recipes is an order statistic — biased upward, noisier than
  any single recipe, and liable to silently describe a different recipe each
  run, so the "regression" it reports can be a recipe swap rather than a quality
  drop. If the baselined recipe is missing from a run, that is a FAIL (the run
  is unverifiable), not a pass on whatever else scored well.
- **Never gates tighter than the measured noise.** The tolerance defaults to
  `cfg.NOISE_FLOOR_COSINE` (the empirical 2σ rerun band), so identical quality
  cannot trip the gate on sampling noise alone. A gate that cries wolf inside
  its own noise band is one everybody learns to ignore.

Full details in [service/README.md](service/README.md).

### Evidence-grounded quality gate

The platform also includes an offline, annotation-driven quality contract for
investigative or evidence-grounded outputs:

- `src/quality_evaluators.py` measures citation completeness, groundedness,
  unsupported-claim rate, source quality, entity-resolution B-cubed F1, and
  ordinal judge/human calibration.
- `src/golden_manifest.py` fingerprints golden files with SHA-256 so a report
  cannot silently change its evaluation set.
- `service/quality_gate.py` blocks a release unless the report carries its
  dataset version, evaluator version and golden-manifest hash, and meets the
  configured quality floors.

These metrics require explicit gold annotations; lexical similarity alone is
not treated as proof that a claim is true.

The golden set lives in **`data/golden/`** — the one thing under `data/` that is
version-controlled, because unlike `outputs/` it is not regenerable. It holds
**61 real items** exported from a shipped product (`parent-check`'s rule engine,
which reports the exact signals behind each verdict), stratified across
scam / benign / health outcomes:

```bash
cd ../parent-check && python export_eval_queue.py --output "../prompt test/data/golden/queue.jsonl"
python -m scripts.annotate_golden --annotator <you>     # ~60-90 min
python -m scripts.annotate_golden --agreement           # inter-annotator kappa
python -m scripts.annotate_golden --build               # -> annotations.json
```

**16 of the 61 verdicts cite no signal at all** — measurable before annotation
begins, and a 26% ceiling on citation completeness. Those are the informative
items: a correct call the engine cannot justify and a lucky guess are
indistinguishable from outside, and only annotation separates them.

Three of the six gate metrics genuinely cannot be computed on this corpus. They
are **declared** with a reason under `not_applicable` rather than omitted — the
gate fails on an undeclared missing metric, and prints declared exclusions even
on a PASS, so a green tick can never quietly cover less than the reader assumes.
[`data/golden/README.md`](data/golden/README.md) has the full rationale and the
honest limits of a set this size.

Three properties the report and gate enforce:

- **Rates are pooled, not averaged over examples.** A denominator-weighted
  (micro) rate is what gets gated; the macro mean is reported alongside so a
  large gap between them is visible rather than hidden. On the seed set the
  macro entity-resolution F1 (0.917) passes the 0.90 floor and the pooled one
  (0.884) does not — averaging incomparable fractions had been buying a pass.
- **Judge kappa is computed once over pooled ratings.** Cohen's kappa is not an
  average-able quantity: on a single agreeing pair it returns 1.0, because
  expected agreement is also 1.0, so a mean of per-example kappas drifts upward
  with every short example.
- **Sample-size floors gate before threshold floors.** Metrics computed on 9
  claims cannot clear a threshold meaningfully, so "not enough data to judge"
  is reported as its own failure and never as a pass.

Example usage:

```powershell
python -m scripts.build_golden_manifest --root data/golden --output data/golden/manifest.json --version 2026.08.22
python -m scripts.build_golden_manifest --root data/golden --verify data/golden/manifest.json
python -m scripts.build_quality_report --input data/golden/annotations.json `
  --manifest data/golden/manifest.json --output outputs/quality_report.json
python -m service.quality_gate outputs/quality_report.json
python -m scripts.aggregate_quality_reports --inputs outputs/run_1.json outputs/run_2.json `
  --output outputs/quality_stability.json
```

The quality report schema is `quality-report/v1`. It is intentionally separate
from the existing prompt-score gate: prompt quality, evidence grounding,
entity resolution and judge calibration have different denominators and must
not be collapsed into one misleading score.

### Multi-tenant service + browser suite

The service is multi-tenant: an API key carries a **tenant** and a **role**
(admin / operator / viewer), every query is filtered by tenant in the data layer
rather than by a check each route has to remember, and a cross-tenant read
returns **404 rather than 403** — a 403 confirms the row exists, which turns run
ids into an enumeration oracle.

`e2e/` drives six authenticated personas (two tenants x three roles) across
Chromium, Firefox and WebKit — 87 tests, ~10 seconds, no setup:

```bash
cd e2e && npm ci && npx playwright install chromium firefox webkit
npx playwright test
```

The suite POSTs real runs against a real server started with
`DISABLE_RUN_EXECUTION=1`: the full API surface is live and no LLM provider is
ever contacted, so it costs nothing and needs no provider keys. `retries` is 0
even in CI — the one flake it has hit was fixed at the root (cold engine launch
moved into `global-setup.ts`) rather than retried. `tests/test_tenancy.py` is
the same 24 assertions without a browser, so a broken scope fails in seconds on
every push. Details in [e2e/README.md](e2e/README.md).

### Judge calibration

```bash
python -m scripts.rate_samples --rater <you>              # blinded, shuffled, resumable
python -m scripts.rate_samples --rater <you> --pass 2 --sample 10   # intra-rater ceiling
python -m scripts.compute_kappa                           # kappa + bootstrap CI
```

The judge's score and the automatic cosine are hidden while rating: an anchored
human agreeing with the judge is not evidence that the judge is right. Kappa is
reported under **both** linear and quadratic weighting because they can differ
by 0.2 on the same ratings — enough to move a result across the 0.7 "trust the
judge" line — and the decision is taken on the **lower bound of the bootstrap
CI**, not the point estimate. Below 30 pairs it refuses to print a headline
number at all.

## Tests

```powershell
python -m pytest                      # offline tests (incl. tests/test_service.py)
```

## QE tooling (optional)

Two industry tools sit alongside the custom scoring, both off unless configured
(`pip install -r requirements-qe.txt`):

- **Langfuse** (`src/observability.py`) — set `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` and every `LLMClient.call()` logs a generation
  (prompt, output, tokens, cost, latency, status) for per-call tracing on top of
  the CSV/JSONL. No keys → silent no-op.
- **Giskard** (`scripts/scan_giskard.py`) — `python -m scripts.scan_giskard
  --task concept_relevant --model haiku` runs an LLM vulnerability scan
  (hallucination, prompt injection, robustness) and writes an HTML report to
  `results/`. Complements the "how well does it score" pipeline with "how does
  it break".

## Layout

```
src/        # library: LLM client, scoring, prompt builder, Drive upload
scripts/    # entry points: run_experiment, analyze, build_xlsx, audit_data,
            #               verify_models, measure_noise, compute_kappa
service/    # API + RQ worker + Postgres models + Streamlit dashboard + eval gate
tests/      # offline tests
prompts.txt # task prompts (8 sentence + 4 keyword versions)
briefs.yml  # client briefs (gitignored — copy from briefs.example.yml)
```

`outputs/`, `Results/`, `briefs.yml`, and secrets (`.env`, `credentials.json`,
`token.json`) are gitignored.
