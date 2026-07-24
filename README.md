# Prompt Evaluation Framework

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
- **Cohen's weighted κ**: AI-judge (Sonnet) vs human ratings; until ≥30 human ratings
  are filled in, the AI judge is reference-only.

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
score drops below the committed baseline). Full details in
[service/README.md](service/README.md).

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
