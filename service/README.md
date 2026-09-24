# Evaluation Service

Wraps the batch evaluation engine (`src/` + `scripts/`) as a running service:
submit eval runs over HTTP, execute them asynchronously on a worker, persist
runs + per-call results in a database, watch them on a dashboard, and gate
releases on eval regression.

```
             POST /runs                     enqueue                  execute
  client ───────────────▶  FastAPI (api.py) ───────▶ Redis + RQ ───▶ worker.py
     ▲                          │                                       │
     │  GET /runs/{id}/...      │ read/write                            │ execute_run()
     │                          ▼                                       ▼
  Streamlit ◀──── HTTP ──── Postgres  ◀──────────── results + status ───┘
  (dashboard.py)            (runs, call_results)         │
                                                         └▶ outputs/results.jsonl
                                                            (mirror → analyze / build_xlsx / audit)
```

**Reuse, not fork.** The actual LLM calls, budgeting, todo-building and scoring
all come from the existing engine unchanged. `runner.execute_run` calls
`scripts.run_experiment.build_todo` and `src.llm_client.LLMClient`; every result
is mirrored to `outputs/results.jsonl` so the existing `analyze → build_xlsx →
audit` chain keeps working.

## Zero-infra mode (default)

No Postgres, no Redis needed:

```powershell
pip install -r requirements-service.txt
uvicorn service.api:app --reload         # http://localhost:8000/docs
```

The DB falls back to SQLite (`data/service.db`) and jobs run **inline**
(synchronously inside the request). Good for local dev, tests and demos.

```powershell
# submit + inspect a run
curl -X POST localhost:8000/runs -H "content-type: application/json" -d "{\"stage\":\"phase0\"}"
curl localhost:8000/runs
curl localhost:8000/runs/<id>/metrics
```

Dashboard:

```powershell
streamlit run service/dashboard.py       # http://localhost:8501
```

## Full stack (Postgres + Redis + async worker)

```powershell
docker compose up --build
# API       http://localhost:8000/docs
# Dashboard http://localhost:8501
```

Put `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` in `.env` — compose passes them to the
api and worker containers. Two things the compose file does not do for you:
runs need the confidential `briefs.yml`, which is kept out of the image (mount
it into api and worker, or every run is marked FAILED), and authentication is
off until you set `API_TOKEN` or `SERVICE_PRINCIPALS` (see below). Postgres and
Redis are bound to `127.0.0.1` with throwaway credentials.

To run the async stack without Docker:

```powershell
$env:DATABASE_URL="postgresql+psycopg://evals:evals@localhost:5432/evals"
$env:REDIS_URL="redis://localhost:6379/0"
uvicorn service.api:app                   # terminal 1
python -m service.worker                  # terminal 2
```

## API

Every route except `/health` requires `Authorization: Bearer <key>` once keys
are configured: `SERVICE_PRINCIPALS` (a JSON list of keys, each with a tenant
and a role: admin / operator / viewer) or the single legacy `API_TOKEN` (admin
on the `default` tenant). With neither set the API runs in **open mode**, where
every caller is an admin; `GET /whoami` reports `open_mode` so this is visible.
Reads are scoped to the caller's tenant, and another tenant's run returns 404.
The dashboard sends `DASHBOARD_API_KEY` as its bearer token.

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | liveness + queue mode (redis/inline); the only public route |
| GET  | `/whoami` | the calling principal: name, tenant, role, `open_mode` |
| GET  | `/stages` | available stages + default budgets |
| POST | `/quality-reports` | store a `quality-report/v1` with its gate decision (operator/admin) |
| GET  | `/quality-reports` | list this tenant's quality reports |
| POST | `/runs` | submit a run (`{stage, budget_usd?, max_calls?, concurrency?, note?}`; operator/admin) |
| GET  | `/runs` | list runs (newest first; `?status=` `?limit=` `?offset=`) |
| GET  | `/runs/{id}` | run detail (status, cost, counts) |
| GET  | `/runs/{id}/results` | per-call results (paginated) |
| GET  | `/runs/{id}/metrics` | ok-rate, cost/calls by model, tokens, latency |
| POST | `/runs/{id}/cancel` | best-effort cancel of a *queued* run (admin) |

Runs are resume-safe: re-submitting the same work skips calls already completed
OK for that run (same `(brief_id, task, config_id, model_key, run_index)` key).

## Eval CI gate

`service/ci_gate.py` blocks a release when quality drops. It reduces
`outputs/scored.csv` to one headline metric per task (the mean of the config
pinned in the baseline: cosine for sentence tasks, F1 for the keyword task) and
compares it to the baseline in `service/eval_baseline.json`. When that file
does not exist yet, the gate prints the current scores and exits 0.

```powershell
python -m scripts.analyze --score            # produce scored.csv
python -m service.ci_gate --update-baseline  # snapshot current as baseline (commit it)
python -m service.ci_gate                     # exit 1 if any task regresses > tolerance
python -m service.ci_gate --tolerance 0.05    # override the baseline's tolerance
```

**Current state:** no baseline is committed, and `outputs/` is gitignored, so
`.github/workflows/eval-gate.yml` skips both gate steps (green) on every pull
request in this repository. It becomes a real gate once it runs where
`scored.csv` exists (a release pipeline with the briefs) and a baseline from
`--update-baseline` is committed.

## Evidence-grounded quality gate

For systems that produce claims backed by sources, run a second gate against an
annotated `quality-report/v1` JSON report:

```powershell
python -m service.quality_gate outputs/quality_report.json --manifest data/golden/manifest.json
```

The report must include `dataset_version`, `metrics` and `provenance` with both
an evaluator version and a SHA-256 golden-manifest hash; `--manifest` also checks
that hash against the manifest file and every golden file against its recorded
hash. Sample-size floors (≥ 30 claims, ≥ 30 judge ratings unless the judge
metric is declared not applicable) are checked before the quality floors, which
default to:
groundedness ≥ 0.90, citation completeness ≥ 0.95, unsupported claims ≤ 0.05,
entity-resolution F1 ≥ 0.90, source acceptable rate ≥ 0.90 and judge weighted
κ ≥ 0.60. The thresholds are intentionally explicit and can be overridden by
calling `check_report` from a release pipeline.

The evaluator is annotation-driven: human/researcher gold labels define which
evidence supports a claim and what source quality means. This prevents a simple
string-overlap score from being presented as factual verification.

## Layout

```
service/
  settings.py     env config (DATABASE_URL, REDIS_URL, INLINE_JOBS, …)
  db.py           SQLAlchemy engine + session factory + init_db
  models.py       Run, CallResultRow
  schemas.py      Pydantic request/response
  repository.py   CRUD + operational metrics
  runner.py       execute_run — bridges a Run to build_todo + LLMClient
  tasks.py        enqueue_run (RQ) with inline fallback
  worker.py       RQ worker entrypoint
  api.py          FastAPI app
  dashboard.py    Streamlit dashboard (HTTP client of the API)
  ci_gate.py      eval regression gate
  quality_gate.py evidence-grounded quality/release gate
```
