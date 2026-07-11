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
api and worker containers. To run the async stack without Docker:

```powershell
$env:DATABASE_URL="postgresql+psycopg://evals:evals@localhost:5432/evals"
$env:REDIS_URL="redis://localhost:6379/0"
uvicorn service.api:app                   # terminal 1
python -m service.worker                  # terminal 2
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | liveness + queue mode (redis/inline) |
| GET  | `/stages` | available stages + default budgets |
| POST | `/runs` | submit a run (`{stage, budget_usd?, max_calls?, concurrency?, note?}`) |
| GET  | `/runs` | list runs (newest first; `?status=` `?limit=` `?offset=`) |
| GET  | `/runs/{id}` | run detail (status, cost, counts) |
| GET  | `/runs/{id}/results` | per-call results (paginated) |
| GET  | `/runs/{id}/metrics` | ok-rate, cost/calls by model, tokens, latency |
| POST | `/runs/{id}/cancel` | best-effort cancel of a *queued* run |

Runs are resume-safe: re-submitting the same work skips calls already completed
OK for that run (same `(brief_id, task, config_id, model_key, run_index)` key).

## Eval CI gate

`service/ci_gate.py` blocks a release when quality drops. It reduces
`outputs/scored.csv` to one headline metric per task (the winning config's mean:
cosine for sentence tasks, F1 for the keyword task) and compares to a committed
baseline (`service/eval_baseline.json`).

```powershell
python -m scripts.analyze --score            # produce scored.csv
python -m service.ci_gate --update-baseline  # snapshot current as baseline (commit it)
python -m service.ci_gate                     # exit 1 if any task regresses > tolerance
```

`.github/workflows/eval-gate.yml` runs the gate in CI; it self-skips (green)
when no `scored.csv` is present, so wire it into the release pipeline where
scoring actually runs. `outputs/` is gitignored, so the baseline JSON is the
only committed artifact.

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
```
