# Prompt Evaluation Framework

Reproducible paired evaluation of prompt recipes and LLMs across 23 client briefs. It reports task scores, rerun noise, paired statistical comparisons and an audit of the source data. The brief count—not the number of API calls—is the independent sample size.

## Run an experiment

```bash
python -m scripts.run_experiment --stage phase0
```

Continue with `stage_a`, `stage_b` and `phase4` as appropriate. Stages can make paid API calls; review budgets and provider settings in `src/config.py` first. The command prints each stage's output and audit result. See `METHODS.md` for metric definitions and limitations.

## Service

Install `requirements-service.txt`, configure principals using `.env.example`, then start the API:

```bash
pip install -r requirements-service.txt
uvicorn service.api:app --reload
```

## Tests

```bash
python -m pytest --ignore=tests/test_llm_client_live.py -q
```

Live-provider tests need API keys. Browser tests are under `e2e/`; see `.github/workflows/e2e.yml` for the CI command.
