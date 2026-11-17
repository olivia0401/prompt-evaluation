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

## Evaluation scope and evidence boundaries

This repository contains **two different evaluation datasets**. They must not be
combined or described as one Golden Dataset:

- **23 client briefs** support the brand-naming extraction experiments: task,
  prompt and input-field configurations, model comparisons, and related scoring.
  The real `briefs.yml` is excluded from Git by `.gitignore`;
  `briefs.example.yml` is an illustrative example, not a released client brief.
  The independent sample size is 23 briefs, not the number of model calls.
- **61 safety/evidence annotation items** live under `data/golden/` and are
  described in `data/golden/README.md`. They come from the separate
  `parent-check` rule-engine evaluation and exercise evidence grounding and
  citation-quality checks. They are not the 23 brand briefs or an additional
  brand-naming benchmark. The current annotations have one pass by one annotator;
  do not treat this as an independently adjudicated gold standard.

The related [`brief_product` prototype](https://github.com/olivia0401/brief_product)
reports overlapping results on the brand-brief benchmark. Repeated metrics
(including the keyword parsing correction) refer to related evaluation work,
not independent client deployments or additive improvements.

### Quality gates and what CI proves

- The development-time regression gate in `service/ci_gate.py` uses a pinned
  task configuration and measured rerun noise. No committed
  `service/eval_baseline.json` is available at this snapshot, so a successful
  unit-test or browser-test workflow does **not** mean the regression gate
  certified a production release.
- Tests verify the code and stated scenarios. The evidence-quality gate and
  the limitations of the 61-item annotation set are documented separately;
  neither is replaced by ordinary pytest or Playwright passes.
- The model-selection comparisons over many configurations do not yet apply
  multiple-comparison correction (see `METHODS.md`). Treat any “best model” or
  “best recipe” claim as conditional on the evaluated sample and setup.

**Data sharing:** Publicly committed evaluation examples and annotations should
be reviewed for authorisation and sensitive content before reuse. Do not commit
real client briefs, credentials, or private source documents to this repository.
