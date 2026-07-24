# Cross-Validation with promptfoo

This folder reproduces the main experiment of this repository — the custom
prompt-evaluation pipeline (`../src`, `../scripts`, ~13k LOC) — using the
industry-standard evaluation tool [**promptfoo**](https://www.promptfoo.dev/),
and checks whether two fully independent methods reach the same conclusions.

**They do.** That is the point: it validates the custom pipeline's reliability
by having a completely different toolchain arrive at the same answers.

## Why do this

The custom pipeline answers a real question for an LLM naming product: *when
feeding a client brief to the model, which fields and which model give the most
on-target output at the lowest cost?* Re-running the same question through a
standard tool is an independent check — if a mature, widely-used evaluator
agrees, the custom conclusions are trustworthy; if it disagrees, something in
the custom code needs a second look.

## What promptfoo does vs. what it doesn't

| Layer | Tool |
|---|---|
| Batch calls, multi-model, cosine scoring, cost tracking, results table | **promptfoo** (a few dozen lines of YAML) |
| Noise floor, paired sign test, "tie → choose cheapest" decision rule | **custom** (`summarize.py`, `phase4_summary.py`) — promptfoo does not do this |

promptfoo gets you to *scores*. It does not tell you whether a 0.014 gap is real
or noise — that judgement is the statistics layer, and it stays custom. Same
`NOISE_FLOOR_COSINE = 0.036` and paired test as the main pipeline (`../src/config.py`).

## Design

- **17 recipes** per task: full brief · 5 single fields · 10 field pairs · no-context baseline
- **8 extraction tasks** (concept, position, emotion, function, benefit, category, feature, context)
- **6 models** across 3 tiers (GPT-5-mini/Haiku → GPT-5/Sonnet → GPT-5.5/Opus)
- **Staged** like the original: Stage A (cheap models, all recipes) → statistics
  → Phase 4 (winners × all 6 models, premium ladder)

## Results — custom pipeline vs. promptfoo

| Dimension | Custom (13k LOC) | promptfoo (~350 lines) | Match |
|---|---|---|---|
| Recipe screening | 8/8 tasks tied → choose cheapest | 8/8 tasks tied → choose cheapest | ✅ |
| Recommended recipe | 8 tasks | 5/8 identical, other 3 within noise | ✅ |
| Premium models worth it? | Basically no (1/8 weak signal) | Basically no (1/8) | ✅ |
| **Core conclusion** | **choose by cost** | **choose by cost** | ✅ |

The 3 non-identical recommendations are **not** disagreements: on those tasks
nearly all 17 recipes are statistically tied (tie-pool of 8–9), so which recipe
is nominally first is noise — exactly what the noise-floor method is designed to
catch. An extra finding: on these terse "one sentence" tasks, premium models
(Opus / GPT-5.5) were often *worse*, being more verbose and drifting from the
concise ground truth.

## Files

```
gen_tests.py       briefs.yml → tests.yaml (promptfoo test cases)
gen_configs.py     generate 8 task configs (17 recipes × 2 cheap models)
gen_phase4.py      generate Phase-4 configs (winner recipe × 6 models × 3 briefs)
concept.yaml …     8 Stage-A configs
phase4-*.yaml      8 Phase-4 configs
summarize.py       Stage-A stats + comparison to the custom pipeline's conclusions
phase4_summary.py  premium-vs-cheap analysis
```

## Run

```bash
python gen_tests.py                                    # briefs.yml → tests.yaml
npx promptfoo@latest eval -c concept.yaml --env-file ../.env --output results/concept.json
python summarize.py                                    # stats + cross-check
```

## Data privacy

`tests.yaml`, `tests3.yaml` and `results/` are **gitignored** — they are
generated from `briefs.yml`, which contains confidential client briefs. Only the
config templates (placeholders only, no client data) and the analysis scripts
are committed. Regenerate the test data locally with `gen_tests.py`.
