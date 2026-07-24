# Cross-Validation with promptfoo

Reproduces this repository's main experiment — the custom prompt-evaluation
pipeline (`../src`, `../scripts`, ~13k LOC) — with the industry-standard tool
[promptfoo](https://www.promptfoo.dev/), as an independent check on whether two
different toolchains reach the same conclusions.

They do. That is the point: an independent method arriving at the same answers
validates the custom pipeline's reliability.

## What promptfoo does vs. what stays custom

| Layer | Tool |
|---|---|
| Batch calls, multi-model, cosine scoring, cost tracking | promptfoo (YAML config) |
| Noise floor, paired sign test, tie→cheapest decision rule | custom (`summarize.py`, `analyze.py`) |

promptfoo produces scores. It does not decide whether a 0.014 gap is real or
noise — that judgement is the statistics layer, and it stays custom, using the
same `NOISE_FLOOR_COSINE = 0.036` and paired test as `../src/config.py`.

## Design

- 17 recipes per task: full brief, 5 single fields, 10 field pairs, no-context baseline
- 8 extraction tasks (concept, position, emotion, function, benefit, category, feature, context)
- 6 models across 3 tiers (GPT-5-mini/Haiku, GPT-5/Sonnet, GPT-5.5/Opus)
- Staged like the original: Stage A (cheap models, all recipes) → statistics → Phase 4 (winners × 6 models)

## Results — custom pipeline vs. promptfoo

| Dimension | Custom (13k LOC) | promptfoo (~350 lines) | Match |
|---|---|---|---|
| Recipe screening | 8/8 tasks tied → cheapest | 8/8 tasks tied → cheapest | yes |
| Recommended recipe | 8 tasks | 5/8 identical, other 3 within noise | yes |
| Premium models worth it? | Basically no (1/8) | Basically no (1/8) | yes |
| Core conclusion | choose by cost | choose by cost | yes |

The 3 non-identical recommendations are not disagreements: on those tasks nearly
all recipes are statistically tied, so the nominal first place is noise — exactly
what the noise-floor rule is built to catch. Side note: on these terse one-sentence
tasks, premium models (Opus, GPT-5.5) were often *worse* — more verbose, drifting
from the concise ground truth.

## Layout

```
gen_tests.py       briefs.yml → tests.yaml
gen_configs.py     8 Stage-A configs → configs/
gen_phase4.py      8 Phase-4 configs → configs/
analyze.py         one-task stats on the latest eval (noise floor + sign test)
summarize.py       Stage-A summary + comparison to the custom pipeline
phase4_summary.py  premium-vs-cheap analysis
configs/           generated promptfoo configs (committed)
tests*.yaml, results/   gitignored — generated from confidential briefs.yml
```

## Run

```bash
python gen_tests.py
python gen_configs.py && python gen_phase4.py
npx promptfoo@latest eval -c configs/concept.yaml --env-file ../.env --output results/concept.json
python summarize.py
```

## Data privacy

`tests.yaml`, `tests3.yaml` and `results/` are gitignored — they are generated
from `briefs.yml`, which holds confidential client briefs. Only config templates
(placeholders only) and the analysis scripts are committed. Regenerate the test
data locally with `gen_tests.py`.
