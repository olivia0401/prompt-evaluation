# Where this sits

Everything in this repository has a name the rest of the field already uses.
This file is that mapping — what each component is called here, what it is
called elsewhere, and what was deliberately added on top.

It exists because "custom scoring pipeline" and "RAGAS-equivalent faithfulness
scoring with a measured noise floor and paired significance testing" describe
the same code, and only one of them tells you whether the author knows the
field.

## In 60 seconds

There is no industry-standard LLM eval tool as of 2026 and no single tool covers
both development-time evaluation and production monitoring, so mature teams run
one of each. This repo is a **development-time evaluation platform** — the
promptfoo / DeepEval / RAGAS slot — with **Langfuse in the production-tracing
slot** and **Giskard in the red-team slot**.

Two things it does that none of those tools do for you, and which are most of
the point:

1. **It measures its own noise before setting any threshold.** Every eval tool
   ships default thresholds. All of them are guesses. Here the tie band is 2σ of
   measured rerun variance, and every downstream threshold — decision rules, CI
   tolerance — is derived from it rather than picked.
2. **It refuses to report on an inadequate sample.** Sample-size floors gate
   before threshold checks, so "not enough data to judge" is its own outcome and
   can never be mistaken for a pass.

## Component map

### Scoring

| Here | Called elsewhere | What is different |
|---|---|---|
| `src/evaluators.py` — embedding cosine, ROUGE-L, length compliance | Reference-based scoring; RAGAS `answer_similarity`; the semantic-similarity family in DeepEval | Failed calls are excluded with their own rate, never scored 0. Cache keys are scoped to the embedding backend so an offline run cannot silently poison an online one. |
| `score_keywords` — Porter-stemmed set P/R/F1 | Near-match term-set scoring | Duplicates that stem to one term are de-duplicated rather than double-counted. |

### Evidence and grounding

| Here | Called elsewhere | What is different |
|---|---|---|
| `score_evidence_chain` → `groundedness` | RAGAS `faithfulness`; DeepEval `FaithfulnessMetric` | Grounding requires a *gold-accepted* evidence item that also clears a source-quality floor. A claim resting only on weak evidence scores as ungrounded even when it is correct. |
| `citation_completeness`, `evidence_precision` | RAGAS `context_precision` / `context_recall` family | Computed against human claim→evidence annotations, not against retrieval overlap. Lexical overlap is never treated as proof that a claim is true. |
| `unsupported_claim_rate` | Hallucination rate; DeepEval `HallucinationMetric` | Reported as a rate with its denominator attached, and pooled (micro) rather than averaged across examples. |
| `score_entity_resolution` | B-cubed F1 (Bagga & Baldwin), standard in coreference evaluation | Reports over-merge and under-merge separately, because a namesake collapse and a split identity are different product failures. |
| `src/golden_manifest.py` | Dataset versioning; provenance / supply-chain integrity | SHA-256 per file, and the gate refuses any report that does not carry the manifest hash. A report that cannot name the bytes it scored is not auditable. |

### Judge calibration

| Here | Called elsewhere | What is different |
|---|---|---|
| `weighted_kappa`, `calibrate_judge` | LLM-as-judge calibration; "human review alignment" in Braintrust / LangSmith | Reported under **both** linear and quadratic weighting, because they differ by up to 0.2 on the same ratings and a κ without its weighting scheme cannot be read. Category set fixed to the full scale, not inferred from observed values. Zero-variance input raises rather than returning 1.0. |
| `scripts/rate_samples.py` | Human review / annotation UI | The judge's score and the automatic metric are hidden while rating, and order is shuffled. An anchored human agreeing with the judge is not evidence. A second pass gives intra-rater κ — the ceiling any judge-vs-human κ can honestly claim. |
| `scripts/pairwise_judge.py` | Pairwise LLM-as-judge; the MT-Bench / Chatbot Arena methodology | A/B position swapped and averaged, so what is measured is preference and not position bias. |

### Statistics — the layer the tools leave to you

| Here | Called elsewhere | Note |
|---|---|---|
| `NOISE_FLOOR_COSINE`, `scripts/measure_noise.py` | Test-retest reliability / measurement repeatability | Borrowed from metrology and psychometrics, not from LLM eval tooling — no eval framework ships this, and every one of them asks you for a threshold anyway. |
| `_paired_signed_rank_p` | Wilcoxon signed-rank test on paired differences | Comparison is A vs B on the *same* items. Two independent means need roughly 10× the sample for the same power. |
| Leave-one-brief-out | Leave-one-out stability / jackknife | If dropping one item changes the winner, there is no winner — there is an item. |
| `_bootstrap_ci_mean`, bootstrap κ CI | Percentile bootstrap | Decisions are taken on the interval bound, not the point estimate. |
| **Known gap:** no multiple-comparison correction | Post-selection inference; the winner's curse | The winner is chosen and tested on the same 23 briefs across ~142 recipes × 9 tasks, so reported p-values are anti-conservative. Fixes: a selection/holdout split, or Holm / Benjamini–Hochberg. Documented rather than hidden. |

### Gates and infrastructure

| Here | Called elsewhere | What is different |
|---|---|---|
| `service/ci_gate.py` | Eval regression gate; DeepEval's pytest assertions; Braintrust CI gates | Pins the baselined recipe instead of taking `max` over candidates (an order statistic that can silently describe a different recipe each run), and never gates tighter than measured noise. |
| `service/quality_gate.py` | Release gate | Sample-size floors run *before* threshold checks. Metrics may be excluded only by a declared reason, and declared exclusions print even on a PASS. |
| `src/observability.py` | LLM tracing — Langfuse, aligning with OpenTelemetry GenAI semantic conventions | Off unless keys are set; no-op rather than a hard dependency. |
| `scripts/scan_giskard.py` | LLM red-teaming / vulnerability scanning — Giskard, promptfoo redteam | Answers "how does it break", which is a different question from "how well does it score". |
| `promptfoo-real/` | promptfoo | Used as an independent cross-check of the custom pipeline's conclusions, not as the primary engine. |
| `e2e/` — 6 personas × 3 engines | Multi-tenant authorisation testing | Not LLM-specific and not novel; it is table stakes that eval platforms routinely skip. `retries: 0`, with the one observed flake root-caused rather than retried. |
| `BUDGET_CAP`, per-model caps | Cost governance / spend guardrails | Hard caps, resume-safe runs, and cost reported per evaluated item. |

## What is deliberately not here

- **Agent-trajectory evaluation** (tool-call correctness, step efficiency). The
  system under evaluation is single-turn extraction; adding trajectory metrics
  would be scaffolding with nothing to measure.
- **Online A/B testing.** There is no production traffic to split.
- **A vector database.** 23 briefs and ~142 recipes fit in memory; a vector
  store here would be architecture cosplay.

## Keeping this file honest

The tool names above go stale roughly every six months. The concepts under them
— paired testing, measured repeatability, chance-corrected agreement,
groundedness, post-selection inference — have been stable for decades and are
what this repo is actually built on. **When re-reading this file, check the
right-hand columns first: the tool names are the perishable part.**

Last mapped against the landscape: 2026-08.
