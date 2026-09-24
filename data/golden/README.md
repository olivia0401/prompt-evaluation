# Golden evaluation set

The fixed, hand-annotated reference the evidence-grounded quality gate scores
against. Unlike `outputs/` this is **not** regenerable, so it is the one thing
under `data/` that is version-controlled (see the `!data/golden/` rule in
`.gitignore`).

```
queue.jsonl            61 real items awaiting annotation  (annotation-queue/v1)
annotations_raw.jsonl  per-annotator decisions, append-only   (produced by you)
annotations.json       the built golden set                   (produced by --build)
manifest.json          SHA-256 fingerprint of every file above
```

## Where the queue came from

Real output from a real shipped product, not invented examples. `parent-check`'s
rule engine classifies a message and reports the exact signals that fired; a
verdict is a claim and each fired signal is a piece of cited evidence.

```bash
cd ../parent-check
python export_eval_queue.py --output "../prompt test/data/golden/queue.jsonl"
```

61 items across four strata:

| Stratum | Items |
|---|---|
| `scam/danger` | 31 |
| `benign/ok` | 18 |
| `health/caution` | 11 |
| `health/danger` | 1 |

**16 of the 61 verdicts cite no signal at all.** That is measurable before any
annotation: a 26% ceiling on citation completeness, and every one of those
verdicts is ungrounded whether or not it happens to be correct. Those 16 are the
most informative items in the set — a correct call the engine cannot justify and
a lucky guess look identical from the outside, and only annotation separates
them.

## Annotate it

```bash
python -m scripts.annotate_golden --annotator <you>          # ~60-90 min for 61
python -m scripts.annotate_golden --annotator <second>       # a second pass
python -m scripts.annotate_golden --agreement                # inter-annotator kappa
python -m scripts.annotate_golden --build                    # -> annotations.json
```

Per cited signal you decide two things: does it **genuinely support** this
verdict (or did it merely match?), and how strong is it as evidence, 1-5. Then
whether the verdict itself was correct — recorded separately, because *right for
the wrong reason* is a distinct and more dangerous failure than *wrong*.

## Score and gate it

```bash
python -m scripts.build_golden_manifest --root data/golden --output data/golden/manifest.json --version 2026.08.22
python -m scripts.build_golden_manifest --root data/golden --verify data/golden/manifest.json
python -m scripts.build_quality_report --input data/golden/annotations.json \
  --manifest data/golden/manifest.json --output outputs/quality_report.json
python -m service.quality_gate outputs/quality_report.json
```

Every `quality-report/v1` carries `provenance.golden_manifest_sha256`, and the
gate refuses a report without it. A report that does not name the exact bytes it
was scored against is not auditable — the dataset can drift underneath it while
every number stays green.

## What this corpus does not measure, and why that is written down

Three of the six gate metrics genuinely cannot be computed here. They are
**declared** in `annotations.json` under `not_applicable`, each with a reason,
rather than quietly omitted — the gate treats an undeclared missing metric as a
failure, and prints declared exclusions even on a PASS.

| Metric | Why not here |
|---|---|
| `entity_resolution_f1` | No entity mentions to resolve in message triage. |
| `source_acceptable_rate` | The engine emits no source-quality prediction, so there is nothing to compare the annotator's rating against. |
| `judge_weighted_kappa` | No LLM judge scores this corpus; judge calibration is measured separately by `scripts.compute_kappa`. |

That leaves **groundedness, citation completeness and unsupported-claim rate**
live — the three that speak to hallucination, which is the point.

## Honest limits

Even fully annotated, this set is small and single-domain. Before a number from
it is quoted as evidence about a system:

| | Needed | Why |
|---|---|---|
| Size | ≥150 items, stratified | so a per-stratum rate has a usable interval |
| Annotators | ≥2 independent, agreement reported | one annotator is an opinion; without agreement you cannot separate annotator noise from model error |
| Adjudication | a written rule for disagreements | otherwise agreement is negotiated rather than measured |
| Provenance | source and retrieval date per evidence item | evidence rots; an unreachable source is not a source |

`--agreement` reports inter-annotator kappa on the support decision. Below ~0.6
the disagreement is in the **instructions**, not the annotators: sharpen the
support criterion and re-annotate rather than averaging two different
definitions of the same word.
