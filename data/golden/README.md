# Golden evaluation set

The fixed, hand-annotated reference the evidence-grounded quality gate scores
against. Unlike `outputs/` this is **not** regenerable, so it is the one thing
under `data/` that is version-controlled (see the `!data/golden/` rule in
`.gitignore`).

```
queue.jsonl            61 real items to annotate            (annotation-queue/v1)
annotations_raw.jsonl  per-annotator decisions, append-only  (one pass so far)
annotations.json       the built golden set                  (produced by --build; not committed)
manifest.json          SHA-256 of README.md, queue.jsonl and annotations_raw.jsonl
```

Files here are checked out with LF line endings on every platform
(`.gitattributes`), so the manifest hashes match on Windows and Linux alike.

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

**16 of the 61 verdicts (26%) cite no signal at all.** That is measurable
before any annotation: citation completeness on this set can be at most 74%,
and every one of those verdicts is ungrounded whether or not it happens to be
correct. Those 16 are the most informative items in the set — a correct call
the engine cannot justify and a lucky guess look identical from the outside,
and only annotation separates them.

## Annotate it

```bash
python -m scripts.annotate_golden --annotator <you>          # one pass over 61 items
python -m scripts.annotate_golden --annotator <second>       # a second pass
python -m scripts.annotate_golden --agreement                # inter-annotator kappa
python -m scripts.annotate_golden --build                    # -> annotations.json
```

Per cited signal you decide two things: does it **genuinely support** this
verdict (or did it merely match?), and how strong is it as evidence, 1-5. Then
whether the verdict itself was correct — recorded separately, because *right for
the wrong reason* is a distinct and more dangerous failure than *wrong*.

### Current annotation status

One pass exists, by a single annotator (`annotations_raw.jsonl`). Its
timestamps span 6 min 35 s for all 61 items, about 6.5 seconds per item. In
that pass every cited signal was marked as supporting its verdict, every
evidence-strength rating was a 3 or a 4, and every verdict was marked correct.

That is a quick first pass, not a careful adjudicated gold set: at that speed
it is closer to a confirmation of the rule engine's output than an independent
judgement of it. No second annotator has been run, so `--agreement` has
nothing to compare yet. Treat any metric built from this pass as a pipeline
demonstration, not as evidence about the engine.

## Score and gate it

```bash
python -m scripts.build_golden_manifest --root data/golden --output data/golden/manifest.json --version 2026.09.24
python -m scripts.build_golden_manifest --root data/golden --verify data/golden/manifest.json
python -m scripts.build_quality_report --input data/golden/annotations.json --manifest data/golden/manifest.json --output outputs/quality_report.json
python -m service.quality_gate outputs/quality_report.json --manifest data/golden/manifest.json
```

Every `quality-report/v1` carries `provenance.golden_manifest_sha256`, and the
gate refuses a report without it. With `--manifest`, the gate also checks that
the hash matches that manifest file and that every file the manifest lists is
unchanged on disk, so the dataset cannot drift underneath a report while every
number stays green. Without `--manifest` the gate only checks that a hash is
present.

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
