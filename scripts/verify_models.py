"""
Verify model availability + parameter acceptance + actual billing fields
BEFORE running any large-scale experiment.

For each model in config.MODELS:
  1. List models from the provider; check if the configured ID exists.
     If not, print similar IDs to help re-pick.
  2. Make ONE minimal call (~10 output tokens) using the configured params.
     Record: status, finish_reason / stop_reason, system_fingerprint,
             input_tokens, output_tokens, reasoning_tokens, cached_tokens,
             provider request id, raw text, actual cost (using prices in config).
  3. For OpenAI reasoning models, additionally probe whether `temperature=0`
     is rejected (the actual confirmation of "is reasoning model").
  4. Dump everything to outputs/model_verification_<ts>.json.
  5. Print a one-line-per-model summary table.

Usage:
    python -m scripts.verify_models

Expected total cost: < $0.05.

After running, update src/config.py:
  - Replace each `id` with the verified value
  - Append `# verified YYYY-MM-DD` to each line
  - If reasoning_tokens > 0 for a model with reasoning_effort=minimal,
    decide whether to keep it (cost overrun risk) — see Engineering Notes.
"""
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src import config as cfg

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROBE_PROMPT = "Reply with the single word: ok"
PROBE_MAX_TOKENS = 20  # reasoning models burn tokens silently; give some headroom

OUT_PATH = cfg.OUTPUTS_DIR / f"model_verification_{datetime.now():%Y%m%d_%H%M%S}.json"


@dataclass
class ProbeResult:
    model_key: str
    model_id: str
    provider: str

    # Step 1: discovery
    listed: Optional[bool] = None
    listed_error: Optional[str] = None
    similar_ids: list = field(default_factory=list)

    # Step 2: primary call (with the configured params)
    call_ok: bool = False
    call_error: Optional[str] = None
    raw_response: Optional[str] = None
    finish_reason: Optional[str] = None
    provider_request_id: Optional[str] = None
    system_fingerprint: Optional[str] = None

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    actual_cost_usd: float = 0.0
    latency_s: float = 0.0

    # Step 3: param-acceptance probes (OpenAI only — Anthropic params don't vary)
    accepts_temperature_0: Optional[bool] = None
    temperature_error: Optional[str] = None
    accepts_max_tokens: Optional[bool] = None  # vs max_completion_tokens

    # Verdict
    is_reasoning_model_confirmed: Optional[bool] = None
    verdict: str = ""


# ---------------- OpenAI ----------------

def _list_openai(client) -> list[str]:
    return [m.id for m in client.models.list().data]


def _try_openai_call(client, model_id: str, params: dict) -> tuple[Any, Optional[Exception]]:
    """One call attempt. Returns (response, None) on success or (None, exception)."""
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            **params,
        )
        return resp, None
    except Exception as e:
        return None, e


def probe_openai(client, model_key: str, spec: dict) -> ProbeResult:
    r = ProbeResult(model_key=model_key, model_id=spec["id"], provider="openai")

    # Step 1: list models
    try:
        all_ids = _list_openai(client)
        r.listed = spec["id"] in all_ids
        if not r.listed:
            family = spec["id"].split("-")[0] + "-" + spec["id"].split("-")[1]  # "gpt-5"
            r.similar_ids = sorted([i for i in all_ids if family in i])[:8]
    except Exception as e:
        r.listed_error = f"{type(e).__name__}: {e}"

    # Step 2: primary call with configured params
    t0 = time.monotonic()
    resp, err = _try_openai_call(client, spec["id"], dict(spec["params"]))
    r.latency_s = round(time.monotonic() - t0, 3)

    if err is not None:
        r.call_ok = False
        r.call_error = f"{type(err).__name__}: {err}"
    else:
        r.call_ok = True
        r.raw_response = resp.choices[0].message.content
        r.finish_reason = resp.choices[0].finish_reason
        r.provider_request_id = resp.id
        r.system_fingerprint = getattr(resp, "system_fingerprint", None)
        u = resp.usage
        r.input_tokens = u.prompt_tokens
        r.output_tokens = u.completion_tokens
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            r.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        cached = getattr(u, "prompt_tokens_details", None)
        if cached is not None:
            r.cached_input_tokens = getattr(cached, "cached_tokens", 0) or 0

        price = spec["price_per_1m"]
        r.actual_cost_usd = round(
            r.input_tokens / 1e6 * price["input"]
            + r.output_tokens / 1e6 * price["output"],  # output_tokens already includes reasoning
            6,
        )

    # Step 3: parameter probes
    # 3a. temperature=0 — fails on reasoning models (this is how we CONFIRM it's a reasoning model)
    _, terr = _try_openai_call(
        client,
        spec["id"],
        {
            "max_completion_tokens": PROBE_MAX_TOKENS,
            "temperature": 0,
        },
    )
    r.accepts_temperature_0 = (terr is None)
    if terr is not None:
        r.temperature_error = f"{type(terr).__name__}: {str(terr)[:200]}"

    # 3b. max_tokens (legacy) vs max_completion_tokens
    _, merr = _try_openai_call(
        client,
        spec["id"],
        {"max_tokens": PROBE_MAX_TOKENS},
    )
    r.accepts_max_tokens = (merr is None)

    # Reasoning model confirmation:
    # - reasoning_tokens > 0  OR  temperature=0 rejected  OR  max_tokens rejected
    r.is_reasoning_model_confirmed = (
        r.reasoning_tokens > 0
        or r.accepts_temperature_0 is False
        or r.accepts_max_tokens is False
    )

    r.verdict = _make_verdict(r, spec)
    return r


# ---------------- Anthropic ----------------

def _list_anthropic(client) -> list[str]:
    out = []
    for page in client.models.list(limit=1000):
        # SDK auto-paginates — but be defensive
        if hasattr(page, "id"):
            out.append(page.id)
    return out


def probe_anthropic(client, model_key: str, spec: dict) -> ProbeResult:
    r = ProbeResult(model_key=model_key, model_id=spec["id"], provider="anthropic")

    # Step 1: list
    try:
        all_ids = _list_anthropic(client)
        r.listed = spec["id"] in all_ids
        if not r.listed:
            family = "sonnet" if "sonnet" in spec["id"] else "haiku" if "haiku" in spec["id"] else ""
            r.similar_ids = sorted([i for i in all_ids if family in i])[:8]
    except Exception as e:
        r.listed_error = f"{type(e).__name__}: {e}"

    # Step 2: call
    t0 = time.monotonic()
    try:
        resp = client.messages.create(
            model=spec["id"],
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            **spec["params"],
        )
        r.latency_s = round(time.monotonic() - t0, 3)
        r.call_ok = True
        r.raw_response = resp.content[0].text if resp.content else ""
        r.finish_reason = resp.stop_reason
        r.provider_request_id = resp.id
        r.input_tokens = resp.usage.input_tokens
        r.output_tokens = resp.usage.output_tokens
        # Anthropic doesn't have reasoning_tokens

        price = spec["price_per_1m"]
        r.actual_cost_usd = round(
            r.input_tokens / 1e6 * price["input"]
            + r.output_tokens / 1e6 * price["output"],
            6,
        )
    except Exception as e:
        r.latency_s = round(time.monotonic() - t0, 3)
        r.call_ok = False
        r.call_error = f"{type(e).__name__}: {e}"

    r.is_reasoning_model_confirmed = False  # current Anthropic models aren't reasoning models in the OpenAI sense
    r.verdict = _make_verdict(r, spec)
    return r


# ---------------- Verdict ----------------

def _make_verdict(r: ProbeResult, spec: dict) -> str:
    if not r.call_ok:
        if r.listed is False:
            return "FAIL: model ID not listed by provider"
        return f"FAIL: {r.call_error[:80]}"
    if r.finish_reason in ("length", "max_tokens"):
        # truncated — fine for verify but worth noting
        return "OK (truncated at probe limit)"
    if spec.get("is_reasoning_model") and r.reasoning_tokens == 0 and r.is_reasoning_model_confirmed is False:
        return "OK — but config says is_reasoning_model=True and we saw none. Reconsider."
    if not spec.get("is_reasoning_model") and r.is_reasoning_model_confirmed:
        return "OK — but model IS reasoning, config says it isn't. Update config."
    return "OK"


# ---------------- Summary table ----------------

def print_summary(results: list[ProbeResult]) -> None:
    print()
    print("=" * 110)
    print(
        f"{'MODEL KEY':<10} {'PROVIDER':<10} {'LISTED':<8} {'CALL':<6} "
        f"{'TEMP=0':<7} {'IN':>5} {'OUT':>5} {'REASON':>7} {'$':>10} {'VERDICT':<40}"
    )
    print("-" * 110)
    for r in results:
        listed = "?" if r.listed is None else ("yes" if r.listed else "NO")
        call = "ok" if r.call_ok else "FAIL"
        temp = "?" if r.accepts_temperature_0 is None else ("yes" if r.accepts_temperature_0 else "NO")
        print(
            f"{r.model_key:<10} {r.provider:<10} {listed:<8} {call:<6} {temp:<7} "
            f"{r.input_tokens:>5} {r.output_tokens:>5} {r.reasoning_tokens:>7} "
            f"{r.actual_cost_usd:>10.6f} {r.verdict[:40]:<40}"
        )
    print("=" * 110)

    total = sum(r.actual_cost_usd for r in results)
    print(f"\nTotal probe cost: ${total:.6f}")
    print(f"Full dump:        {OUT_PATH}\n")


# ---------------- main ----------------

def main():
    # Lazy import so missing one SDK doesn't break the other
    openai_client = None
    anthropic_client = None

    needed_providers = {spec["provider"] for spec in cfg.MODELS.values()}

    if "openai" in needed_providers:
        if not cfg.OPENAI_API_KEY:
            print("WARNING: OPENAI_API_KEY not set; OpenAI probes will be skipped.")
        else:
            try:
                from openai import OpenAI
                openai_client = OpenAI(api_key=cfg.OPENAI_API_KEY, timeout=cfg.HTTP_TIMEOUT)
            except ImportError:
                print("ERROR: openai SDK not installed. Run: pip install -r requirements.txt")
                return

    if "anthropic" in needed_providers:
        if not cfg.ANTHROPIC_API_KEY:
            print("WARNING: ANTHROPIC_API_KEY not set; Anthropic probes will be skipped.")
        else:
            try:
                from anthropic import Anthropic
                anthropic_client = Anthropic(api_key=cfg.ANTHROPIC_API_KEY, timeout=cfg.HTTP_TIMEOUT)
            except ImportError:
                print("ERROR: anthropic SDK not installed. Run: pip install -r requirements.txt")
                return

    results: list[ProbeResult] = []
    for model_key, spec in cfg.MODELS.items():
        print(f"Probing {model_key} ({spec['id']}) ...")
        if spec["provider"] == "openai":
            if openai_client is None:
                print(f"  skipped (no OpenAI client)")
                continue
            results.append(probe_openai(openai_client, model_key, spec))
        elif spec["provider"] == "anthropic":
            if anthropic_client is None:
                print(f"  skipped (no Anthropic client)")
                continue
            results.append(probe_anthropic(anthropic_client, model_key, spec))
        else:
            print(f"  skipped (unknown provider: {spec['provider']})")

    # Dump full results
    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False, default=str)

    print_summary(results)

    # Exit with non-zero if any FAIL — so CI / chained commands stop
    any_fail = any(not r.call_ok or r.listed is False for r in results)
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
