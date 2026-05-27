"""
Unified LLM client for OpenAI + Anthropic.
Provides retry, quota, rate-limit, and error classification on top of the
two providers' SDKs.
"""

import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Optional


# ---- Status taxonomy. Used as enum across CSV/JSONL/resume logic. ----

class Status:
    OK = "ok"
    OK_LENGTH_VIOLATION = "ok_length_violation"
    PARSE_FAIL = "parse_fail"
    REFUSED = "refused"
    TRUNCATED = "truncated"
    API_ERROR = "api_error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"


# "Successfully scored" — used by resume logic. ok_length_violation is
# also kept (we score it, just track separately).
DONE_STATUSES = {Status.OK, Status.OK_LENGTH_VIOLATION}


# ---- Exceptions ----

class LLMError(Exception):
    """Base."""


class QuotaExhaustedError(LLMError):
    pass


class InvalidAPIKeyError(LLMError):
    pass


class CallCapExceededError(LLMError):
    pass


class BudgetExceededError(LLMError):
    pass


# ---- Call result schema ----

@dataclass
class CallResult:
    # identity (resume key = first 5 fields)
    brief_id: str
    task: str
    config_id: str
    model_key: str
    run_id: int

    # request
    prompt: str

    # status
    status: str = Status.API_ERROR

    # response
    raw_response: Optional[str] = None
    parsed_output: Optional[str] = None
    finish_reason: Optional[str] = None  # OpenAI: finish_reason / Anthropic: stop_reason

    # usage / cost
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0  # OpenAI reasoning models only
    cached_input_tokens: int = 0
    cost_usd: float = 0.0

    # observability
    latency_s: float = 0.0
    error: Optional[str] = None
    provider_request_id: Optional[str] = None
    timestamp: Optional[str] = None


# ---- Client ----

class LLMClient:
    """
    Single client that dispatches to OpenAI or Anthropic by model spec.

    Behavior:
      - call_cap / budget_cap / quota_exhausted as kill switches
      - call() never raises; all errors land as Status.* on CallResult
      - _classify_error maps SDK exceptions -> Status enum

    Resume / retry semantics live in scripts/run_experiment.py, not here.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        call_cap: Optional[int] = None,
        budget_cap_usd: Optional[float] = None,
        per_model_cap_usd: Optional[float] = None,
        http_timeout: float = 60.0,
        model_concurrency: Optional[dict[str, int]] = None,
        sdk_max_retries: int = 8,
    ):
        """
        per_model_cap_usd: optional hard cap on accumulated cost for ANY single
        model_key. When a model's running total >= this cap, the next call for
        that model returns Status.BUDGET_EXCEEDED. Calls to other models are
        unaffected. This is independent from `budget_cap_usd` (global cap).
        Default behaviour (None) = no per-model cap.

        model_concurrency: per-model_key cap on simultaneous in-flight calls.
        Critical for Anthropic models — at CP2 Haiku saw 77% rate_limited
        because all 15 worker threads hammered Anthropic at once. With this
        set, threads block on a per-model Semaphore so e.g. haiku stays at 5
        even if the worker pool is 20. Pass from_env() to wire it from
        config.CONCURRENCY automatically.

        sdk_max_retries: forwarded to the OpenAI/Anthropic SDK clients. Both
        SDKs do exponential backoff with jitter and respect Retry-After
        headers on 429. Default 8 (SDK default is 2 — too low for sustained
        burst load).
        """
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.call_cap = call_cap
        self.budget_cap_usd = budget_cap_usd
        self.per_model_cap_usd = per_model_cap_usd
        self.http_timeout = http_timeout
        self.sdk_max_retries = sdk_max_retries

        self._call_count = 0
        self._total_cost_usd = 0.0
        self._per_model_cost_usd: dict[str, float] = {}
        self._quota_exhausted = False
        # Lock for thread-safe counter / quota updates when run with concurrency.
        self._lock = threading.Lock()

        # Per-model semaphores — gate concurrent in-flight calls so a single
        # provider can't get hammered above its RPM/TPM limit. Unknown
        # model_keys fall back to a permissive default.
        self._model_concurrency = dict(model_concurrency or {})
        self._model_semaphores: dict[str, threading.Semaphore] = {
            mk: threading.Semaphore(max(1, n))
            for mk, n in self._model_concurrency.items()
        }
        self._default_semaphore = threading.Semaphore(4)

        # Lazy-init SDK clients (so an empty key for one provider doesn't
        # break the other). SDK clients are thread-safe per httpx docs.
        self._openai = None
        self._anthropic = None

    @classmethod
    def from_env(cls, **overrides):
        try:
            from . import config
        except ImportError:
            import config
        kwargs = dict(
            openai_api_key=config.OPENAI_API_KEY,
            anthropic_api_key=config.ANTHROPIC_API_KEY,
            http_timeout=config.HTTP_TIMEOUT,
            model_concurrency=getattr(config, "CONCURRENCY", None),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---- public ----

    def call(
        self,
        model_key: str,
        prompt: str,
        brief_id: str,
        task: str,
        config_id: str,
        run_id: int = 1,
    ) -> CallResult:
        """
        Main entry point. NEVER raises — every failure ends up as a
        CallResult with status != ok.
        """
        try:
            from . import config as cfg
        except ImportError:
            import config as cfg

        if model_key not in cfg.MODELS:
            raise ValueError(f"Unknown model_key: {model_key}")
        spec = cfg.MODELS[model_key]

        from datetime import datetime, timezone
        result = CallResult(
            brief_id=brief_id,
            task=task,
            config_id=config_id,
            model_key=model_key,
            run_id=run_id,
            prompt=prompt,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )

        # Kill switches — atomic read under lock
        with self._lock:
            if self._quota_exhausted:
                result.status = Status.BUDGET_EXCEEDED
                result.error = "quota exhausted (provider-side)"
                return result
            if self.call_cap is not None and self._call_count >= self.call_cap:
                result.status = Status.BUDGET_EXCEEDED
                result.error = f"call cap {self.call_cap} reached"
                return result
            if self.budget_cap_usd is not None and self._total_cost_usd >= self.budget_cap_usd:
                result.status = Status.BUDGET_EXCEEDED
                result.error = f"budget cap ${self.budget_cap_usd} reached"
                return result
            if self.per_model_cap_usd is not None:
                spent = self._per_model_cost_usd.get(model_key, 0.0)
                if spent >= self.per_model_cap_usd:
                    result.status = Status.BUDGET_EXCEEDED
                    result.error = (
                        f"per-model cap ${self.per_model_cap_usd} reached "
                        f"for {model_key} (spent ${spent:.4f})"
                    )
                    return result

        # Per-model in-flight gate. Threads block here so the worker pool
        # can be large while individual providers stay within RPM/TPM limits.
        # Acquired AFTER budget kill-switches so short-circuits don't queue.
        sem = self._model_semaphores.get(model_key, self._default_semaphore)

        t0 = time.monotonic()
        with sem:
            try:
                if spec["provider"] == "openai":
                    self._call_openai(spec, prompt, result)
                elif spec["provider"] == "anthropic":
                    self._call_anthropic(spec, prompt, result)
                else:
                    raise ValueError(f"Unknown provider: {spec['provider']}")
            except Exception as e:
                self._classify_error(e, result)
            finally:
                result.latency_s = round(time.monotonic() - t0, 3)
                with self._lock:
                    self._call_count += 1
                    self._total_cost_usd += result.cost_usd
                    self._per_model_cost_usd[model_key] = (
                        self._per_model_cost_usd.get(model_key, 0.0) + result.cost_usd
                    )

        return result

    # ---- provider-specific ----

    def _call_openai(self, spec: dict, prompt: str, result: CallResult) -> None:
        """
        OpenAI Chat Completions call. Works for both regular and reasoning
        models (gpt-5*). Per-model params come from config.MODELS[key]["params"].

        Frozen invariants (do not silently change):
          - input_tokens  = usage.prompt_tokens
          - output_tokens = usage.completion_tokens   (already INCLUDES reasoning_tokens)
          - reasoning_tokens recorded separately
          - cost = input/1e6 * price.in + output/1e6 * price.out   (no double-count)
        """
        from openai import OpenAI

        if self._openai is None:
            self._openai = OpenAI(
                api_key=self.openai_api_key,
                timeout=self.http_timeout,
                max_retries=self.sdk_max_retries,
            )

        resp = self._openai.chat.completions.create(
            model=spec["id"],
            messages=[{"role": "user", "content": prompt}],
            **spec["params"],
        )

        choice = resp.choices[0]
        result.raw_response = (choice.message.content or "")
        result.finish_reason = choice.finish_reason
        result.provider_request_id = resp.id

        u = resp.usage
        result.input_tokens = u.prompt_tokens
        result.output_tokens = u.completion_tokens
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            result.reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        cached = getattr(u, "prompt_tokens_details", None)
        if cached is not None:
            result.cached_input_tokens = getattr(cached, "cached_tokens", 0) or 0

        price = spec["price_per_1m"]
        result.cost_usd = round(
            result.input_tokens / 1e6 * price["input"]
            + result.output_tokens / 1e6 * price["output"],
            6,
        )

        if choice.finish_reason == "length":
            result.status = Status.TRUNCATED
        elif choice.finish_reason == "content_filter":
            result.status = Status.REFUSED
        else:
            result.status = Status.OK

    def _call_anthropic(self, spec: dict, prompt: str, result: CallResult) -> None:
        """
        Anthropic Messages call. No seed; reasoning_tokens not applicable.

        stop_reason values: "end_turn" / "max_tokens" / "stop_sequence" / "refusal" / "tool_use"
        """
        from anthropic import Anthropic

        if self._anthropic is None:
            self._anthropic = Anthropic(
                api_key=self.anthropic_api_key,
                timeout=self.http_timeout,
                max_retries=self.sdk_max_retries,
            )

        resp = self._anthropic.messages.create(
            model=spec["id"],
            messages=[{"role": "user", "content": prompt}],
            **spec["params"],
        )

        # Concatenate all text content blocks (usually exactly one).
        text_parts: list[str] = []
        for block in resp.content:
            t = getattr(block, "text", None)
            if t:
                text_parts.append(t)
        result.raw_response = "".join(text_parts)
        result.finish_reason = resp.stop_reason
        result.provider_request_id = resp.id

        result.input_tokens = resp.usage.input_tokens
        result.output_tokens = resp.usage.output_tokens
        # Anthropic prompt caching — if used, cache_read_input_tokens is what we paid less for.
        result.cached_input_tokens = getattr(resp.usage, "cache_read_input_tokens", 0) or 0

        price = spec["price_per_1m"]
        result.cost_usd = round(
            result.input_tokens / 1e6 * price["input"]
            + result.output_tokens / 1e6 * price["output"],
            6,
        )

        if resp.stop_reason == "max_tokens":
            result.status = Status.TRUNCATED
        elif resp.stop_reason == "refusal":
            result.status = Status.REFUSED
        else:
            result.status = Status.OK

    # ---- error classification ----

    def _classify_error(self, e: Exception, result: CallResult) -> None:
        """
        Map SDK exceptions to our Status enum using isinstance, with string
        fallback for unusual cases. Imports are lazy so missing SDKs don't
        break this code path on a single-provider machine.
        """
        result.error = str(e)[:500]

        # Try OpenAI exception types first
        try:
            from openai import (
                APIConnectionError as _OAIConn,
                APITimeoutError as _OAITimeout,
                AuthenticationError as _OAIAuth,
                BadRequestError as _OAIBadReq,
                NotFoundError as _OAINotFound,
                PermissionDeniedError as _OAIPerm,
                RateLimitError as _OAIRate,
            )
            if isinstance(e, _OAIRate):
                result.status = Status.RATE_LIMITED
                return
            if isinstance(e, _OAITimeout):
                result.status = Status.TIMEOUT
                return
            if isinstance(e, _OAIAuth):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _OAIPerm):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _OAINotFound):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _OAIConn):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _OAIBadReq):
                # Could be parameter error OR "credit balance too low"
                low = result.error.lower()
                if any(k in low for k in ("credit", "quota", "billing", "insufficient")):
                    self._quota_exhausted = True
                    result.status = Status.BUDGET_EXCEEDED
                else:
                    result.status = Status.API_ERROR
                return
        except ImportError:
            pass

        # Then Anthropic
        try:
            from anthropic import (
                APIConnectionError as _AConn,
                APITimeoutError as _ATimeout,
                AuthenticationError as _AAuth,
                BadRequestError as _ABadReq,
                NotFoundError as _ANotFound,
                PermissionDeniedError as _APerm,
                RateLimitError as _ARate,
            )
            if isinstance(e, _ARate):
                result.status = Status.RATE_LIMITED
                return
            if isinstance(e, _ATimeout):
                result.status = Status.TIMEOUT
                return
            if isinstance(e, _AAuth):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _APerm):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _ANotFound):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _AConn):
                result.status = Status.API_ERROR
                return
            if isinstance(e, _ABadReq):
                low = result.error.lower()
                if any(k in low for k in ("credit", "quota", "billing", "insufficient")):
                    self._quota_exhausted = True
                    result.status = Status.BUDGET_EXCEEDED
                else:
                    result.status = Status.API_ERROR
                return
        except ImportError:
            pass

        # Unknown — last-resort string heuristic
        low = result.error.lower()
        if "rate" in low and "limit" in low:
            result.status = Status.RATE_LIMITED
        elif "timeout" in low or "timed out" in low:
            result.status = Status.TIMEOUT
        elif any(k in low for k in ("credit", "quota", "billing", "insufficient")):
            self._quota_exhausted = True
            result.status = Status.BUDGET_EXCEEDED
        else:
            result.status = Status.API_ERROR

    # ---- accessors ----

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_cost_usd(self) -> float:
        return round(self._total_cost_usd, 4)

    def cost_for_model(self, model_key: str) -> float:
        """Accumulated USD spent on a single model_key. 0.0 if untouched."""
        with self._lock:
            return round(self._per_model_cost_usd.get(model_key, 0.0), 4)

    def per_model_cost_snapshot(self) -> dict[str, float]:
        """Snapshot of {model_key: cost_usd} for reporting / debugging."""
        with self._lock:
            return {k: round(v, 4) for k, v in self._per_model_cost_usd.items()}

    def reset_counters(self) -> None:
        """Use only between phases. NEVER mid-stage."""
        self._call_count = 0
        self._total_cost_usd = 0.0
        self._per_model_cost_usd = {}
        self._quota_exhausted = False
