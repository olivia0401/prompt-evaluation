"""Langfuse tracing for the LLM calls — off unless LANGFUSE_* env vars are set.

When it's on, every LLMClient.call() logs one generation so I can inspect
individual calls (prompt, output, tokens, cost, latency) instead of only the
aggregate CSVs. Best-effort by design: any error here is swallowed so a trace
can never sink an eval run. Setup lives in requirements-qe.txt / README.
"""
import os

_client = None
_resolved = False


def enabled():
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY")
                and os.environ.get("LANGFUSE_SECRET_KEY"))


def _client_or_none():
    # Build the client once. None = disabled, or the SDK isn't installed.
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    if enabled():
        try:
            import atexit
            from langfuse import get_client
            _client = get_client()
            atexit.register(flush)   # short runs need an explicit flush to send
        except Exception:
            _client = None
    return _client


def log_generation(result):
    client = _client_or_none()
    if client is None:
        return
    try:
        gen = client.start_observation(name=f"{result.task}/{result.config_id}",
                                       as_type="generation")
        gen.update(
            model=result.model_key,
            input=result.prompt,
            output=result.parsed_output or result.raw_response,
            usage_details={"input_tokens": result.input_tokens,
                           "output_tokens": result.output_tokens},
            metadata={
                "brief_id": result.brief_id,
                "config_id": result.config_id,
                "run_id": result.run_id,
                "status": result.status,
                "finish_reason": result.finish_reason,
                "cost_usd": result.cost_usd,
                "latency_s": result.latency_s,
                "provider_request_id": result.provider_request_id,
            },
        )
        gen.end()
    except Exception:
        pass


def flush():
    client = _client_or_none()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
