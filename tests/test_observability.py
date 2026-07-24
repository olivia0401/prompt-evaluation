"""Langfuse tracing must be a silent no-op unless it is configured."""
from src import observability
from src.llm_client import CallResult, Status


def _result():
    return CallResult(brief_id="b", task="concept_relevant", config_id="full",
                      model_key="haiku", run_id=1, prompt="p",
                      status=Status.OK, raw_response="one sentence.",
                      input_tokens=10, output_tokens=5, cost_usd=0.001, latency_s=0.2)


def test_disabled_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert observability.enabled() is False


def test_log_generation_never_raises_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    # Reset the cached client so the env change is picked up.
    monkeypatch.setattr(observability, "_resolved", False)
    monkeypatch.setattr(observability, "_client", None)
    observability.log_generation(_result())   # must not raise
    observability.flush()                      # must not raise
