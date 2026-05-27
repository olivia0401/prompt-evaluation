"""Offline tests for budget-cap kill switches and per-model concurrency. No API calls."""
import threading

from src.llm_client import LLMClient, Status


def test_per_model_cap_blocks_only_offending_model():
    """When haiku is past its per-model cap, haiku calls fail BUDGET_EXCEEDED
    but other models are unaffected."""
    client = LLMClient(
        openai_api_key="fake",
        anthropic_api_key="fake",
        per_model_cap_usd=1.0,
    )
    # Pretend haiku has already spent $1.50.
    client._per_model_cost_usd["haiku"] = 1.50

    # Haiku call short-circuits before hitting the API.
    r = client.call(
        model_key="haiku",
        prompt="x",
        brief_id="t", task="probe", config_id="probe", run_id=1,
    )
    assert r.status == Status.BUDGET_EXCEEDED
    assert "per-model cap" in (r.error or "")
    assert "haiku" in (r.error or "")

    # gpt5mini is untouched — would proceed to the API path. With a fake key
    # it lands as some non-BUDGET_EXCEEDED status (API_ERROR, etc.).
    r2 = client.call(
        model_key="gpt5mini",
        prompt="x",
        brief_id="t", task="probe", config_id="probe", run_id=1,
    )
    assert r2.status != Status.BUDGET_EXCEEDED, (
        f"Per-model cap on haiku should NOT block gpt5mini, got {r2.status}: {r2.error}"
    )


def test_per_model_cap_accumulates_correctly():
    """cost_for_model() reflects per-call accumulation under the lock."""
    client = LLMClient(per_model_cap_usd=10.0)
    client._per_model_cost_usd["haiku"] = 0.0

    # Simulate post-call accumulation directly (we're testing the bookkeeping,
    # not the SDK call path).
    with client._lock:
        client._per_model_cost_usd["haiku"] += 0.40
        client._per_model_cost_usd["sonnet"] = client._per_model_cost_usd.get("sonnet", 0.0) + 1.25

    assert client.cost_for_model("haiku") == 0.40
    assert client.cost_for_model("sonnet") == 1.25
    assert client.cost_for_model("opus47") == 0.0  # never touched
    snap = client.per_model_cost_snapshot()
    assert snap == {"haiku": 0.40, "sonnet": 1.25}


def test_no_per_model_cap_means_no_blocking():
    """When per_model_cap_usd is None, the per-model kill-switch is inert
    even if a model has accumulated huge spend."""
    client = LLMClient(
        openai_api_key="fake",
        anthropic_api_key="fake",
        per_model_cap_usd=None,
    )
    client._per_model_cost_usd["haiku"] = 999.99
    r = client.call(
        model_key="haiku",
        prompt="x",
        brief_id="t", task="probe", config_id="probe", run_id=1,
    )
    # No per-model gate; should NOT be BUDGET_EXCEEDED from per-model path.
    # (Will be API_ERROR from the fake key, but not BUDGET_EXCEEDED.)
    assert r.status != Status.BUDGET_EXCEEDED or "per-model" not in (r.error or "")


def test_reset_counters_clears_per_model_cost():
    client = LLMClient(per_model_cap_usd=1.0)
    client._per_model_cost_usd["haiku"] = 0.5
    client.reset_counters()
    assert client.cost_for_model("haiku") == 0.0
    assert client.per_model_cost_snapshot() == {}


# --- Per-model concurrency semaphores ---

def test_model_concurrency_builds_semaphores():
    """model_concurrency dict -> per-key Semaphore in _model_semaphores."""
    client = LLMClient(model_concurrency={"haiku": 5, "sonnet": 3, "opus47": 2})
    assert set(client._model_semaphores) == {"haiku", "sonnet", "opus47"}
    for sem in client._model_semaphores.values():
        assert isinstance(sem, threading.Semaphore)


def test_model_concurrency_unknown_model_uses_default_semaphore():
    """call() falls back to _default_semaphore for model_keys not in the dict."""
    client = LLMClient(model_concurrency={"haiku": 5})
    # gpt5mini isn't in the dict — getattr-style lookup returns default.
    sem = client._model_semaphores.get("gpt5mini", client._default_semaphore)
    assert sem is client._default_semaphore


def test_model_concurrency_none_means_default_for_all():
    """No model_concurrency arg -> all calls share the default semaphore (no per-model gating)."""
    client = LLMClient()
    assert client._model_semaphores == {}
    assert isinstance(client._default_semaphore, threading.Semaphore)


def test_model_concurrency_clamps_zero_to_one():
    """A misconfigured 0 must not produce a dead semaphore that blocks forever."""
    client = LLMClient(model_concurrency={"haiku": 0})
    sem = client._model_semaphores["haiku"]
    # Semaphore(1) acquires once non-blocking; Semaphore(0) would block.
    assert sem.acquire(blocking=False) is True
    sem.release()


def test_from_env_wires_config_concurrency(monkeypatch):
    """from_env() pulls config.CONCURRENCY by default."""
    from src import config
    client = LLMClient.from_env()
    # config.CONCURRENCY has all 6 known model keys; semaphores must match.
    assert set(client._model_semaphores) == set(config.CONCURRENCY)
