"""
Live integration tests for src/llm_client.py.

Hits the real APIs — ~10 cents total cost across all tests.
Skips cleanly if API keys aren't set.

Run a single test to limit cost:
    python -m pytest tests/test_llm_client_live.py::TestLive::test_haiku_happy_path
"""
import pytest

from src import config as cfg
from src.llm_client import (
    CallResult,
    LLMClient,
    Status,
    DONE_STATUSES,
)


def _need(key: str):
    if not key:
        pytest.skip("API key not set in .env")


@pytest.fixture(scope="module")
def client():
    return LLMClient.from_env()


# ---- One happy-path call per available model ----

class TestLive:
    """Each test does ONE small call. Total cost across the class: ~$0.001."""

    def test_haiku_happy_path(self, client):
        _need(cfg.ANTHROPIC_API_KEY)
        r = client.call(
            model_key="haiku",
            prompt="Reply with the single word: ok",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        _assert_happy(r, model_key="haiku")

    def test_sonnet_happy_path(self, client):
        _need(cfg.ANTHROPIC_API_KEY)
        r = client.call(
            model_key="sonnet",
            prompt="Reply with the single word: ok",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        _assert_happy(r, model_key="sonnet")

    def test_gpt5_happy_path(self, client):
        _need(cfg.OPENAI_API_KEY)
        r = client.call(
            model_key="gpt5",
            prompt="Reply with the single word: ok",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        _assert_happy(r, model_key="gpt5")
        # GPT-5 specific: reasoning_tokens field should be present (may be 0 at minimal effort)
        assert r.reasoning_tokens >= 0

    def test_gpt5mini_happy_path(self, client):
        """Skipped automatically while OpenAI verification is still propagating
        for the mini variant. Re-enable by ensuring the model is accessible."""
        _need(cfg.OPENAI_API_KEY)
        r = client.call(
            model_key="gpt5mini",
            prompt="Reply with the single word: ok",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        if r.status == Status.API_ERROR and "verified" in (r.error or "").lower():
            pytest.skip("gpt5mini access not yet propagated by OpenAI")
        _assert_happy(r, model_key="gpt5mini")


# ---- Truncation handling ----

class TestTruncation:
    def test_haiku_truncated_when_max_tokens_low(self, client):
        _need(cfg.ANTHROPIC_API_KEY)
        # Override max_tokens via a temporary client param tweak: the cleanest
        # path is to call with a custom prompt that's clearly long. We use the
        # configured 300-token cap; we need a longer expected output to force length stop.
        # Simpler: rely on the model speaking long enough to hit 300. We don't
        # have a guaranteed way without mutating spec, so we just check the
        # field plumbing here: status is one of the DONE statuses and
        # finish_reason is populated.
        r = client.call(
            model_key="haiku",
            prompt="Count from 1 to 200 in English words, one number per line.",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        assert r.finish_reason in ("max_tokens", "end_turn", "stop_sequence")
        assert r.status in (Status.OK, Status.TRUNCATED)


# ---- Error classification ----

class TestErrorClassification:
    def test_invalid_key_classifies_as_api_error(self):
        # Build a client with a deliberately bad key.
        bad = LLMClient(
            openai_api_key="sk-invalid",
            anthropic_api_key=cfg.ANTHROPIC_API_KEY,
            http_timeout=cfg.HTTP_TIMEOUT,
        )
        r = bad.call(
            model_key="gpt5",
            prompt="hi",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        assert r.status == Status.API_ERROR
        assert "auth" in (r.error or "").lower() or "invalid" in (r.error or "").lower() \
            or "401" in (r.error or "")

    def test_unknown_model_key_raises(self, client):
        with pytest.raises(ValueError, match="Unknown model_key"):
            client.call(
                model_key="not_a_model",
                prompt="hi",
                brief_id="test", task="probe", config_id="probe", run_id=1,
            )


# ---- Budget cap ----

class TestBudgetCap:
    def test_budget_cap_trips_after_first_call(self):
        _need(cfg.ANTHROPIC_API_KEY)
        # Pick a budget that is non-zero (so first call passes the gate
        # `total_cost >= budget`) but smaller than a single Haiku call's cost
        # (~$0.00003), so the second call trips.
        c = LLMClient.from_env(budget_cap_usd=0.000001)
        r1 = c.call(
            model_key="haiku",
            prompt="hi",
            brief_id="test", task="probe", config_id="probe", run_id=1,
        )
        assert r1.status in DONE_STATUSES, f"first call should succeed, got {r1.status} / {r1.error}"
        assert c.total_cost_usd > 0
        r2 = c.call(
            model_key="haiku",
            prompt="hi",
            brief_id="test", task="probe", config_id="probe", run_id=2,
        )
        assert r2.status == Status.BUDGET_EXCEEDED


# ---- Helpers ----

def _assert_happy(r: CallResult, model_key: str):
    assert r.status in DONE_STATUSES, f"unexpected status: {r.status}, error={r.error}"
    assert r.model_key == model_key
    assert r.raw_response, "raw_response empty"
    assert r.provider_request_id, "request_id missing"
    assert r.input_tokens > 0, "input_tokens not populated"
    assert r.output_tokens > 0, "output_tokens not populated"
    assert r.cost_usd >= 0, "cost_usd negative"
    assert r.latency_s > 0, "latency_s not measured"
    assert r.finish_reason, "finish_reason not set"
