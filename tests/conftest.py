"""
Shared pytest configuration.

Some tests exercise the real experiment dataset (`briefs.yml`), which is
confidential and gitignored — so it is absent in CI and any fresh checkout.
Rather than fail those tests with a FileNotFoundError, we skip them when the
data isn't present, mirroring how tests/test_llm_client_live.py self-skips
without API keys. Tests that use synthetic/fixture data are unaffected.

Detection is generic (no hardcoded test-name list): a test is skipped only if
its function directly references `build_todo` or `load_briefs` — the two entry
points that read briefs.yml.
"""
import pytest

from src import config as cfg

_BRIEFS_CONSUMERS = {"build_todo", "load_briefs"}


def _needs_real_briefs(item) -> bool:
    func = getattr(item, "function", None)
    code = getattr(func, "__code__", None)
    if code is None:
        return False
    return bool(_BRIEFS_CONSUMERS.intersection(code.co_names))


def pytest_collection_modifyitems(config, items):
    if cfg.BRIEFS_FILE.exists():
        return  # real data present — run everything
    skip = pytest.mark.skip(reason=f"requires {cfg.BRIEFS_FILE.name} (confidential, not in CI)")
    for item in items:
        if _needs_real_briefs(item):
            item.add_marker(skip)
