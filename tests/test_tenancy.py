"""Multi-tenant isolation and role enforcement.

Six authenticated personas across two tenants:

    acme-admin      acme     admin       read + write + cancel
    acme-operator   acme     operator    read + write
    acme-viewer     acme     viewer      read only
    globex-admin    globex   admin
    globex-operator globex   operator
    globex-viewer   globex   viewer

The same six personas drive the Playwright suite in ``e2e/`` against a real
server in three browsers. These tests are the fast layer: same assertions, no
browser, so a broken scope fails in seconds on every push rather than in the
nightly browser run.

The load-bearing assertion in the whole file is that a cross-tenant read returns
**404 and not 403**. 403 confirms the row exists, which turns run ids into an
enumeration oracle. A test that accepts either status would pass while the leak
is live, so these assert the exact code.
"""
from __future__ import annotations

import json

import pytest

from service.auth import PrincipalRegistry

# Keys are test fixtures, not secrets — they never leave this file or e2e/.
PERSONAS = [
    {"key": "k-acme-admin", "name": "acme-admin", "tenant": "acme", "role": "admin"},
    {"key": "k-acme-operator", "name": "acme-operator", "tenant": "acme", "role": "operator"},
    {"key": "k-acme-viewer", "name": "acme-viewer", "tenant": "acme", "role": "viewer"},
    {"key": "k-globex-admin", "name": "globex-admin", "tenant": "globex", "role": "admin"},
    {"key": "k-globex-operator", "name": "globex-operator", "tenant": "globex", "role": "operator"},
    {"key": "k-globex-viewer", "name": "globex-viewer", "tenant": "globex", "role": "viewer"},
]


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'tenancy.db').as_posix()}")
    monkeypatch.setenv("INLINE_JOBS", "1")
    monkeypatch.setenv("SERVICE_PRINCIPALS", json.dumps(PERSONAS))
    monkeypatch.setenv("API_TOKEN", "")

    import importlib

    from fastapi.testclient import TestClient

    from service import api, db, models, repository, settings
    # Reload order matters: db.py builds a fresh declarative Base, so models
    # must be reloaded against it or Base.metadata is empty and create_all
    # silently makes no tables. Reloading api last rebinds its repo reference.
    for module in (settings, db, models, repository, api):
        importlib.reload(module)

    # Runs are created directly through the repository so these tests never
    # depend on the runner (or spend money); only the read/authz paths matter.
    monkeypatch.setattr(api, "enqueue_run", lambda run_id: None)
    with TestClient(api.app) as c:
        yield c


def _create_run(client, key: str, note: str) -> str:
    resp = client.post("/runs", json={"stage": "phase0", "note": note}, headers=_auth(key))
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# --------------------------------------------------------------- registry level

def test_registry_rejects_two_personas_sharing_one_key():
    """Shared keys make the audit trail a guess about who did what."""
    dupes = [
        {"key": "same", "name": "a", "tenant": "acme", "role": "admin"},
        {"key": "same", "name": "b", "tenant": "globex", "role": "viewer"},
    ]
    with pytest.raises(ValueError, match="duplicate key"):
        PrincipalRegistry.from_env(json.dumps(dupes), None)


def test_registry_rejects_an_unknown_role():
    bad = [{"key": "k", "name": "a", "tenant": "acme", "role": "superuser"}]
    with pytest.raises(ValueError, match="unknown role"):
        PrincipalRegistry.from_env(json.dumps(bad), None)


def test_open_mode_only_when_nothing_is_configured():
    assert PrincipalRegistry.from_env(None, None).open_mode is True
    assert PrincipalRegistry.from_env(None, "legacy-token").open_mode is False
    assert PrincipalRegistry.from_env(json.dumps(PERSONAS), None).open_mode is False


# ------------------------------------------------------------ authentication

def test_health_is_public_and_everything_else_is_not(client):
    assert client.get("/health").status_code == 200
    for path in ("/whoami", "/runs", "/quality-reports"):
        assert client.get(path).status_code == 401, path


@pytest.mark.parametrize("persona", PERSONAS, ids=lambda p: p["name"])
def test_every_persona_authenticates_as_itself(client, persona):
    body = client.get("/whoami", headers=_auth(persona["key"])).json()
    assert body == {"name": persona["name"], "tenant": persona["tenant"],
                    "role": persona["role"], "open_mode": False}


def test_a_wrong_or_malformed_key_is_401(client):
    assert client.get("/whoami", headers=_auth("not-a-key")).status_code == 401
    assert client.get("/whoami", headers={"Authorization": "Basic k-acme-admin"}).status_code == 401
    assert client.get("/whoami", headers={"Authorization": "k-acme-admin"}).status_code == 401


# --------------------------------------------------------------- authorisation

@pytest.mark.parametrize("key", ["k-acme-admin", "k-acme-operator"])
def test_write_roles_can_create_a_run(client, key):
    assert client.post("/runs", json={"stage": "phase0"}, headers=_auth(key)).status_code == 201


@pytest.mark.parametrize("key", ["k-acme-viewer", "k-globex-viewer"])
def test_viewers_are_authenticated_but_cannot_write(client, key):
    """403, not 401: the key is valid, the role is not sufficient."""
    resp = client.post("/runs", json={"stage": "phase0"}, headers=_auth(key))
    assert resp.status_code == 403
    assert "cannot write" in resp.json()["detail"]


def test_only_admin_can_cancel(client):
    run_id = _create_run(client, "k-acme-operator", "cancel-me")
    denied = client.post(f"/runs/{run_id}/cancel", headers=_auth("k-acme-operator"))
    assert denied.status_code == 403
    assert "Requires admin" in denied.json()["detail"]
    allowed = client.post(f"/runs/{run_id}/cancel", headers=_auth("k-acme-admin"))
    assert allowed.status_code == 200
    assert allowed.json()["status"] == "cancelled"


# ------------------------------------------------------------------- isolation

def test_a_run_is_only_visible_inside_its_own_tenant(client):
    acme_run = _create_run(client, "k-acme-admin", "acme work")
    globex_run = _create_run(client, "k-globex-admin", "globex work")

    acme_ids = {r["id"] for r in client.get("/runs", headers=_auth("k-acme-viewer")).json()}
    globex_ids = {r["id"] for r in client.get("/runs", headers=_auth("k-globex-viewer")).json()}

    assert acme_ids == {acme_run}
    assert globex_ids == {globex_run}
    assert acme_ids.isdisjoint(globex_ids)


@pytest.mark.parametrize("path", ["", "/results", "/metrics"])
def test_cross_tenant_read_is_404_not_403(client, path):
    """404 is the security-relevant answer, so the code is asserted exactly.

    403 would confirm the run exists. An attacker who can tell "not yours" from
    "not there" can enumerate every run id in the system, which leaks how much
    work another tenant is doing even without reading any of it.
    """
    acme_run = _create_run(client, "k-acme-admin", "confidential")
    for key in ("k-globex-admin", "k-globex-operator", "k-globex-viewer"):
        resp = client.get(f"/runs/{acme_run}{path}", headers=_auth(key))
        assert resp.status_code == 404, f"{key} got {resp.status_code} for {path}"


def test_cross_tenant_cancel_cannot_destroy_another_tenants_work(client):
    """The nastiest version of the leak: a write that reaches across tenants."""
    acme_run = _create_run(client, "k-acme-admin", "keep running")
    resp = client.post(f"/runs/{acme_run}/cancel", headers=_auth("k-globex-admin"))
    assert resp.status_code == 404
    still = client.get(f"/runs/{acme_run}", headers=_auth("k-acme-admin")).json()
    assert still["status"] != "cancelled"


def test_a_tenant_sees_its_own_run_through_every_read_route(client):
    """The mirror of the isolation test: scoping must not over-block either."""
    run_id = _create_run(client, "k-acme-operator", "mine")
    for key in ("k-acme-admin", "k-acme-operator", "k-acme-viewer"):
        assert client.get(f"/runs/{run_id}", headers=_auth(key)).status_code == 200
        assert client.get(f"/runs/{run_id}/results", headers=_auth(key)).status_code == 200
        assert client.get(f"/runs/{run_id}/metrics", headers=_auth(key)).status_code == 200


def test_runs_record_which_principal_created_them(client):
    run_id = _create_run(client, "k-acme-operator", "audited")
    body = client.get(f"/runs/{run_id}", headers=_auth("k-acme-admin")).json()
    assert body["tenant_id"] == "acme"
    assert body["created_by"] == "acme-operator"


def test_quality_reports_are_tenant_scoped_too(client):
    """Release evidence is as confidential as the runs it came from."""
    report = {
        "schema": "quality-report/v1",
        "dataset_version": "golden-acme",
        "metrics": {"groundedness": 0.95, "citation_completeness": 0.98,
                    "unsupported_claim_rate": 0.01, "entity_resolution_f1": 0.95,
                    "judge_weighted_kappa": 0.75, "source_acceptable_rate": 0.95},
        "denominators": {"claims": 120, "mentions": 340, "sources": 95,
                         "judge_ratings": 40},
        "provenance": {"evaluator_version": "v1", "golden_manifest_sha256": "abc"},
    }
    created = client.post("/quality-reports", json={"report": report},
                          headers=_auth("k-acme-admin"))
    assert created.status_code == 201
    assert created.json()["tenant_id"] == "acme"

    assert len(client.get("/quality-reports", headers=_auth("k-acme-viewer")).json()) == 1
    assert client.get("/quality-reports", headers=_auth("k-globex-viewer")).json() == []
