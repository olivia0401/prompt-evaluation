"""Authentication, tenancy and roles for the evaluation service.

An evaluation service is a multi-tenant problem whether or not anyone planned
for it: runs cost real money, results contain client-confidential briefs, and
quality reports are release evidence. "Everyone with the shared token sees
everything" is not a security model, it is the absence of one.

Model
-----
A **principal** is one API key. It carries a tenant and a role:

* ``admin``    — read + write + cancel, within its own tenant.
* ``operator`` — read + create runs, within its own tenant. Cannot cancel.
* ``viewer``   — read only, within its own tenant.

There is deliberately no cross-tenant role. Isolation is enforced in the data
layer (every query is filtered by ``tenant_id``), not by remembering to check in
each route — a check you have to remember is a check you will eventually forget.

Two decisions worth stating explicitly
--------------------------------------
**Cross-tenant access returns 404, not 403.** A 403 confirms the resource
exists, which is itself a leak: an attacker enumerating run ids learns which
ones are real. To a principal from another tenant, the row simply does not
exist. This is why the isolation tests assert 404 and would fail on a 403.

**Keys are compared in constant time** and looked up by digest, so the lookup
does not leak key material through timing.

Configuration
-------------
Set ``SERVICE_PRINCIPALS`` to a JSON array::

    [{"key": "...", "name": "acme-admin", "tenant": "acme", "role": "admin"}]

When it is unset the service falls back to the legacy single ``API_TOKEN``
(admin on the ``default`` tenant), and when that is empty too it runs open —
one implicit admin principal on the ``default`` tenant, which is what local dev
and the unit tests want.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import dataclass

DEFAULT_TENANT = "default"

ROLE_ADMIN = "admin"
ROLE_OPERATOR = "operator"
ROLE_VIEWER = "viewer"
ROLES = (ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER)

# What each role may do. Reads are implicit — every authenticated principal can
# read within its own tenant.
WRITE_ROLES = frozenset({ROLE_ADMIN, ROLE_OPERATOR})
CANCEL_ROLES = frozenset({ROLE_ADMIN})


@dataclass(frozen=True)
class Principal:
    name: str
    tenant_id: str
    role: str

    @property
    def may_write(self) -> bool:
        return self.role in WRITE_ROLES

    @property
    def may_cancel(self) -> bool:
        return self.role in CANCEL_ROLES

    def to_dict(self) -> dict:
        return {"name": self.name, "tenant": self.tenant_id, "role": self.role}


ANONYMOUS_ADMIN = Principal(name="anonymous", tenant_id=DEFAULT_TENANT, role=ROLE_ADMIN)


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class PrincipalRegistry:
    """Key digest -> Principal, plus the open-mode flag."""

    def __init__(self, principals: dict[str, Principal] | None = None):
        self._by_digest: dict[str, Principal] = principals or {}

    @property
    def open_mode(self) -> bool:
        """True when no keys are configured: every request is the anon admin."""
        return not self._by_digest

    @property
    def principals(self) -> list[Principal]:
        return sorted(self._by_digest.values(), key=lambda p: (p.tenant_id, p.name))

    def resolve(self, presented_key: str) -> Principal | None:
        """Constant-time-compared lookup of one presented bearer token."""
        if not presented_key:
            return None
        candidate = self._by_digest.get(_digest(presented_key))
        if candidate is None:
            # Burn a comparison anyway so a miss and a hit cost the same.
            secrets.compare_digest(_digest(presented_key), _digest("unused"))
            return None
        return candidate

    @classmethod
    def from_env(cls, raw_principals: str | None, legacy_token: str | None) -> "PrincipalRegistry":
        by_digest: dict[str, Principal] = {}
        if raw_principals:
            try:
                entries = json.loads(raw_principals)
            except json.JSONDecodeError as exc:
                raise ValueError(f"SERVICE_PRINCIPALS is not valid JSON: {exc}") from exc
            if not isinstance(entries, list):
                raise ValueError("SERVICE_PRINCIPALS must be a JSON array of objects")
            for entry in entries:
                key = str(entry.get("key") or "")
                if not key:
                    raise ValueError("every principal needs a non-empty 'key'")
                role = str(entry.get("role") or ROLE_VIEWER)
                if role not in ROLES:
                    raise ValueError(f"unknown role '{role}'; expected one of {ROLES}")
                principal = Principal(
                    name=str(entry.get("name") or "unnamed"),
                    tenant_id=str(entry.get("tenant") or DEFAULT_TENANT),
                    role=role,
                )
                digest = _digest(key)
                if digest in by_digest:
                    raise ValueError(
                        f"duplicate key: '{principal.name}' reuses the key of "
                        f"'{by_digest[digest].name}' — two identities sharing a key "
                        f"makes the audit trail meaningless"
                    )
                by_digest[digest] = principal
        elif legacy_token:
            by_digest[_digest(legacy_token)] = Principal(
                name="api-token", tenant_id=DEFAULT_TENANT, role=ROLE_ADMIN
            )
        return cls(by_digest)


def load_registry() -> PrincipalRegistry:
    from . import settings

    return PrincipalRegistry.from_env(
        os.getenv("SERVICE_PRINCIPALS", "").strip() or None,
        settings.API_TOKEN or None,
    )
