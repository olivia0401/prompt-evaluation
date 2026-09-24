# Browser suite — 6 personas, 3 engines, real multi-tenant coverage

```bash
npm ci
npx playwright install chromium firefox webkit
npx playwright test                    # 87 tests, ~10s
npx playwright test --grep @smoke      # tier 1 only
npx playwright test --project=webkit   # one engine
```

No setup beyond that: the config starts its own API on a throwaway SQLite
database with the six personas configured, so there is nothing to run first.

## Tiers

| Tier | File | What it proves |
|---|---|---|
| 1 · `@smoke` | `service-smoke.spec.ts` | The service is up, `/docs` renders in a real browser, and authentication is actually switched on. |
| 2 · `@auth` `@rbac` | `personas.spec.ts` | Each of the six personas authenticates as exactly itself; roles are enforced, not merely recorded. |
| 3 · `@isolation` | `tenant-isolation.spec.ts` | Two tenants working in one database, neither able to read or touch the other's runs or release evidence. |

## The personas

Two tenants × three roles, defined once in [`personas.ts`](personas.ts) and
reused by the API (`SERVICE_PRINCIPALS`) and by the fast pytest layer
(`tests/test_tenancy.py`), so a persona cannot drift between the three.

| Persona | Tenant | Role | Read | Create | Cancel |
|---|---|---|---|---|---|
| `acme-admin` | acme | admin | ✓ | ✓ | ✓ |
| `acme-operator` | acme | operator | ✓ | ✓ | — |
| `acme-viewer` | acme | viewer | ✓ | — | — |
| `globex-admin` | globex | admin | ✓ | ✓ | ✓ |
| `globex-operator` | globex | operator | ✓ | ✓ | — |
| `globex-viewer` | globex | viewer | ✓ | — | — |

## Three decisions worth reading

**404, not 403, across the tenant boundary.** A 403 confirms the resource
exists, so an outsider can enumerate run ids and learn how much work another
tenant is doing without reading any of it. Every cross-tenant assertion checks
the exact code — a test written as `expect([403, 404]).toContain(status)` would
stay green while that leak was live. Within a tenant the opposite holds: a
viewer denied a write gets a loud 403, because they already know the resource
exists and "unauthenticated" would send them to rotate a working key.

**`retries: 0`, even in CI.** Retrying until green converts a real signal into a
green tick. The suite has one documented flake and it was fixed at the root
rather than retried — see below.

**The suite never spends money.** It POSTs real runs against a real server, but
the server runs with `DISABLE_RUN_EXECUTION=1`: the full API surface is live
(create, list, scope, cancel) and no LLM provider is ever contacted. Without
that switch, `npx playwright test` would cost money on every run, need provider
keys in CI, and mark runs FAILED when those keys were absent — making the cancel
tests fail for a reason that has nothing to do with cancelling.

## The one flake, and why it was not retried

On the first run after the engines were downloaded:

```
[webkit] API documentation renders in a real browser
Error: browserContext.newPage: Target page, context or browser has been closed
```

It then passed 5/5 — three isolated runs, two full-suite runs. Fails cold,
passes warm: that shape points at first-launch cost, not at the test. A cold
engine unpacks and initialises, and with three engines cold-starting
concurrently that cost was being charged against a 30-second *test* budget that
was never meant to cover it.

`retries: 1` would have hidden it. [`global-setup.ts`](global-setup.ts) instead
launches each engine once, sequentially, before any test runs, so the
cold-start cost is paid outside every test's timeout; `workers` is capped so
three engines never stampede at once. Warm, the setup costs well under a second
per engine.

## Running against a deployed environment

```bash
E2E_BASE_URL=https://staging.example.com \
E2E_PRINCIPALS='[{"key":"...","name":"acme-admin","tenant":"acme","role":"admin"}]' \
npx playwright test
```

Setting `E2E_BASE_URL` disables the built-in server. Real persona keys live in
staging secrets and are injected through `E2E_PRINCIPALS` — never committed, and
never the same keys as the fixtures in `personas.ts`.
