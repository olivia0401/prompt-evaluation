import { test, expect } from '@playwright/test';

/**
 * Tier 1 — smoke. Runs on every push, in every engine, in a couple of seconds.
 * If these fail, nothing further is worth running.
 *
 * Tier 2 (personas.spec.ts) covers authentication and roles; tier 3
 * (tenant-isolation.spec.ts) covers cross-tenant isolation.
 */

test('health endpoint is reachable @smoke', async ({ request }) => {
  const response = await request.get('/health');
  expect(response.ok()).toBeTruthy();
  const body = await response.json();
  expect(body.status).toBe('ok');
});

test('API documentation renders in a real browser @smoke', async ({ page }) => {
  await page.goto('/docs');
  await expect(page).toHaveTitle(/FastAPI|Swagger/i);
});

test('the service is not running open @smoke', async ({ request }) => {
  // Replaces an earlier test that skipped itself unless E2E_API_TOKEN was set —
  // so on the default configuration it silently never ran, and the suite would
  // have stayed green against a service with authentication switched off
  // entirely. That is the failure mode this file exists to catch, so it is now
  // an unconditional assertion.
  const unauthenticated = await request.get('/runs');
  expect(unauthenticated.status(), 'listing runs must require a key').toBe(401);

  const whoami = await request.get('/whoami', {
    headers: { Authorization: 'Bearer k-acme-viewer' },
  });
  expect(whoami.status()).toBe(200);
  expect(
    (await whoami.json()).open_mode,
    'open_mode true means every caller is an admin — never correct for a deployed service',
  ).toBe(false);
});
