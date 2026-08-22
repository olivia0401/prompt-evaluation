import { test, expect } from '@playwright/test';

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

test('protected run submission rejects missing bearer token when enabled', async ({ request }) => {
  test.skip(!process.env.E2E_API_TOKEN, 'Set E2E_API_TOKEN against an authenticated deployment');
  const response = await request.post('/runs', { data: { stage: 'phase0' } });
  expect(response.status()).toBe(401);
});
