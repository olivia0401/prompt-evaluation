import { test, expect } from '@playwright/test';
import { PERSONAS, auth, canWrite, canCancel } from '../personas';

/**
 * Authentication and role enforcement, one real HTTP session per persona,
 * repeated in Chromium, Firefox and WebKit.
 *
 * Running six personas across three browsers is not about browser rendering —
 * these are API calls. It is about the request path being genuinely different
 * in each engine: header casing, redirect handling, CORS preflight and cookie
 * policy all differ, and an auth layer that works in Chromium and leaks in
 * WebKit is a real and boring class of bug.
 */

test.describe('every persona authenticates as exactly itself', () => {
  for (const persona of PERSONAS) {
    test(`${persona.name} @auth`, async ({ request }) => {
      const response = await request.get('/whoami', { headers: auth(persona) });
      expect(response.status()).toBe(200);
      expect(await response.json()).toEqual({
        name: persona.name,
        tenant: persona.tenant,
        role: persona.role,
        open_mode: false,
      });
    });
  }
});

test('health is the only public route @auth', async ({ request }) => {
  expect((await request.get('/health')).status()).toBe(200);
  for (const path of ['/whoami', '/stages', '/runs', '/quality-reports']) {
    expect((await request.get(path)).status(), `${path} must require a key`).toBe(401);
  }
});

test('a wrong or malformed key is rejected @auth', async ({ request }) => {
  const cases = [
    { Authorization: 'Bearer not-a-real-key' },
    { Authorization: 'Basic k-acme-admin' },
    { Authorization: 'k-acme-admin' },
    { Authorization: 'Bearer ' },
  ];
  for (const headers of cases) {
    const response = await request.get('/whoami', { headers });
    expect(response.status(), JSON.stringify(headers)).toBe(401);
  }
});

test.describe('roles are enforced, not just recorded', () => {
  for (const persona of PERSONAS) {
    test(`${persona.name} write permission @rbac`, async ({ request }) => {
      const response = await request.post('/runs', {
        headers: auth(persona),
        data: { stage: 'phase0', note: `rbac-${persona.name}` },
      });
      if (canWrite(persona)) {
        expect(response.status()).toBe(201);
      } else {
        // 403 and not 401: the key is valid, the role is not sufficient. A
        // viewer being told "unauthenticated" would send them to rotate a key
        // that is working perfectly well.
        expect(response.status()).toBe(403);
        expect((await response.json()).detail).toContain('cannot write');
      }
    });
  }
});

test.describe('cancelling is admin-only', () => {
  for (const persona of PERSONAS.filter(canWrite)) {
    test(`${persona.name} cancel permission @rbac`, async ({ request }) => {
      const created = await request.post('/runs', {
        headers: auth(persona),
        data: { stage: 'phase0', note: `cancel-${persona.name}` },
      });
      expect(created.status()).toBe(201);
      const runId = (await created.json()).id;

      const response = await request.post(`/runs/${runId}/cancel`, { headers: auth(persona) });
      if (canCancel(persona)) {
        expect(response.status()).toBe(200);
        expect((await response.json()).status).toBe('cancelled');
      } else {
        expect(response.status()).toBe(403);
        expect((await response.json()).detail).toContain('Requires admin');
      }
    });
  }
});

test('a created run records which principal made it @audit', async ({ request }) => {
  const persona = PERSONAS.find((p) => p.name === 'acme-operator')!;
  const created = await request.post('/runs', {
    headers: auth(persona),
    data: { stage: 'phase0', note: 'audit trail' },
  });
  const body = await created.json();
  expect(body.tenant_id).toBe('acme');
  expect(body.created_by).toBe('acme-operator');
});
