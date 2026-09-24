import { test, expect } from '@playwright/test';
import { byName, inTenant, notInTenant, auth } from '../personas';

/**
 * Real multi-tenant coverage: two tenants creating work in the same database
 * at the same time, and neither able to see or touch the other's.
 *
 * The assertion that carries the weight is **404, not 403**. 403 confirms the
 * run exists, which turns run ids into an enumeration oracle — another tenant
 * can learn how much work you are doing without reading a single result. A test
 * written as `expect([403, 404]).toContain(status)` would pass while that leak
 * was live, so every check here asserts the exact code.
 */

const acmeAdmin = byName('acme-admin');
const globexAdmin = byName('globex-admin');

async function createRun(request: any, persona: any, note: string): Promise<string> {
  const response = await request.post('/runs', { headers: auth(persona), data: { stage: 'phase0', note } });
  expect(response.status(), `${persona.name} should be able to create a run`).toBe(201);
  return (await response.json()).id;
}

test('each tenant lists only its own runs @isolation', async ({ request }) => {
  const acmeRun = await createRun(request, acmeAdmin, 'acme confidential');
  const globexRun = await createRun(request, globexAdmin, 'globex confidential');

  for (const persona of inTenant('acme')) {
    const ids = (await (await request.get('/runs', { headers: auth(persona) })).json())
      .map((r: any) => r.id);
    expect(ids, `${persona.name} must see acme's run`).toContain(acmeRun);
    expect(ids, `${persona.name} must NOT see globex's run`).not.toContain(globexRun);
  }
  for (const persona of inTenant('globex')) {
    const ids = (await (await request.get('/runs', { headers: auth(persona) })).json())
      .map((r: any) => r.id);
    expect(ids, `${persona.name} must see globex's run`).toContain(globexRun);
    expect(ids, `${persona.name} must NOT see acme's run`).not.toContain(acmeRun);
  }
});

test.describe('a run from another tenant does not exist', () => {
  for (const suffix of ['', '/results', '/metrics']) {
    test(`GET /runs/{id}${suffix} is 404 across the boundary @isolation`, async ({ request }) => {
      const acmeRun = await createRun(request, acmeAdmin, `cross-read${suffix}`);
      for (const outsider of notInTenant('acme')) {
        const response = await request.get(`/runs/${acmeRun}${suffix}`, { headers: auth(outsider) });
        expect(
          response.status(),
          `${outsider.name} must get 404 (a 403 would confirm the run exists)`,
        ).toBe(404);
      }
      // And the owning tenant still reads it perfectly well — scoping must not
      // over-block, which is the failure mode you only catch by asserting both.
      for (const insider of inTenant('acme')) {
        const response = await request.get(`/runs/${acmeRun}${suffix}`, { headers: auth(insider) });
        expect(response.status(), `${insider.name} must still see its own run`).toBe(200);
      }
    });
  }
});

test('a cross-tenant cancel cannot destroy another tenant\'s work @isolation', async ({ request }) => {
  const acmeRun = await createRun(request, acmeAdmin, 'must keep running');

  const attempt = await request.post(`/runs/${acmeRun}/cancel`, { headers: auth(globexAdmin) });
  expect(attempt.status()).toBe(404);

  // The important half: confirm nothing actually changed. A route can return
  // 404 and still have performed the write.
  const after = await (await request.get(`/runs/${acmeRun}`, { headers: auth(acmeAdmin) })).json();
  expect(after.status).not.toBe('cancelled');
});

test('quality reports are tenant-scoped too @isolation', async ({ request }) => {
  const report = {
    schema: 'quality-report/v1',
    dataset_version: 'golden-acme-e2e',
    metrics: {
      groundedness: 0.95,
      citation_completeness: 0.98,
      unsupported_claim_rate: 0.01,
      entity_resolution_f1: 0.95,
      judge_weighted_kappa: 0.75,
      source_acceptable_rate: 0.95,
    },
    denominators: { claims: 120, mentions: 340, sources: 95, judge_ratings: 40 },
    provenance: { evaluator_version: 'e2e', golden_manifest_sha256: 'abc' },
  };

  const created = await request.post('/quality-reports', {
    headers: auth(acmeAdmin),
    data: { report },
  });
  expect(created.status()).toBe(201);
  expect((await created.json()).tenant_id).toBe('acme');

  const acmeVersions = (await (await request.get('/quality-reports', { headers: auth(byName('acme-viewer')) })).json())
    .map((r: any) => r.dataset_version);
  expect(acmeVersions).toContain('golden-acme-e2e');

  const globexVersions = (await (await request.get('/quality-reports', { headers: auth(byName('globex-viewer')) })).json())
    .map((r: any) => r.dataset_version);
  expect(globexVersions, 'release evidence is as confidential as the runs behind it')
    .not.toContain('golden-acme-e2e');
});

test('two tenants working concurrently stay separated @isolation', async ({ request }) => {
  // Interleaved rather than sequential: a scoping bug that only shows up under
  // concurrent writes is exactly the kind a serial test walks straight past.
  const created = await Promise.all([
    createRun(request, acmeAdmin, 'concurrent-acme-1'),
    createRun(request, globexAdmin, 'concurrent-globex-1'),
    createRun(request, acmeAdmin, 'concurrent-acme-2'),
    createRun(request, globexAdmin, 'concurrent-globex-2'),
  ]);
  const [acme1, globex1, acme2, globex2] = created;

  const acmeIds = (await (await request.get('/runs', { headers: auth(acmeAdmin) })).json())
    .map((r: any) => r.id);
  expect(acmeIds).toEqual(expect.arrayContaining([acme1, acme2]));
  expect(acmeIds).not.toContain(globex1);
  expect(acmeIds).not.toContain(globex2);
});
