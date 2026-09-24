/**
 * The six authenticated personas the browser suite runs as.
 *
 * Two tenants x three roles. This is the same roster the fast pytest layer uses
 * (tests/test_tenancy.py) and the same JSON the API reads from
 * SERVICE_PRINCIPALS, so a persona cannot drift between the three places.
 *
 * The keys here are test fixtures, not secrets: they only ever authenticate a
 * throwaway local server started by playwright.config.ts. Credentials for a
 * real staging deployment belong in staging secrets and are injected through
 * E2E_PRINCIPALS — never committed.
 */
export type Role = 'admin' | 'operator' | 'viewer';

export interface Persona {
  key: string;
  name: string;
  tenant: string;
  role: Role;
}

export const PERSONAS: Persona[] = [
  { key: 'k-acme-admin', name: 'acme-admin', tenant: 'acme', role: 'admin' },
  { key: 'k-acme-operator', name: 'acme-operator', tenant: 'acme', role: 'operator' },
  { key: 'k-acme-viewer', name: 'acme-viewer', tenant: 'acme', role: 'viewer' },
  { key: 'k-globex-admin', name: 'globex-admin', tenant: 'globex', role: 'admin' },
  { key: 'k-globex-operator', name: 'globex-operator', tenant: 'globex', role: 'operator' },
  { key: 'k-globex-viewer', name: 'globex-viewer', tenant: 'globex', role: 'viewer' },
];

export const byName = (name: string): Persona => {
  const found = PERSONAS.find((p) => p.name === name);
  if (!found) throw new Error(`No persona named ${name}`);
  return found;
};

export const inTenant = (tenant: string): Persona[] =>
  PERSONAS.filter((p) => p.tenant === tenant);

export const notInTenant = (tenant: string): Persona[] =>
  PERSONAS.filter((p) => p.tenant !== tenant);

export const canWrite = (p: Persona): boolean => p.role === 'admin' || p.role === 'operator';
export const canCancel = (p: Persona): boolean => p.role === 'admin';

export const auth = (p: Persona) => ({ Authorization: `Bearer ${p.key}` });
