import { defineConfig, devices } from '@playwright/test';
import { PERSONAS } from './personas';

/**
 * Tiered browser suite.
 *
 * By default this starts its own API on a throwaway SQLite database with the
 * six personas configured, so `npx playwright test` is one command with no
 * setup. Point E2E_BASE_URL at a deployed environment to run the same specs
 * against it — supply that environment's own persona keys through
 * E2E_PRINCIPALS and never commit them.
 *
 * `retries` is 0 even in CI, deliberately. Retrying a failed test until it
 * passes converts a flake into a green tick and destroys the signal the suite
 * exists to provide. A test that fails intermittently gets quarantined and
 * root-caused, not retried.
 */
const externalTarget = Boolean(process.env.E2E_BASE_URL);
const baseURL = process.env.E2E_BASE_URL || 'http://127.0.0.1:8765';
const principals = process.env.E2E_PRINCIPALS || JSON.stringify(PERSONAS);

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  expect: { timeout: 5_000 },
  fullyParallel: false, // shared database: tenant listings must be deterministic
  forbidOnly: !!process.env.CI,
  retries: 0,
  // Cold engine start is paid once, outside any test's timeout. See the
  // observed WebKit failure documented in global-setup.ts.
  globalSetup: './global-setup.ts',
  // Three engines cold-starting at once was the contention behind that
  // failure. Four workers keeps the suite fast without stampeding launches.
  workers: process.env.CI ? 2 : 4,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    extraHTTPHeaders: { Accept: 'application/json' },
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
  ],
  ...(externalTarget
    ? {}
    : {
        webServer: {
          command: 'python -m uvicorn service.api:app --host 127.0.0.1 --port 8765',
          cwd: '..',
          url: `${baseURL}/health`,
          reuseExistingServer: !process.env.CI,
          timeout: 60_000,
          env: {
            SERVICE_PRINCIPALS: principals,
            API_TOKEN: '',
            INLINE_JOBS: '1',
            // The suite POSTs real runs. This keeps the whole API surface live
            // while guaranteeing no provider is ever contacted, so the browser
            // suite costs nothing and needs no LLM keys.
            DISABLE_RUN_EXECUTION: '1',
            // A dedicated database, separate from the dev data/service.db. It
            // is not wiped between local runs, so every assertion is written
            // against runs created in the same test, never against totals.
            DATABASE_URL: 'sqlite:///./data/e2e.db',
            PYTHONUTF8: '1',
          },
        },
      }),
});
