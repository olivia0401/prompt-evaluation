import { chromium, firefox, webkit, type BrowserType } from '@playwright/test';

/**
 * Warm each browser engine once, before any test starts.
 *
 * Why this file exists — an actual observed failure, not a precaution:
 *
 *   [webkit] API documentation renders in a real browser
 *   Error: browserContext.newPage: Target page, context or browser has been closed
 *
 * It failed exactly once, on the first run after the engines were downloaded,
 * and then passed 5/5 (3 isolated, 2 full-suite). That shape — fails cold,
 * passes warm — points at first-launch cost, not at the test. On a cold start
 * an engine unpacks and initialises, and with three engines cold-starting
 * concurrently that cost was being charged against a 30-second *test* budget
 * that was never meant to cover it.
 *
 * The tempting fix is `retries: 1`. That is not a fix: it converts a real
 * signal into a green tick, and the next cold-start failure — in CI, on a
 * slower box, on a bigger engine — hides behind the same retry. Paying the
 * cold-start cost here, once, outside every test's timeout, removes the cause.
 *
 * Cost when already warm: well under a second per engine.
 */
async function warm(name: string, engine: BrowserType): Promise<void> {
  const started = Date.now();
  const browser = await engine.launch();
  const page = await browser.newPage();   // the exact call that failed cold
  await page.close();
  await browser.close();
  console.log(`  warmed ${name} in ${Date.now() - started}ms`);
}

export default async function globalSetup(): Promise<void> {
  console.log('Warming browser engines (cold first-launch is paid here, not in a test):');
  // Sequential on purpose: warming three engines in parallel recreates the
  // resource contention this is meant to remove.
  await warm('chromium', chromium);
  await warm('firefox', firefox);
  await warm('webkit', webkit);
}
