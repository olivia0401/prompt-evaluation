# Playwright environments

The harness runs the same smoke suite in Chromium, Firefox and WebKit:

```powershell
cd e2e
npm install
npx playwright install --with-deps
$env:E2E_BASE_URL = "http://localhost:8000"
npm test
```

Set `E2E_API_TOKEN` to exercise the authenticated API boundary. Add real
persona/tenant credentials as environment secrets in the deployment-specific
test suite; they are intentionally not committed to this repository.

The smoke tests are deliberately small. Business-critical tenant isolation,
role permissions and destructive workflows should be added as authenticated
tests against a seeded staging environment, not against production.
