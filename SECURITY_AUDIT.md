# Security Audit

Date: 2026-11-18 (local build host, Asia/Calcutta)
Scope: the ModelSwitch gateway source, authentication/authorization, public and
admin listeners, model path handling, request pipeline, Cloudflare integration,
and runtime configuration.

> This is a self-audit of a locally built product, not a third-party penetration
> test. Findings and residual risks are listed so another operator can reproduce
> and challenge them.

## 1. Summary

| Area | Result |
|---|---|
| Critical findings | 0 |
| High findings | 0 |
| Medium findings at release | 0 (two Bandit SQL-construction warnings were fixed) |
| Low findings at release | 0 open Bandit issues after review/annotations |
| Python dependency audit | `pip-audit`: 67 packages, 0 known vulnerabilities |
| Node production audit | `npm audit --omit=dev`: 0 vulnerabilities |
| Auth bypass tests | Pass |
| Admin/public listener isolation | Pass (including through a real quick tunnel) |
| Path traversal / external path confinement | Pass |
| Rate-limit enforcement | Pass |
| Request-size enforcement | Pass |
| Secret leakage checks | Pass for tested paths |

Reports: `reports/bandit.json`, `reports/pip-audit.json`,
`reports/npm-audit.json`.

## 2. Architecture-level controls

- **Listener split.** The public ASGI app has no admin routes. The admin app binds
  a separate loopback port. A tunnel targets only the public port. Verified by
  request through a real Cloudflare quick tunnel: public `/admin/api/overview` →
  404.
- **Key storage.** API keys are 256-bit random values (`msk_` prefix). SQLite
  stores a deterministic SHA-256 lookup fingerprint and a salted scrypt hash
  (`N=16384, r=8, p=1`). Plaintext is returned exactly once. Revocation/expiry are
  checked in the DB on every request before the scrypt verification cache is used.
- **Roles and scopes.** `admin` and `inference` roles. Inference keys can be
  constrained to model ids, denied streaming/embeddings, and capped in output
  tokens. The last active admin key cannot be revoked.
- **Path confinement.** Model paths must be absolute-resolvable, `.gguf`,
  regular files, and inside configured model roots unless an admin explicitly
  enables external paths. Symlink targets are resolved before the root check.
- **No shell.** Backend and diagnostic subprocesses use argument arrays and fixed
  commands, never `shell=True`. The tunnel token is passed via environment, not
  argv. Admin-supplied model `extra_args` are filtered against a restricted flag
  list.
- **Backend isolation.** llama-server binds `127.0.0.1` on an ephemeral port, has
  no API key, and receives no client Authorization header. Model files are never
  served as static files.
- **Body and error hygiene.** Request bodies are capped before parsing (including
  chunked). Validation errors report field locations/messages but not the
  submitted values. A generic unhandled-error response reveals only a request ID.
- **Browser hygiene.** Dashboard is same-origin, Authorization-header-only,
  sessionStorage, no cookies. CSP blocks external script/frame content. Security
  headers are added to all responses.
- **Brute force.** Per-IP auth-failure brake (20/minute) plus per-key RPM/TPM
  limits. Admin endpoints use the key's configured limits.
- **Logs.** Structured logging redacts key-shaped strings, authorization and tunnel
  token patterns. Raw keys are never logged by application code. Raw prompts and
  bodies are not persisted.
- **Secrets at rest.** DB 0600, data dir 0700, backend/tunnel logs 0600. `.env`,
  `data/`, and key files are gitignored; `.env.example` contains placeholders.

## 3. Verification tests

Security regression tests live in:

- `tests/test_api_security.py`
- `tests/test_security.py`
- `tests/test_registry.py`
- `tests/test_admin_api.py`
- `tests/test_cloudflare.py`

Notable cases:

| Test | Expected | Result |
|---|---|---|
| Missing auth on `/v1/models` | 401 + `WWW-Authenticate` | Pass |
| Invalid key | 401 | Pass |
| Basic auth | 401 | Pass |
| Inference key on admin API | 403 | Pass |
| Admin routes on public listener | 404 | Pass |
| Admin route through real quick tunnel | 404 | Pass |
| Path `/etc/passwd` | 400 (`invalid_model`), not 500 | Pass |
| Alias collision / reserved role aliases | rejected | Pass |
| Oversized body (3 MiB > 2 MiB) | 413 | Pass |
| Invalid message role | 400, no crash | Pass |
| Per-key RPM limit | second request 429 + Retry-After | Pass |
| Admin key RPM limit | second request 429 | Pass |
| Last admin revoke | 409 | Pass |
| Raw key in captured logs | absent | Pass |
| Tunnel URL `http://` / localhost | rejected | Pass |
| Named tunnel without token | rejected | Pass |
| Invalid public URL on admin API | 400, normalized error | Pass |

The whole suite: see `RELEASE_READINESS.md`.

## 4. Findings fixed during the audit

1. **ASGI body middleware fabricated disconnects** for streaming responses,
   causing SSE to abort before the first event. Fixed by delegating subsequent
   `receive()` calls to the real transport. Re-verified with real SSE.
2. **Admin dependency referenced an undefined runtime variable** after introducing
   admin rate limiting. Fixed and covered by the admin test suite.
3. **Bandit B608 dynamic SQL strings.** Update methods still assembled `SET`
   clauses from whitelisted keys. Replaced with fixed column→fragment maps;
   annotated the final constant joins. Bandit is now clean.
4. **Alias collision.** A model alias could shadow another model's id or alias.
   Registration/update now rejects collisions with 409.
5. **Dashboard import overwrote inferred capabilities.** Importing through the UI
   sent `["chat","general"]`, which would have removed coding/reasoning metadata.
   Discovery now returns inferred capabilities and the form sends `undefined`
   when untouched, letting the backend infer.
6. **Interrupted streams recorded zero completion tokens.** The SSE parser now
   accumulates content characters and estimates usage on cancellation, while still
   preferring backend-reported usage when available.

## 5. Residual risks and operator responsibilities

- **Local TLS is not provided.** The inference API is HTTP on loopback. Do not
  bind `MS_PUBLIC_HOST=0.0.0.0` on an untrusted network without a TLS reverse proxy
  or a Cloudflare Tunnel. Admin should remain loopback.
- **Quick tunnels are public endpoints.** They require a valid API key, but the
  hostname is discoverable and Cloudflare offers no uptime guarantee. Use named
  tunnels with Access policies for production.
- **Dashboard-entered tunnel tokens are held in memory**, not persisted to disk.
  A gateway restart requires `CLOUDFLARE_TUNNEL_TOKEN`. Do not paste production
  tokens into shared browser sessions without understanding sessionStorage.
- **Session storage is readable by same-origin JavaScript.** CSP and the absence
  of third-party scripts reduce exposure. For high-assurance environments, add an
  identity-aware proxy in front of the admin listener.
- **SQLite is local and single-process.** It is not a multi-tenant auth database.
  Filesystem access to `data/modelswitch.db` is equivalent to control-plane
  compromise; protect the OS account.
- **Model files are trusted inputs.** GGUF parsers and llama.cpp process untrusted
  model metadata. Only administrators can register files, but a malicious GGUF may
  exploit a parser/model-loader bug. Treat downloaded weights as software.
- **`MS_ALLOW_EXTERNAL_MODEL_PATHS=true`** lets an admin register any `.gguf` path
  readable by the service account. It is intentionally off by default.
- **Docker image is built CPU-only** and the compose service uses host UID/GID.
  Running Docker as root without user mapping would give the container broader
  host access.
- **No MFA.** Admin auth is a bearer key. Use a secret manager, short expiry and
  rotation for remote administration.
- **Dependency freshness.** `pip-audit`/`npm audit` reflect the audit date; CI
  should re-run them.

## 6. Reproducing the audit

```bash
.venv/bin/bandit -q -r app -f json -o reports/bandit.json
.venv/bin/pip-audit --cache-dir .cache/pip-audit --format=json \
  --output reports/pip-audit.json
(cd frontend && npm audit --omit=dev --json > ../reports/npm-audit.json)
.venv/bin/pytest -q
```

Expected: Bandit 0 findings (after reviewed annotations), pip-audit 0 known
vulnerabilities, npm 0 production vulnerabilities, all tests passing.

## 7. Release conclusion

No critical or high-severity issue remains in the reviewed scope. The controls
above are implemented and exercised by automated tests or live verification.
Residual risks are deployment/operator concerns, not concealed known defects.
