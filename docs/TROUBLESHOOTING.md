# Troubleshooting

Start with:

```bash
.venv/bin/python -m app doctor
```

It checks Python, data directory, database, API keys, models, llama.cpp and
cloudflared, then prints hardware and a recommended profile.

## Startup

### "ModelSwitch is not initialized: no API keys exist"

Run:

```bash
.venv/bin/python -m app init
```

The keys are printed once. If you lost them, delete `data/modelswitch.db` only if
you accept losing all keys, model registrations, and history; then re-run init and
re-register models.

### Port already in use

```bash
ss -ltnp | grep -E '8000|8001'      # Linux
lsof -nP -iTCP:8000 -sTCP:LISTEN    # macOS
```

Set `MS_PUBLIC_PORT` / `MS_ADMIN_PORT` or pass `serve --public-port ... --admin-port ...`.

### "llama-server binary not found"

```bash
./scripts/build_llama.sh
# or
export MS_LLAMA_SERVER_BIN=/path/to/llama-server
.venv/bin/python -m app doctor
```

### Dashboard assets missing / 503 page

```bash
cd frontend
npm install
npm run build
```

The API works without the dashboard.

## Model registration and loading

### "Model path is outside the configured model roots"

Move the GGUF into a root from `MS_MODEL_ROOTS`, or add its directory:

```bash
export MS_MODEL_ROOTS="$MS_MODEL_ROOTS:/data/models"
```

`MS_ALLOW_EXTERNAL_MODEL_PATHS=true` is an admin-only escape hatch; prefer roots.

### "Invalid GGUF model"

The file header could not be parsed. Common causes: partial download, wrong file
type, or a file that is not GGUF. Re-download and verify the size. ModelSwitch
does not decode tensors during registration, so a file with a valid header can
still fail to load if tensor metadata is corrupt; check the backend log in the
Models page or `data/logs/llama-<id>.log`.

### Model fails to load / exits immediately

- Open the model log in the dashboard.
- Confirm architecture support: `llama-server --version` and upstream llama.cpp
  release notes.
- Look for "unknown model architecture", "out of memory", or missing CUDA runtime.
- Try `n_gpu_layers=0` to isolate GPU issues.
- Lower context or concurrency; KV cache grows with both.
- Check free disk (mmap and logs need headroom).

### "insufficient memory" / "resident model limit reached"

ModelSwitch estimates weights + KV + overhead and refuses to overcommit. Options:

- lower `context_length` and `max_concurrency` for that model;
- lower `MS_DEFAULT_CONTEXT_LENGTH`;
- unload another model;
- raise `MS_MAX_RESIDENT_MODELS` only if physical memory allows;
- use a smaller quantization.

The router excludes infeasible models; it will not blindly choose the largest.

### Model unload returns 409

The model has active or queued requests. This is intentional: active models are
never silently unloaded. Wait for inference, cancel clients, or use
`/reload` for an explicit force reload.

### A model produces garbage after loading

That is a model/file issue, not routing. Quarantine it by disabling it in the
dashboard. During validation on the build host, one local
`Qwen2.5-Coder-3B-Instruct-Q3_K_M` file produced incoherent tokens while the 2B and
7B files were healthy; it was disabled with a note.

## Requests and routing

### 400 context_length_exceeded

Request prompt + max output + margin exceeds the model's configured context.
Reduce `max_tokens`, shorten the prompt, register a larger-context model, or
increase the model's context if hardware permits.

### 503 no_feasible_model

All candidates were excluded. Check:

- model is enabled and its file exists;
- context fits;
- enough RAM/VRAM is free;
- API key model scopes permit a model;
- no resident model is pinning the budget (unload it).

The error or the routing history in the dashboard includes concrete reasons.

### 503 queue_timeout / queue_full

All model slots are busy. Reduce request rate, raise `max_concurrency` (uses more
KV memory), lower `MS_QUEUE_TIMEOUT`, or add another model/instance. Per-key RPM
limits also exist.

### The router chooses a smaller model than expected

That is often correct. Capability, context, memory, queue and cold-load cost are
scored. To prefer quality, set strategy `quality`, give the specialist a higher
priority, or use an explicit model id. Check the explanation and alternatives in
the response or Routing history.

### Routing feels slow on the first request

The first routing decision after startup parses GGUF headers. ModelSwitch warms
the metadata cache during startup; if you add many large models, the first
decision may still spend a few hundred milliseconds reading headers. Subsequent
decisions are typically sub-millisecond.

## Streaming and clients

### OpenAI client gets 404 at `/v1/models`

Use the base URL including `/v1`, and ensure the target is the **public** port
(8000), not the admin port (8001).

### Streaming stops immediately

Check that your proxy does not buffer SSE. ModelSwitch already sends
`Cache-Control: no-cache` and `X-Accel-Buffering: no`. For nginx, disable proxy
buffering. Cloudflare quick tunnels relay SSE normally.

### Client disconnect leaves a cancelled request

A closed client connection is recorded as `cancelled` (HTTP 499 internally). The
slot is released in the streaming generator's `finally` path. If you see a stuck
`running` request, check the backend log and restart the gateway; this is a bug.

### CORS errors from a browser app

Set explicit origins:

```bash
MS_CORS_ORIGINS=https://app.example,http://localhost:3000
```

Do not use wildcard origins with credentials. ModelSwitch does not use cookies.

## Keys and security

### 401 after rotating a key

Rotation revokes the old key immediately. Update every client with the new
plaintext. If the key is stored in a secret manager, update it there first.

### Locked out of the dashboard

Admin keys are required. If the only admin key is lost:

```bash
.venv/bin/python -m app keys create --name recovery-admin --role admin
```

This CLI runs locally with filesystem access; protect the host account. If no keys
exist, `python -m app init` creates bootstrap keys.

### "Too many failed authentication attempts"

Wait 60 seconds. This is a per-IP brake against key brute force, separate from
per-key RPM/TPM.

## Cloudflare

See [CLOUDFLARE.md](CLOUDFLARE.md). Quick checklist:

- `cloudflared --version`
- admin → Remote access shows connector running and health check;
- local `/healthz` works;
- named tunnel public hostname points to `http://127.0.0.1:8000`;
- admin URL through the tunnel must return 404 by design.

## Hardware acceleration

### CUDA requested but CPU is used

- `nvidia-smi` must work (driver loaded).
- llama.cpp must be built with CUDA (`MS_LLAMA_CUDA=1 ./scripts/build_llama.sh`).
- check `llama-server --version` / `--list-devices`;
- set `MS_LLAMA_N_GPU_LAYERS=99` or per-model `n_gpu_layers` to force offload.
- The gateway reduces GPU layers if free VRAM is insufficient; 0 means CPU.

### Apple Metal

Build llama.cpp on the Mac; use a model with supported Metal; expect unified
memory to count toward both RAM and GPU pressure. Start with one resident model.

### GPU not detected but `lspci` shows one

A missing/crashed driver is common. The build host itself had an RTX 3050 Mobile
visible in `lspci` but `nvidia-smi` failed, so the gateway correctly used CPU.
Install/repair the vendor driver and reboot, then `python -m app doctor`.

## Logs

- Service: stdout / `journalctl -u modelswitch@$USER`
- Backend: `data/logs/llama-<model>.log`
- Tunnel: `data/logs/cloudflared.log`
- Request history: dashboard Logs, or `GET /admin/api/logs/requests`
- Lifecycle events: dashboard Logs → Backend events

If reporting a bug, include the request ID, model id, hardware backend, and the
relevant log tail. Never include raw API keys or tunnel tokens.
