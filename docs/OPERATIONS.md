# Operations and Installation

## 1. Supported platforms

| Platform | Native install | Inference backend | Verification on build host |
|---|---|---|---|
| Linux x86_64 | `scripts/setup.sh` | CPU (verified), CUDA (implemented) | CPU verified |
| macOS arm64/x86_64 | `scripts/setup.sh` | Metal / CPU | Not executed on macOS hardware |
| Windows x86_64 | `scripts/setup.ps1` | CPU / CUDA | Not executed on Windows hardware |
| Docker Linux | `docker compose` | CPU | Not built on build host (Docker unavailable) |

The shell scripts are shared by Linux and macOS. GPU support depends on how
llama.cpp is compiled on the host. The installer builds CPU by default and
enables CUDA when `nvcc` is present or `MS_LLAMA_CUDA=1`.

## 2. Prerequisites

- Python **3.11+** (3.12 recommended)
- CMake 3.20+ and a C++17 compiler for a native llama.cpp build
- 4 GB+ RAM for small models; 8 GB+ for 3–4B Q4 models; more for 7B+
- Optional: CUDA toolkit + NVIDIA driver, Apple Metal, or a prebuilt
  `llama-server` on `PATH`
- Node 20+ only if rebuilding the dashboard from source

No cloud account is required. Cloudflare is optional.

## 3. Linux / macOS native install

```bash
cd modelSwitch
./scripts/setup.sh
./scripts/start.sh
```

`setup.sh` creates `.venv`, installs the project, builds `llama.cpp` into
`vendor/llama-build`, builds the dashboard if assets are missing, initializes the
database, and prints the bootstrap admin/inference keys exactly once.

To use an existing llama.cpp build:

```bash
export MS_LLAMA_SERVER_BIN=/opt/llama.cpp/build/bin/llama-server
```

To rebuild with CUDA:

```bash
MS_LLAMA_CUDA=1 ./scripts/build_llama.sh
```

`llama-server` and its shared libraries live in the same `bin` directory; the
gateway adds that directory to `LD_LIBRARY_PATH` automatically.

### One-command startup

```bash
./scripts/start.sh
```

If the venv, llama.cpp, keys, or dashboard are missing, the script performs the
required setup. It prints credentials only on the first initialization.

## 4. Windows native install

Install:

- Python 3.11+ from python.org (check "Add to PATH")
- CMake
- Visual Studio Build Tools with the "Desktop development with C++" workload
- Git
- Node.js (only for dashboard rebuilds)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup.ps1
.\scripts\start.ps1
```

If the bundled llama.cpp build fails, install a prebuilt `llama-server.exe` and
set:

```powershell
$env:MS_LLAMA_SERVER_BIN = "C:\path\to\llama-server.exe"
```

CUDA builds: install the CUDA toolkit and set `$env:MS_LLAMA_CUDA="1"` before
running `scripts\build_llama.ps1`.

## 5. macOS notes

- Apple Silicon uses Metal automatically when llama.cpp is built with Metal
  support; the default CMake build on macOS does this.
- If `cmake` is missing: `brew install cmake`.
- Install cloudflared with `brew install cloudflared`.
- For memory pressure, lower `MS_DEFAULT_CONTEXT_LENGTH` and
  `MS_MAX_RESIDENT_MODELS=1`.

## 6. Docker (optional)

```bash
mkdir -p models data
docker compose build
docker compose run --rm modelswitch init   # prints keys once; store them
docker compose up -d
```

- `./models` is mounted read-only at `/models`.
- `./data` persists the SQLite DB, logs, and run metadata.
- Both ports are published on host loopback only.
- The container runs as `${UID}:${GID}` so `./data` remains owned by you.
- The image builds llama.cpp CPU on Ubuntu 24.04.
- Cloudflare Tunnel is not containerized; run the connector on the host pointing
  at `http://127.0.0.1:8000`, or add your own connector container with the same
  public-only target.

Health check:

```bash
docker compose ps
curl http://127.0.0.1:8000/healthz
```

## 7. Configuration

Copy `.env.example` to `.env` and edit. All app settings use `MS_`; the Cloudflare
token uses `CLOUDFLARE_TUNNEL_TOKEN` (or `MS_CLOUDFLARE_TUNNEL_TOKEN`).

Important settings:

| Variable | Default | Meaning |
|---|---|---|
| `MS_PUBLIC_HOST` / `MS_PUBLIC_PORT` | `127.0.0.1` / `8000` | inference listener |
| `MS_ADMIN_HOST` / `MS_ADMIN_PORT` | `127.0.0.1` / `8001` | dashboard/control listener |
| `MS_DATA_DIR` | `./data` | SQLite and logs |
| `MS_MODEL_ROOTS` | `~/models,./models,~/Downloads` | allowed model paths |
| `MS_ALLOW_EXTERNAL_MODEL_PATHS` | `false` | admin may register any `.gguf` |
| `MS_LLAMA_SERVER_BIN` | auto | llama-server path |
| `MS_MAX_RESIDENT_MODELS` | `2` | loaded model budget |
| `MS_DEFAULT_CONTEXT_LENGTH` | `8192` | per-request context |
| `MS_MAX_CONTEXT_LENGTH` | `32768` | registry ceiling |
| `MS_MODEL_CONCURRENCY` | `2` | llama.cpp slots per model |
| `MS_BACKEND_IDLE_TIMEOUT` | `900` | seconds before idle unload |
| `MS_MEMORY_SAFETY_FACTOR` | `0.82` | fraction of available memory usable |
| `MS_MIN_FREE_RAM_MB` | `1024` | reserve never allocated to models |
| `MS_QUEUE_TIMEOUT` | `120` | max seconds waiting for a slot |
| `MS_REQUEST_TIMEOUT` | `300` | backend inference timeout |
| `MS_MAX_REQUEST_BYTES` | `2097152` | request body cap |
| `MS_DEFAULT_RATE_LIMIT_RPM` | `120` | new inference key default |
| `MS_DEFAULT_RATE_LIMIT_TPM` | `200000` | new inference key default |
| `MS_ROUTER_STRATEGY` | `balanced` | balanced/quality/latency/local |
| `MS_ROUTER_LLM_ENABLED` | `false` | optional LLM arbitration |
| `MS_ROUTER_LLM_MODEL` | empty | decision model; empty = any resident |

Bootstrap keys are **not** read from `.env`; they are generated once by `init`.
The DB stores only hashes. Keep the printed keys in a secret manager.

### Changing the key pepper

`MS_KEY_PEPPER` is optional. If set, key lookup is bound to it. Changing it
invalidates all existing keys. Set it before first initialization for
deployment-specific hardening, and back it up with the database.

## 8. Model management

Scan configured roots:

```bash
.venv/bin/python -m app models scan
```

Register:

```bash
.venv/bin/python -m app models add \
  --path /models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf \
  --name "Qwen2.5 Coder 7B" \
  --context 8192 --priority 80 --residency on_demand --concurrency 2
```

List:

```bash
.venv/bin/python -m app models list
```

The dashboard supports discovery, registration, metadata/capability editing,
enable/disable, load/unload/reload, memory estimates, logs, and deletion of
registration (never deletion of the GGUF file).

### Choosing context and slots

Memory grows with `context × slots`. A model configured with 8192 tokens and 2
slots requests 16384 total KV positions. Start with 4096 and 1–2 slots on small
machines; increase only after observing memory.

### Keep-loaded vs on-demand

- `on_demand` (default): unload after idle timeout; preferred victim under
  pressure.
- `keep_loaded`: survive idle cleanup; still evictable if all resident capacity is
  exhausted and the model is idle. Never evicted while serving.

### GPU layer planning

`MS_LLAMA_N_GPU_LAYERS=-1` means automatic. Per-model `n_gpu_layers` overrides it.
On a 4 GB GPU, a 7B Q4 model may fit only a subset of layers; llama.cpp happily
runs the rest on CPU. On CPU-only machines all layers stay on CPU and the router
prefers smaller models.

## 9. API keys

```bash
.venv/bin/python -m app keys list
.venv/bin/python -m app keys create --name continue --role inference \
  --rpm 120 --tpm 200000 --models qwen3.5-2b,fast
.venv/bin/python -m app keys rotate <key-id>
.venv/bin/python -m app keys revoke <key-id>
```

The CLI prints a new key once. The dashboard can also create, rotate, revoke, and
apply model scopes, streaming/embedding permissions, token caps and expiry. The
last active admin key cannot be revoked.

## 10. Service operation

Health and readiness:

```bash
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/readyz
```

Structured logs go to stdout (JSON by default) and backend logs to
`data/logs/llama-<model>.log`. The dashboard's Logs page shows request history,
routing traces, lifecycle events, and backend failures.

Graceful shutdown:

- send SIGTERM/SIGINT to the `serve` process;
- uvicorn stops accepting, waits up to 15s for active streams;
- `AppRuntime.close()` stops the tunnel, all llama-server processes, HTTP clients,
  and SQLite.

### systemd

A template unit is provided:

```bash
sudo cp deploy/modelswitch@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now modelswitch@$USER
journalctl -u modelswitch@$USER -f
```

Adjust paths for your installation. The unit protects home as read-only except
the data/runtime directories; add external model roots if needed.

## 11. Remote access

See [CLOUDFLARE.md](CLOUDFLARE.md). Short version:

```bash
scripts/install_cloudflared.sh
# dashboard → Remote access → Quick tunnel (no account/domain)
# or named tunnel with CLOUDFLARE_TUNNEL_TOKEN
```

The public base URL is `https://<hostname>/v1`. Keep admin on loopback.

## 12. Backup, upgrade, reset

- Back up `data/modelswitch.db` (WAL mode: stop the service or use
  `sqlite3 data/modelswitch.db ".backup backup.db"`).
- Model files are not managed by the gateway; back them up separately.
- To upgrade: replace the repository, reinstall the venv if dependencies changed,
  rebuild llama.cpp if desired, then restart. Database migrations run at startup.
- To reset credentials: stop the service, delete `data/modelswitch.db`, run
  `python -m app init`, then re-register models. This invalidates all keys and
  request history.

## 13. Benchmarking

See [BENCHMARKS](../reports/BENCHMARKS.md) and:

```bash
MS_ADMIN_KEY=... MS_INFERENCE_KEY=... \
  .venv/bin/python scripts/benchmark.py --models qwen3.5-2b,smollm3-3b
```

The script unloads/reloads models and measures real cold-load, TTFT, tokens/s,
concurrency, and switching behavior. Run it on an idle machine for comparable
numbers.

## 14. Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
