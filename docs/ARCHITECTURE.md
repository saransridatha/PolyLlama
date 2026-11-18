# Architecture

ModelSwitch is a local control plane in front of one or more `llama-server`
processes. It owns authentication, routing, admission control, lifecycle,
observability, and the dashboard. llama.cpp owns inference.

## 1. Process and listener topology

```
                       ┌──────────────────────────────────────────────┐
                       │              ModelSwitch process             │
                       │                                              │
  OpenAI client ──────►│ :8000 public ASGI app                        │
  Coding agent         │   BodyLimit → RequestContext → auth          │
  Open WebUI           │   /v1/chat/completions, /v1/models, /healthz │
                       │                    │                         │
                       │                    ▼                         │
                       │        InferenceService                      │
                       │   routing → lifecycle → llama.cpp proxy      │
                       │            │                │                │
  Browser ────────────►│ :8001 admin ASGI app     │                │
  (loopback only)      │   React SPA + control API│                │
                       │                          ▼                │
                       │        llama-server #1 :ephemeral loopback  │
                       │        llama-server #2 :ephemeral loopback  │
                       └──────────────────────────────────────────────┘
                                          ▲
                                          │ optional outbound tunnel
                              Cloudflare edge / quick tunnel
```

Two uvicorn servers run in one asyncio process. They share `AppRuntime`, so model
state, rate limits, metrics, and SQLite are coherent. The admin app is bound to a
separate port and is not mounted on the public app. The Cloudflare connector is
hard-wired to `http://127.0.0.1:<public-port>`.

Why this shape:

- A single process avoids a control-plane RPC layer and duplicate state.
- Separate ASGI apps make administrative exposure structural, not a routing
  convention that can regress.
- One process per model isolates crashes and allows per-model context, GPU layer,
  slot count, and thread configuration. llama.cpp's multi-model router would still
  run everything in one address space; a crash or OOM would take down all models.

## 2. Request lifecycle

1. **Body limit middleware** buffers up to `MS_MAX_REQUEST_BYTES` (2 MiB default),
   including chunked bodies without Content-Length.
2. **Request context middleware** assigns/accepts a request ID, sets a contextvar,
   and adds security + correlation headers to every response.
3. **Authentication** reads `Authorization: Bearer` (or `X-API-Key`). The key is
   looked up by a deterministic SHA-256 fingerprint; a salted scrypt hash is
   verified once and cached for five minutes. Revocation and expiry are still
   checked on every request against SQLite, so revocation is immediate.
4. **Rate limit** uses a per-key sliding 60-second window for RPM and TPM. Failed
   auth attempts are separately throttled per client IP. Admin keys are rate-limited
   too.
5. **Validation** enforces OpenAI request shapes, message roles, body sizes, and
   token limits.
6. **Classification** runs the deterministic classifier. If the operator enabled
   LLM arbitration and confidence is ambiguous, a resident small model gets one
   short, bounded JSON-classification call; failure always falls back to heuristics.
7. **Policy** scores enabled, available, permitted models. Models that cannot fit
   the required context or memory are excluded. The result includes a ranked
   candidate list for fallback.
8. **Admission / load** acquires the per-model gate. If the model is cold, the
   lifecycle manager makes room (evicting idle models only), starts llama-server,
   and waits for `/health`. Concurrent loads are serialized.
9. **Proxy** sends the request to the loopback backend with `model=<resolved id>`.
   The client's Authorization header is not forwarded. Streaming responses are
   passed through as SSE while usage/timings are parsed for accounting.
10. **Normalization** fills missing OpenAI fields, strips backend-only `timings`,
    attaches authenticated routing metadata, and records usage.
11. **Persistence/metrics**: request row, routing row, model EWMA performance,
    Prometheus counters/histograms, structured logs.
12. **Fallback**: for auto/role aliases, a load failure or a pre-stream 5xx from the
    backend is retried on the next feasible candidate, up to three distinct models.
    Explicit model requests never silently switch models.

## 3. Router

### Layer 1: deterministic classifier

`app/router/classifier.py` extracts:

- code fences, code-language keywords, generation verbs
- error/traceback/debugging language
- math expressions/vocabulary
- summarize, extract, translate, creative, classify, JSON output
- reasoning/analysis vocabulary
- whole-document/long-context signals
- tool declarations, image parts, system prompts, response format
- estimated tokens and language/script

It returns scores, top category, confidence, signals, complexity, and an
ambiguity flag. This runs in well under a millisecond after metadata warmup.

Categories: `general`, `reasoning`, `coding`, `code_generation`, `debugging`,
`math`, `summarization`, `extraction`, `translation`, `creative_writing`,
`structured_json`, `long_context`, `classification`, `vision`, `embeddings`.

### Layer 2: optional LLM arbitration

Only when enabled, only when Layer 1 is ambiguous, and only if a suitable model is
already resident. It is never loaded just to classify. Timeout is bounded
(`MS_ROUTER_LLM_TIMEOUT_MS`, default 1500 ms). Failure falls back to Layer 1.
The response records `classifier_source: "heuristic" | "llm"`.

### Layer 3: policy scoring

For each candidate:

- **Capability fit**: exact task capability, related coding/reasoning capabilities,
  general fallback penalty.
- **Quality**: parameter count, quantization penalty, priority, specialist bonus.
- **Latency**: measured EWMA TTFT/TPS from SQLite if available, otherwise a
  hardware-aware estimate; cold-load penalty; queue penalty.
- **Resource fit**: estimated RAM/VRAM versus live free memory plus idle-evictable
  memory.
- **Residency**: loaded bonus, avoid cold loads.
- **Queue**: per-model in-use/waiting depth.
- **Priority**: operator metadata 0–100.
- **Context**: hard reject if the model cannot hold prompt + max output + margin.

Strategies:

| Strategy | Emphasis |
|---|---|
| `balanced` (default) | task + quality + latency + memory |
| `quality` | capability and model size |
| `latency` | measured/warm latency, queue |
| `local` | small resident models, avoid cold loads |

Role aliases override the score weights: `fast` heavily weights latency,
`coding`/`reasoning` require or strongly prefer the matching capability. If no
specialist exists and `allow_general_fallback` is true, the best general model is
used with a penalty and an explanation.

The decision includes alternatives, scores, constraints, required tokens,
resident models, exclusions and reasons, routing latency, fallback state, and the
classifier signals. This is returned to authenticated clients and stored in
`routing_history`.

## 4. Model lifecycle and memory safety

State machine: `unloaded → loading → loaded → unloading → unloaded`, plus `failed`
and `crashed`.

- **One lifecycle lock** serializes model start/stop operations. This prevents two
  simultaneous requests from spawning duplicate llama-server processes and
  prevents eviction/load races.
- **Memory estimate** = model file size (minus GPU-offloaded weights) + KV cache
  + max(256 MiB, 7.5% weights) runtime overhead. KV estimate:
  `2 (K/V) × n_layers × n_ctx × n_kv_heads × head_dim × 2 bytes × slots × 1.20`.
  If metadata is incomplete the estimate is conservative.
- **Usable memory** = live available RAM × safety factor − minimum free reserve.
  VRAM uses GPU free memory and the offload planner.
- **GPU layers** are selected by `SystemMonitor.plan_offload` using free VRAM,
  per-layer size, and a 512 MiB reserve. Explicit per-model `n_gpu_layers`
  overrides auto planning. No accelerator → `0` (CPU).
- **Context slots**: llama-server receives total context = per-request context ×
  parallel slots, so every slot retains the configured request context. Memory
  estimation includes all slots.
- **Eviction**: when resident count or memory is over budget, idle models are
  evicted oldest-first, preferring `on_demand` over `keep_loaded`. Models with
  active requests or queued waiters are never evicted. If nothing can be evicted,
  the request gets `503` with a concrete memory/residency reason.
- **Idle unload**: every `MS_IDLE_CHECK_INTERVAL` seconds, idle `on_demand` models
  past `MS_BACKEND_IDLE_TIMEOUT` are unloaded. `keep_loaded` is exempt unless
  capacity pressure forces eviction.
- **Crash isolation**: maintenance detects dead processes, marks the runtime
  `crashed`, logs the event, and drops the gate. The next request reloads. Active
  streams error cleanly; the gateway stays up.

## 5. Concurrency and fairness

`PriorityGate` (per model) is a bounded async queue:

- Immediate acquisition while slots are free.
- Waiters ordered by priority, then fewest active requests for that key, then FIFO.
  Admin requests get priority 10; inference 5.
- Queue size is bounded; overflow returns `503 queue_full`.
- Wait timeout returns `503 queue_timeout`.
- Cancellation removes the waiter; releases are idempotent.
- A lease is held for the entire stream; client disconnect cancels the generator
  and releases the slot.

Admission is per model. Memory-wide admission is handled by residency limits and
eviction, not a global request semaphore. This preserves throughput while avoiding
unbounded model starts.

## 6. Persistence

SQLite in WAL mode, single guarded connection, short synchronous queries on the
event loop. Tables:

- `api_keys` — lookup hash, scrypt hash, role, permissions, limits, lifecycle.
- `models` — registry and routing metadata, never weights.
- `requests` — request ID, model, task, status, timings, tokens, routing JSON.
- `routing_history` — explainable decisions.
- `backend_events` — load/unload/eviction/crash/tunnel/key events.
- `settings` — router policy JSON.
- `model_perf` — EWMA TTFT/TPS/sample counts.

Retention is configurable and purged hourly. The DB file is chmod 0600; the data
directory is 0700.

## 7. Security boundaries

- Public app contains only inference/health/docs. No admin routes.
- Admin app contains control API + static SPA. It is bound to `MS_ADMIN_HOST`
  (loopback default). Cloudflare targets the public port only.
- Model paths are resolved and confined to `MS_MODEL_ROOTS`; `.gguf` extension
  and regular-file checks; GGUF header validation; admin-only.
- Plaintext keys are never persisted. They are returned once by init/create/rotate.
- No raw request bodies or prompts are persisted; logs contain IDs, models, tasks,
  timings, and errors.
- Authorization headers are stripped before backend proxying. Backend has no API
  key and listens on loopback only.
- CSP blocks framing and external scripts on the dashboard. Security headers are
  added to all responses.

## 8. Cloudflare

`CloudflareTunnel` supports:

- **Quick tunnel**: no account or domain; ephemeral `*.trycloudflare.com`; parsed
  from connector logs; not for production.
- **Named tunnel**: connector token supplied via `CLOUDFLARE_TUNNEL_TOKEN` or the
  dashboard (kept in memory). Cloudflare account + domain on Cloudflare + named
  tunnel + public hostname route to `http://127.0.0.1:8000`.

The connector is always started with the public port target. Ingress rules for
named tunnels must not route `/admin`; the gateway itself also keeps admin on a
separate listener, so a tunnel pointed at the public port cannot reach it.

Health is checked by fetching `<public_url>/healthz`. Process death, missing URL,
and edge errors are surfaced in the dashboard with the connector log tail
(redacted).

## 9. Extension points

- `app/backend/base.py` defines the backend protocol. `LlamaBackend` is the only
  shipping implementation; another backend must expose start/stop/health and a
  loopback OpenAI-compatible base URL.
- Router policy is isolated in `app/router/router.py`; new strategies can reuse
  the `Candidate`/`RoutingDecision` structures.
- The public API is a thin adapter over `InferenceService`; adding an endpoint
  should not duplicate lifecycle logic.
- The dashboard polls admin JSON; no websockets are required.

## 10. Known constraints

- llama.cpp owns tokenization; ModelSwitch estimates token counts for admission.
  Very unusual tokenizers may require a larger safety margin.
- SQLite is appropriate for a single-process local gateway, not a multi-node
  cluster. Running two gateway processes against one database is unsupported.
- A GPU runtime must be validated on the actual driver/toolkit; the estimator is
  intentionally conservative.
- Quick tunnels are best-effort and ephemeral by Cloudflare's design.
