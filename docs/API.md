# API Reference

Base URL defaults to `http://127.0.0.1:8000/v1`. All inference endpoints require
an API key. The admin control API lives on a separate loopback listener
(`http://127.0.0.1:8001/admin/api`) and requires an `admin` role key.

Interactive OpenAPI docs are served at the public listener's `/docs`.

## Authentication

```http
Authorization: Bearer msk_...
```

`X-API-Key: msk_...` is also accepted. Keys are created by the CLI or dashboard
and are shown once. Inference keys can be scoped to specific model IDs, denied
streaming/embeddings, and capped at a maximum output token count.

Rate limits are sliding-window requests-per-minute and tokens-per-minute per key.
Admin endpoints use the same limiter. `429` responses include `Retry-After`.

## Model identifiers

| Identifier | Behavior |
|---|---|
| `auto` | Classify the task and select the best feasible model. |
| `default` | Alias for `auto`. |
| `fast` | Lowest expected latency feasible model. |
| `coding` | Best coding/debugging model, general fallback if configured. |
| `reasoning` | Best reasoning/math model. |
| registered id | Explicit model; no auto-selection. |
| model alias | Explicit alias configured on a model. |

`GET /v1/models` returns role aliases, registered models, and model aliases.

## GET /v1/models

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer $MODELSWITCH_API_KEY"
```

```json
{
  "object": "list",
  "data": [
    {"id": "auto", "object": "model", "owned_by": "modelswitch"},
    {"id": "qwen2.5-coder-7b", "object": "model", "owned_by": "modelswitch-local",
     "context_length": 8192, "quantization": "Q4_K_M", "parameter_size": "7B"}
  ]
}
```

Only enabled, available models are listed. Paths and internal lifecycle details
are never included.

## POST /v1/chat/completions

Standard OpenAI chat request fields are accepted. Unknown fields are forwarded to
llama.cpp so template-specific options (for example `chat_template_kwargs`) work.

```json
{
  "model": "auto",
  "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "Write a Python function that reverses a string."}
  ],
  "max_tokens": 256,
  "temperature": 0.2,
  "stream": false
}
```

Response:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1790000000,
  "model": "qwen2.5-coder-7b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "def reverse(s): ..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 31, "completion_tokens": 42, "total_tokens": 73},
  "modelswitch": {
    "routing": {
      "task": "code_generation",
      "task_label": "Code generation",
      "capability": "code_generation",
      "confidence": 0.91,
      "strategy": "balanced",
      "selected_model": "qwen2.5-coder-7b",
      "explanation": "Selected …",
      "alternatives": [
        {"model": "qwen3.5-2b", "score": 45.1, "loaded": false,
         "capability_match": false, "reason": "general fallback"}
      ],
      "constraints": {
        "required_context_tokens": 351,
        "available_ram_bytes": 6311935672,
        "max_resident_models": 2,
        "excluded": []
      },
      "routing_ms": 0.9,
      "classifier_source": "heuristic",
      "fallback_used": false
    },
    "queue_ms": 0.0,
    "load_ms": 0.0,
    "wait_ms": 0.0,
    "request_id": "req_..."
  }
}
```

Response headers:

- `X-Request-Id`
- `X-ModelSwitch-Model`
- `X-ModelSwitch-Task`
- `X-ModelSwitch-Routing-Ms`
- `X-ModelSwitch-Queue-Ms`
- `X-ModelSwitch-Load-Ms`
- `X-ModelSwitch-Fallback`

### Streaming

```json
{"model":"auto","stream":true,"stream_options":{"include_usage":true}, "...": "..."}
```

The response is `text/event-stream`; events are passed through from llama.cpp. A
final `data: [DONE]` is emitted. Include `stream_options.include_usage` for the
final usage chunk. Closing the client connection cancels the upstream request and
releases the model slot.

## POST /v1/completions

Legacy text completion. Accepts `prompt` as a string, token array, or list of
prompts plus the usual sampling parameters. Supports streaming.

```json
{"model":"auto","prompt":"The capital of France is","max_tokens":16}
```

Response object is `text_completion` with `choices[].text`.

## POST /v1/embeddings

Only models advertising the `embeddings` capability are eligible. Auto selection
chooses an embeddings model; explicit chat models return `400
embeddings_not_supported`. The response is proxied from llama.cpp and tagged with
the resolved model.

```json
{"model":"auto","input":["first text","second text"]}
```

## Errors

All errors use the OpenAI envelope:

```json
{
  "error": {
    "message": "Request needs ~9000 tokens but model context is 8192",
    "type": "invalid_request_error",
    "param": null,
    "code": "context_length_exceeded"
  }
}
```

| HTTP | Typical codes |
|---|---|
| 400 | `invalid_request`, `context_length_exceeded`, `embeddings_not_supported` |
| 401 | `invalid_api_key` |
| 403 | `permission_denied`, `model_not_permitted` |
| 404 | `model_not_found` |
| 409 | `model_disabled`, `model_busy` |
| 413 | `request_too_large` |
| 429 | `rate_limit_exceeded` |
| 503 | `no_feasible_model`, `queue_full`, `queue_timeout`, `model_load_failed`, `backend_unavailable` |
| 504 | `backend_timeout` |

Validation errors never echo the submitted value, so keys or prompts are not
reflected back into logs or errors.

## Health and readiness

- `GET /healthz` — unauthenticated liveness; process and version.
- `GET /readyz` — 200 when llama-server is present and at least one enabled model
  file is available; 503 otherwise. Returns counts and resident model IDs.
- `GET /` — service descriptor.

## Admin control API

All routes require an admin key and are only served on the admin listener.

| Method | Path | Purpose |
|---|---|---|
| GET | `/admin/api/auth/me` | Validate key and return role/limits |
| GET | `/admin/api/overview` | Dashboard aggregate |
| GET | `/admin/api/system` | Hardware/process/resource snapshot |
| GET | `/admin/api/readiness` | Readiness details |
| GET | `/admin/api/models` | Registry + runtime + estimates + perf |
| GET | `/admin/api/models/discover` | Scan configured roots |
| POST | `/admin/api/models` | Register a GGUF path |
| PATCH | `/admin/api/models/{id}` | Update metadata/configuration |
| DELETE | `/admin/api/models/{id}` | Remove registration (file kept) |
| POST | `/admin/api/models/{id}/load` | Load now |
| POST | `/admin/api/models/{id}/unload` | Unload if idle |
| POST | `/admin/api/models/{id}/reload` | Force reload |
| GET | `/admin/api/models/{id}/log` | Backend log tail |
| GET | `/admin/api/models/{id}/estimate` | Memory estimate |
| GET | `/admin/api/routing` | Config, aliases, history, perf |
| PATCH | `/admin/api/routing/config` | Update strategy/classifier |
| GET | `/admin/api/routing/history` | Explainable decisions |
| GET | `/admin/api/keys` | Keys + 24h usage |
| POST | `/admin/api/keys` | Create key; plaintext returned once |
| PATCH | `/admin/api/keys/{id}` | Update limits/permissions/expiry |
| POST | `/admin/api/keys/{id}/rotate` | New key; old revoked |
| DELETE | `/admin/api/keys/{id}` | Revoke |
| GET | `/admin/api/keys/usage` | Per-key usage |
| GET | `/admin/api/remote` | Tunnel status/diagnostics |
| POST | `/admin/api/remote/start` | Quick/named tunnel |
| POST | `/admin/api/remote/stop` | Stop tunnel |
| POST | `/admin/api/remote/restart` | Restart with last config |
| POST | `/admin/api/remote/check` | Public health check |
| GET | `/admin/api/logs/requests` | Request history |
| GET | `/admin/api/logs/requests/{id}` | Full request + routing |
| GET | `/admin/api/logs/events` | Backend/lifecycle events |
| GET | `/admin/api/diagnostics` | Consolidated diagnostics |
| GET | `/admin/api/info` | URLs and docs metadata |
| GET | `/metrics` | Prometheus exposition (admin auth) |

The dashboard is served at `/admin` (SPA fallback). Admin responses never contain
plaintext keys or hashes.

## Admin request examples

Create an inference key:

```bash
curl -X POST http://127.0.0.1:8001/admin/api/keys \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"continue","role":"inference",
       "permissions":{"models":["*"],"allow_streaming":true},
       "rate_limit_rpm":60,"rate_limit_tpm":100000}'
```

Register a local model:

```bash
curl -X POST http://127.0.0.1:8001/admin/api/models \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"path":"/home/me/models/model.gguf","context_length":8192,
       "priority":70,"residency":"on_demand","max_concurrency":2}'
```

Update routing:

```bash
curl -X PATCH http://127.0.0.1:8001/admin/api/routing/config \
  -H "Authorization: Bearer $ADMIN_KEY" \
  -H "Content-Type: application/json" \
  -d '{"strategy":"balanced","llm_enabled":false,"fallback_enabled":true}'
```

## Prometheus metrics

`GET /metrics` with an admin bearer key exposes:

- `modelswitch_requests_started_total`
- `modelswitch_requests_total{endpoint,status,model}`
- `modelswitch_request_duration_seconds{model}`
- `modelswitch_ttft_seconds{model}`
- `modelswitch_tokens_total{model,type}`
- `modelswitch_routing_seconds{task}`
- `modelswitch_fallback_total{model,status}`
- `modelswitch_model_loaded{model}`
- `modelswitch_model_active_requests{model}`
- `modelswitch_model_queued_requests{model}`
- resource gauges (`modelswitch_cpu_percent`, RAM, GPU, resident estimates)
