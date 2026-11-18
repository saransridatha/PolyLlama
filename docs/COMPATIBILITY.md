# Client Compatibility

ModelSwitch implements the OpenAI Chat Completions protocol. Any client that
accepts a custom base URL, a bearer key, and a model name should work with
`model: "auto"`.

## Compatibility matrix

| Client | Status | Notes |
|---|---|---|
| `curl` / raw HTTP | **Verified** | Non-streaming and SSE tested against the live gateway |
| OpenAI Python SDK | **Verified** | `models.list`, chat completion, and streaming path tested |
| OpenAI JavaScript SDK | Protocol-compatible; not separately executed | Same JSON/SSE protocol |
| Continue (VS Code/JetBrains) | Protocol-compatible; config provided | Use `provider: openai`, `model: auto`, `apiBase` |
| Open WebUI | Protocol-compatible; not installed on build host | Set OpenAI base URL and inference key |
| LiteLLM | Protocol-compatible; not installed on build host | Use `openai/auto` with `api_base` |
| Generic coding agents | Protocol-compatible | Prefer `model: auto` or `coding` |
| Embedding clients | Only if an embeddings-capable GGUF is registered | Chat models return `400 embeddings_not_supported` |

"Protocol-compatible" is not the same as "verified here". Only the rows marked
verified were executed as part of this build's evidence. The public contract is
the standard OpenAI JSON shape; no proprietary client SDK is required.

## Base URL

Use the public listener, including `/v1`:

```text
http://127.0.0.1:8000/v1
```

Through a Cloudflare Tunnel, replace the origin:

```text
https://llm.example.com/v1
```

Never use the admin port (`8001`) as a client base URL.

## Python (verified)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ["MODELSWITCH_API_KEY"],
)

models = client.models.list()
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain Kubernetes networking"}],
)
print(response.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Write a haiku about local inference"}],
    stream=True,
)
for event in stream:
    delta = event.choices[0].delta.content if event.choices else None
    if delta:
        print(delta, end="", flush=True)
```

## cURL

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $MODELSWITCH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }'
```

Streaming:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $MODELSWITCH_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"fast","messages":[{"role":"user","content":"Count to 5"}],
       "stream":true,"stream_options":{"include_usage":true}}'
```

## Continue

Add to `config.yaml` (path varies by IDE):

```yaml
models:
  - name: ModelSwitch Auto
    provider: openai
    model: auto
    apiBase: http://127.0.0.1:8000/v1
    apiKey: <your-inference-key>
    roles: [chat, edit, apply]
    defaultCompletionOptions:
      contextLength: 8192
```

For a coding-specialist route, use `model: coding`. Continue may send tools or
`max_tokens`; both are forwarded to llama.cpp's OpenAI server.

## Open WebUI

- Settings → Connections → OpenAI API
- Base URL: `http://127.0.0.1:8000/v1`
- API key: an inference key
- Models should appear automatically after `GET /v1/models`; select `auto`.

If the gateway runs on the same host as Open WebUI in Docker, use
`http://host.docker.internal:8000/v1` or the host LAN address, and set
`MS_PUBLIC_HOST=0.0.0.0`. Keep the admin listener on loopback.

## LiteLLM

```yaml
model_list:
  - model_name: modelswitch-auto
    litellm_params:
      model: openai/auto
      api_base: http://127.0.0.1:8000/v1
      api_key: <your-inference-key>
```

LiteLLM may call `/v1/models` and use streaming; both are implemented. If it sends
provider-specific fields, ModelSwitch forwards unknown JSON fields to llama.cpp.

## Coding agents: model choices

| Need | Recommended model argument |
|---|---|
| Let the gateway decide | `auto` |
| Fast autocomplete/chat | `fast` |
| Code generation/debugging | `coding` |
| Architecture/analysis | `reasoning` |
| Pin a known model | registered id, e.g. `qwen2.5-coder-7b` |
| Pin a team alias | configured alias, e.g. `workhorse` |

## Protocol notes and differences

- Response adds a `modelswitch` object with routing/timing metadata. Standard
  clients ignore unknown top-level fields.
- llama.cpp-only `timings` are stripped from non-streaming responses. Streaming
  chunks may include llama.cpp fields; clients ignore them.
- The backend's `system_fingerprint` is preserved.
- `max_completion_tokens` is accepted and normalized to `max_tokens`.
- `messages[].content` can be a string or an array of content parts. Vision parts
  require a vision-capable model.
- `stream_options.include_usage` is forwarded for a final usage chunk.
- `n > 1` is forwarded but backend support varies by llama.cpp build/model.
- Embeddings require an embeddings-capable GGUF; ModelSwitch does not fake
  embeddings with a chat model.
- Errors always use the OpenAI `{ "error": { ... } }` envelope.

## Authentication troubleshooting

- Use `Authorization: Bearer <key>`; `X-API-Key` is also accepted.
- A 401 means missing/revoked/expired/invalid key, or too many failed attempts.
- A 403 on an inference call means the key's model scope excludes the resolved
  model, or the key lacks streaming/embedding permission.
- A 429 includes `Retry-After`.
