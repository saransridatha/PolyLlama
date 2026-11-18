# Open WebUI

Settings → Connections → OpenAI API:

- Base URL: `http://127.0.0.1:8000/v1`
- API key: a ModelSwitch inference key
- Model: `auto` (appears in the model list)

If Open WebUI runs in Docker on the same host, use
`http://host.docker.internal:8000/v1` and set `MS_PUBLIC_HOST=0.0.0.0`.
Keep the admin listener on loopback.
