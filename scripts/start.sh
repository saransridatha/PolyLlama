#!/usr/bin/env bash
# One-command local start. The first run performs setup and prints initial keys.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up ModelSwitch…"
  scripts/setup.sh
fi

if [ ! -x "${MS_LLAMA_SERVER_BIN:-$ROOT/vendor/llama-build/bin/llama-server}" ] && ! command -v llama-server >/dev/null 2>&1; then
  echo "llama-server is missing. Building it now…"
  scripts/build_llama.sh
fi

if .venv/bin/python - <<'PY'
from app.config import Settings
from app.db import Database
s = Settings(); s.ensure_dirs(); db = Database(s.db_path)
count = db.count_api_keys(); db.close()
raise SystemExit(0 if count == 0 else 1)
PY
then
  .venv/bin/python -m app init
fi

exec .venv/bin/python -m app serve "$@"
