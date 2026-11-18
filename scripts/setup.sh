#!/usr/bin/env bash
# Native, non-root setup: virtualenv, dependencies, llama.cpp, dashboard, DB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.11+ is required. Set PYTHON=/path/to/python3." >&2
  exit 1
fi
if ! "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)'; then
  echo "Python 3.11+ is required; found $("$PYTHON" --version 2>&1)." >&2
  exit 1
fi

echo "==> Python environment"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[dev]"

echo "==> llama.cpp"
if [ -x "${MS_LLAMA_SERVER_BIN:-$ROOT/vendor/llama-build/bin/llama-server}" ] || command -v llama-server >/dev/null 2>&1; then
  echo "llama-server already available."
else
  scripts/build_llama.sh
fi

echo "==> Dashboard"
if [ ! -f app/static/dashboard/index.html ]; then
  if command -v npm >/dev/null 2>&1; then
    (cd frontend && npm install --no-audit --no-fund && npm run build)
  else
    echo "Dashboard build missing and npm is unavailable. The API still works; install Node 20+ and run: cd frontend && npm install && npm run build" >&2
  fi
else
  echo "Prebuilt dashboard assets present."
fi

mkdir -p data/logs data/run models

if .venv/bin/python - <<'PY'
from app.config import Settings
from app.db import Database
s = Settings()
s.ensure_dirs()
db = Database(s.db_path)
count = db.count_api_keys()
db.close()
raise SystemExit(0 if count == 0 else 1)
PY
then
  echo "==> Credentials"
  .venv/bin/python -m app init
else
  echo "==> Credentials already initialized"
fi

echo
echo "Setup complete."
echo "Start: ./scripts/start.sh"
echo "Tests: .venv/bin/pytest -q"
