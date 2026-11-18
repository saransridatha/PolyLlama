#!/usr/bin/env bash
# Container entrypoint. Initialization is explicit so credentials are never
# written into container logs automatically.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  set -- serve
fi

case "$1" in
  serve|init|doctor|keys|models)
    if [ "$1" = "serve" ]; then
      exec python -m app serve
    fi
    exec python -m app "$@"
    ;;
  python|/opt/venv/bin/python)
    exec "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
