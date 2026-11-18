#!/usr/bin/env bash
# Install cloudflared locally (vendor/cloudflared) on Linux.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

case "$(uname -s)" in
  Darwin)
    if command -v brew >/dev/null 2>&1; then
      brew install cloudflared
      exit 0
    fi
    echo "Install cloudflared with Homebrew: brew install cloudflared" >&2
    exit 1
    ;;
  Linux)
    case "$(uname -m)" in
      x86_64|amd64) ARCH=amd64 ;;
      aarch64|arm64) ARCH=arm64 ;;
      *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}"
    echo "Downloading $URL"
    curl -fL --retry 3 -o vendor/cloudflared "$URL"
    chmod +x vendor/cloudflared
    ./vendor/cloudflared --version
    ;;
  *)
    echo "On Windows use: winget install --id Cloudflare.cloudflared" >&2
    exit 1
    ;;
esac
