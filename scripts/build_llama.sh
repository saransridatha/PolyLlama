#!/usr/bin/env bash
# Build llama.cpp's server into vendor/llama-build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="${LLAMA_CPP_SRC:-$ROOT/vendor/llama.cpp}"
BUILD="${LLAMA_BUILD_DIR:-$ROOT/vendor/llama-build}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ ! -d "$SRC/.git" ] && [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "llama.cpp source not found at $SRC"
  echo "Cloning ggml-org/llama.cpp (shallow)…"
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$SRC"
fi

CUDA_ARGS=()
if [ "${MS_LLAMA_CUDA:-auto}" = "1" ] || { [ "${MS_LLAMA_CUDA:-auto}" = "auto" ] && command -v nvcc >/dev/null 2>&1; }; then
  echo "CUDA toolkit detected/requested; enabling GGML_CUDA=ON"
  CUDA_ARGS=(-DGGML_CUDA=ON)
else
  echo "Building the CPU backend (set MS_LLAMA_CUDA=1 to enable CUDA)."
  CUDA_ARGS=(-DGGML_CUDA=OFF)
fi

cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_CURL=OFF \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  "${CUDA_ARGS[@]}"

cmake --build "$BUILD" --config Release -j "$JOBS" --target llama-server

SERVER="$BUILD/bin/llama-server"
if [ ! -x "$SERVER" ]; then
  echo "Build finished but $SERVER is missing." >&2
  exit 1
fi
echo
echo "llama-server ready: $SERVER"
"$SERVER" --version | head -2
