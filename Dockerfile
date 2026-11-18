# ModelSwitch CPU image. GPU images should be built with the CUDA base and
# -DGGML_CUDA=ON; the native setup is recommended for accelerator tuning.
FROM ubuntu:24.04 AS llama-build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates curl libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ARG LLAMA_REF=master
RUN git clone --depth 1 --branch "${LLAMA_REF}" https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp
RUN cmake -S /opt/llama.cpp -B /opt/llama-build \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON -DGGML_CUDA=OFF \
    && cmake --build /opt/llama-build -j"$(nproc)" --target llama-server


FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MS_PUBLIC_HOST=0.0.0.0 \
    MS_ADMIN_HOST=0.0.0.0 \
    MS_DATA_DIR=/data \
    MS_MODEL_ROOTS=/models \
    MS_LLAMA_SERVER_BIN=/opt/llama/bin/llama-server

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ca-certificates libgomp1 libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app ./app
COPY scripts ./scripts
COPY --from=llama-build /opt/llama-build/bin /opt/llama/bin

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir .

RUN mkdir -p /data/logs /data/run /models /opt/llama/bin \
    && chmod +x /app/scripts/docker-entrypoint.sh /app/scripts/*.sh

ENV PATH="/opt/venv/bin:${PATH}"
VOLUME ["/data", "/models"]
EXPOSE 8000 8001

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
CMD ["serve"]
