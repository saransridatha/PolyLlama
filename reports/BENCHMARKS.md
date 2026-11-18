# ModelSwitch benchmark report

Generated: `2026-11-17T18:40:23.140977+00:00`

These are measured values from a live gateway. Hardware, model choice,
context size and background load materially affect results.

## Environment

- Host: Linux Fedora Linux 44 (Workstation Edition) (x86_64)
- CPU: 12th Gen Intel(R) Core(TM) i5-12500H · 12 cores / 16 threads
- RAM: 15.3 GiB
- Backend: cpu
- llama.cpp: version: 1.0.0-dev (build 12477, commit 054a1babf)
- Resident limit: 2

- GPU: NVIDIA Corporation GA107BM [GeForce RTX 3050 Mobile] (rev a1) · 0 MiB · backend cpu
- Warnings: No CUDA/Metal GPU detected; inference will use the CPU backend.

## Per-model measurements

| Model | Quant | Cold load | TTFT median | Stream tok/s | Non-stream median | RSS |
|---|---|---:|---:|---:|---:|---:|
| `qwen2.5-coder-7b` | Q4_K_M | 3199 ms | 97 ms | 11.2 | 2931 ms | 8.15 GiB |
| `qwen3.5-2b` | Q4_K_M | 1435 ms | 66 ms | 29.6 | 1129 ms | 2.05 GiB |
| `smollm3-3b` | Q4_K_M | 2493 ms | 46 ms | 23.5 | 1431 ms | 3.60 GiB |

## Concurrency

| Model | Parallel requests | Wall | Aggregate tok/s | p50 latency | Errors |
|---|---:|---:|---:|---:|---:|
| `qwen3.5-2b` | 1 | 1015 ms | 23.6 | 1005 ms | 0 |
| `qwen3.5-2b` | 2 | 1314 ms | 36.5 | 1294 ms | 0 |
| `qwen3.5-2b` | 4 | 2492 ms | 38.5 | 1813 ms | 0 |

## Model switching

Alternating between `qwen2.5-coder-7b` and `qwen3.5-2b`: median cold-switch request `4426 ms`.

## Methodology

- Streaming TTFT is measured client-side from request start to the first content delta (includes routing and any cold load).
- Stream tok/s is completion tokens divided by (total time − TTFT).
- Cold load is the admin load endpoint duration until llama.cpp reports ready.
- Concurrency uses ThreadPoolExecutor and explicit model requests; aggregate tok/s is total completion tokens divided by wall time.
- No values are synthesized. Raw samples are in the JSON report alongside this file.
