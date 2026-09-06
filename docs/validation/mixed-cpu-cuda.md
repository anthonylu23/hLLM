# Mixed CPU/CUDA qualification

## Baseline (PR A, 2026-09-05)

Linux loopback, RTX 3060 Ti (8 GB), driver 610.57.04, CUDA 13.3,
LibTorch 2.13.0+cu130, GCC 14.4, RelWithDebInfo. Tiny four-layer Llama and
two-layer Qwen3 fixtures; F32 execution/KV, FP16 wire; F32/F16/BF16 storage,
tied and untied heads. Both stage orders and every valid split match CPU/CPU
with the same FP16 boundary. Includes repeated requests, stop IDs, 256 generated
tokens, and prepare → plan → generate through the CLI.

All 46 Linux CTest entries passed (42 CPU/common cases, 13 CPU process cases,
7 CUDA numerical cases, 5 CUDA smoke cases, 4 mixed matrix cases). The macOS
CPU-only 43 CTest entries, 35 Python tests, Ruff and Pyright also passed.

Run the mixed suite with a CUDA-enabled build using
`ctest --test-dir build/cuda -R MixedPipelineIntegration --output-on-failure`.
See `tests/cuda/test_mixed.py` for the executable CLI example using
`examples/profiles/workers-cpu-cuda-local.yaml`, `links-cpu-local.yaml`,
`examples/workloads/cpu-demo.yaml`, and `planner-cpu.yaml`. Start the CPU peer on port
50051 with a 536870912-byte host cap and the CUDA peer on port 50053 with
67108864-byte host and 134217728-byte device caps. Both use the same model root.

These profiles are conservative tiny-fixture qualification estimates. This is
not a full-checkpoint fit or throughput result. Cross-machine execution remains
Milestone 4; mixed precision remains globally F32.

## Pinned staging (PR B)

All 47 Linux CTest entries passed, including eight mixed process cases (the
baseline matrix in both modes), eight CUDA numerical/admission cases and one
transfer case. Final focused native rerun also passed after adding explicit
pinned-subset reporting and cancellation cleanup assertions. CPU-only CTest,
36 Python tests, Ruff and Pyright passed.

A 100-iteration CUDA runtime round-trip probe (upload + download, including
host staging copies and event waits) measured these mean microseconds:

| Payload | Pageable | Pinned |
| --- | ---: | ---: |
| 12 bytes (tiny decode) | 8.47 | 9.95 |
| 6,144 bytes (512 × 6 FP16) | 9.30 | 11.18 |
| 65,536 bytes | 31.35 | 19.69 |

These are small transfer microbenchmarks on this machine, not generation
latency, statistically rigorous performance claims, or a reason to change the
default. Reproduce with `ctest --test-dir build/cuda -R CudaTransfers -V`.
