# M5.1–M5.2 qualification — 2026-09-08

M5.1 profile contracts and M5.2 isolated native assignment-memory profiling are
implemented and tested on CPU, Apple MLX and NVIDIA CUDA. Both endpoint roles of
the pinned Qwen3-0.6B split completed the 512/256 workload on each accelerator, with
successful admission, cleanup and conservative physical-envelope checks. This is
memory evidence for isolated stage execution; automatic placement and the 15%
performance criterion remain unfinished.

The [summary](milestone-5-memory/summary.json) links measurements by their artifact
digests. [Source identity](milestone-5-memory/source-identity.json) records base revision
`5c205f6cdba5fa5d69d18de1cff93b320dc5d403` and the implementation snapshot hash. Each
profile independently records its native binary hash and backend/toolchain identity.

## Experiment

Use `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, F16
execution and KV, F16 boundary buffers, 512 prompt tokens, 256 output steps and
768-token sequence capacity. The split is layer 14 of 28. Two cycles run in each
fresh process: one cold assignment and one reload, without allocator cache eviction.

The four probes execute the real assigned weights with deterministic synthetic
inputs: token zero for first stages and F16 zero boundary activations for final
stages. They exercise prefill and the 255 subsequent decode steps. Pair the first
MLX profile with final CUDA for MLX→CUDA, and first CUDA with final MLX for CUDA→MLX.
These pairs establish assignment-memory coverage, not distributed generation or link
performance. Full checkpoint hashes agree across the two machines.

| Assignment | Peak allocator active | Lifetime RSS peak | Conservative host/unified envelope | Device envelope |
| --- | ---: | ---: | ---: | ---: |
| [mlx-first](milestone-5-memory/mlx-first.json) | 0.889 GiB | 1.510 GiB | 2.888 GiB | — |
| [mlx-final](milestone-5-memory/mlx-final.json) | 0.891 GiB | 1.497 GiB | 2.876 GiB | — |
| [cuda-first](milestone-5-memory/cuda-first.json) | 0.805 GiB | 2.115 GiB | 2.578 GiB | 1.363 GiB |
| [cuda-final](milestone-5-memory/cuda-final.json) | 0.805 GiB | 2.135 GiB | 2.600 GiB | 1.363 GiB |

MLX used a 2,936,012,800-byte unified admission cap (2.734375 GiB), plus 256 MiB of
explicit physical headroom, 256 MiB of extra process/transport overhead and a 10%
safety allowance. The native conservative load requirement is about 2.730 GiB for
these assignments. Higher initial probe caps were rejected before loading as desktop
and background-indexing memory use changed. Existing applications remained running.

CUDA used 2 GiB host and 2 GiB device admission caps, with 1 GiB host / 512 MiB device
physical headroom, 256 MiB extra overhead per physical domain and a 10% safety allowance.
The Fedora desktop and existing Java server remained running; no competing inference
was launched. CUDA allocator and process-GPU observations remain separate from host
RSS and logical reservations.

MLX's envelope deliberately overcounts RSS plus allocator residency because their
physical overlap is unknown. It is an upper-bound fit check, not an estimate of actual
system usage. All physical observations retain sampling/coverage limitations; see the
[accounting conventions](../milestone-5-profiling.md). Fit reports alongside each
artifact preserve the actual pre-launch availability and allowances.

All four probes retired their model/sequence objects and completed both cycles.
CUDA retained reusable allocator cache after unload, while active allocator bytes
returned to framework workspace levels. Two full-checkpoint cycles do not establish
an arbitrary-length RSS/cache plateau; the native lifecycle regressions and prior
[eight-cycle CUDA evidence](cuda-memory-reuse.md) remain separate checks.

## Larger target preflight

The [4B assessment](milestone-5-memory/qwen3-4b-preflight.json) uses the existing
metadata-only shape fixture for `Qwen/Qwen3-4B-Base` revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`. No 4B checkpoint payload was loaded.

The vocabulary/hidden dimensions are 151,936 × 2,560 and embeddings are tied. In either
two-stage order, MLX owns an embedding copy: input embedding when first, or tied LM
head when final. The current MLX loader's conservative admission requirement includes
its F16 resident embedding, BF16 source payload, three F32 conversion/copy buffers
and fixed loader scratch. These endpoint terms alone total **5.796 GiB**, excluding
all transformer layer weights and any tied-head verification scratch.

The Mac availability observation was **4.812 GiB**, before subtracting headroom or
adding safety/transport allowances. Thus all 70 two-worker candidates fail this
current conservative cold-load preflight on their MLX side. This is not a measured
4B peak or proof that the checkpoint cannot fit on an otherwise idle machine.
Physical profiling and inference qualification for 4B remain pending better headroom
or a separately implemented and measured reduction in loader requirements. No 0.6B
memory result was extrapolated to establish 4B fit.

## Verification

- All 67 Python tests pass, including artifact round trips/tamper detection, device,
  dtype, ownership and shape compatibility, nonfinite timings, missing telemetry,
  incomplete measurements, physical/admission separation and saved preflight rejection.
- Ruff, Pyright, generated-protobuf consistency and whitespace checks pass.
- All 53 local CPU/MLX CTest entries passed before the profiler phase-peak normalization;
  the two profiling CTest entries passed again after that isolated harness change.
  This includes both native numerical suites and CPU/MLX process integration.
- CUDA numerical parity and CPU/CUDA profiling CTest entries passed on the Fedora
  machine. Profiling covers both roles, three-cycle tiny-model reloads, cleanup,
  allocation failure and `cudaMallocAsync` returning unavailable allocator telemetry.
- The four published full-checkpoint profiles were rerun after phase-peak normalization
  and the conservative MLX physical-envelope adjustment.

The runtime serving protocol and its lifetime telemetry semantics are unchanged.
M5.3–M5.6 still need native timing, directional transport measurement, measured planner
integration and independent exhaustive split comparison.
