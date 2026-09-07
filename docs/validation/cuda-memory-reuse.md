# CUDA memory reuse — 2026-09-07

Reusing one CUDA execution stream per device reduced allocator residency after four
full Qwen3-0.6B reloads from 4.56 GiB to 1.14 GiB (75%). The fixed worker stayed at
1.14 GiB through eight cycles. Every cycle generated the expected 16 reference
tokens and retired all model/sequence reservations.

[Recorded results](cuda-memory-reuse.json) contain snapshots from each phase,
external process peaks, reference hashes, numerical checks and the regression logs.
[Milestone 5](../milestone-5.md) describes the implementation and profiler usage.
The measured native/tooling source is `848b017`, based on Milestone 4 `926a29d`.

## Method and results

The complete pinned `Qwen/Qwen3-0.6B` checkpoint and independent F32 Transformers
oracle are those in the [Milestone 4 report](full-checkpoint-cross-machine.md).
Measurements used the same Fedora/RTX 3060 Ti/LibTorch 2.13 environment, native
caching allocator, F16 execution, 33 prompt tokens plus 16 generated tokens, and
2 GiB host / 6 GiB device admission budgets. Each variant started in a fresh
worker process. No cache flush was performed between reloads. Other applications
remained running; a separate memory watcher sampled the owned worker every 0.5 s.

The baseline retained the original `CudaStage` per-load pooled-stream acquisition,
with the new telemetry enabled. The startup probe already used the shared stream,
so the comparison isolates stage reload behavior. Four cycles were run on each
variant, followed by four additional cycles on the fixed worker.

| Cycle | Baseline reserved after unload | Shared-stream reserved after unload |
| --- | ---: | ---: |
| 1 | 1,224,736,768 B | 1,222,639,616 B |
| 2 | 2,447,376,384 B | 1,222,639,616 B |
| 3 | 3,670,016,000 B | 1,222,639,616 B |
| 4 | 4,892,655,616 B | 1,222,639,616 B |
| 5–8 | Not run | 1,222,639,616 B each |

Reserved means active plus cached bytes from the allocator. Baseline active bytes
also rose from 8,519,680 to 34,078,720; fixed active bytes stayed at 8,519,680 after
unload. These remaining active bytes belong to framework workspaces, while native
model reservations were zero. The baseline's maximum sampled NVIDIA process usage
rose to 4,856 MiB; the fixed process stayed at a sampled maximum of 1,356 MiB per
cycle. Neither metric is a substitute for the other.

Host RSS was not bounded by this change: the fixed run's sampled peak reached about
2.22 GiB, including framework/host-allocator overhead beyond model admission
accounting. Larger-model fit assessment must include that physical overhead. The
configured 2 GiB host budget is not a process RSS cap.

## Verification

The revised lifecycle regression runs identical mixed assignments on fresh host
threads, in both stage orders and pageable/pinned modes, checking stability after
two warmup cycles. It failed with the per-load stream implementation and passed
with stream reuse: active, cached and RSS ranges were zero in the measured native
run. The test no longer warms the entire 32-stream pool before checking stability.

All 53 CUDA-side CTest entries passed (325.75 s), including mixed execution and
fault recovery. A final focused native/smoke run passed after adding the telemetry
backend guard, including the `cudaMallocAsync` case returning unavailable telemetry.
The 42 Python tests, Ruff, Pyright, generated bindings and whitespace checks passed.
Full-checkpoint CUDA F16 and F32 probes passed their nine sampled decisions and
numerical bounds. Both direct MLX/CUDA stage orders were also repeated with the
fixed CUDA worker and matched 256/256 reference tokens.

The measurements establish reuse for this workload, not a universal memory limit.
They do not qualify 4B, arbitrary contexts, concurrent GPU workloads or automatic
placement. Profiling timings include telemetry overhead. Numerical/transport
limitations from Milestone 4, including historical CUDA-only F16 long-continuation
drift, remain applicable.
