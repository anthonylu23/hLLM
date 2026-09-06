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

## Profile reconciliation

The CLI profile uses a 512-token cache capacity, hidden size 6, four attention
heads of dimension 4, two KV heads, intermediate width 10, vocabulary 11,
F32 compute and F32 KV. Runtime reservation at that capacity is:

| Component | Bytes |
| --- | ---: |
| CUDA device workspace (including dense attention) | 26,935,648 |
| CUDA KV per assigned layer | 32,768 |
| Pageable host workspace | 92,160 |
| Additional pinned staging | 6,144 |

The configured 33,554,432-byte CUDA workspace exceeds this estimate. The device
budget is 128 MiB with a 16 MiB reserve and 10% safety margin; host is 64 MiB
with an 8 MiB reserve and 10% margin. The pinned profile additionally caps pinned
staging at 8 MiB. Its conservative 8 MiB host transport allowance is checked
against both host and pinned budgets. Runtime admission remains authoritative.
These figures exclude framework/context overhead from the model-owned report;
physical memory headroom must be assessed separately before full checkpoints.

## Failure and memory results (PR C, 2026-09-06)

The Linux suite passed all 48 CTest entries: 42 CPU/common native cases,
13 CPU process cases, 10 CUDA numerical/admission/lifecycle cases, three
transfer cases, eight mixed parity cases, 68 mixed fault/memory cases, and
five CUDA worker cases. macOS passed 43 CPU-only CTest entries; 36 Python
unit tests, Ruff and Pyright passed. Final focused native tests and a 12-case
process rerun (deadlines and RSS) passed after strengthening admission and
transfer-failure checks.

The relay matrix covers both stage orders and transfer modes, with faults at
prefill and decode boundaries: client/control cancellation, deadline expiration,
disconnect, and loss of either worker. Tests require the expected RPC status,
no false terminal completion or replay, reservation cleanup on surviving peers,
and successful reload/generation after recoverable faults. Active unload rejects.
Additional mixed cases cover malformed/stale stage messages, idle deadlines,
partial-load rollback and downstream resource rejection.

The matrix found a generation deadline race: forcing transport cancellation
could replace `DEADLINE_EXCEEDED` with `CANCELLED`. The watchdog now allows the
handler to return after cancelling the peer, retaining a 100 ms fallback for
blocked writes. See the execution notes for synchronous-read and kernel limits.

Native lifecycle measurements use a three-token prompt, eight generated tokens,
16-token reserved capacity, and tiny Qwen3. Warm-up explicitly covers LibTorch's
32-stream low-priority pool before measuring 12 equivalent load/request/unload
cycles per mode/order. A three-cycle warm-up had incorrectly depended on earlier
tests initializing that pool; running the test alone exposed and corrected it.

Across the final ordinary native run, post-warm-up allocated and reserved device
memory ranges were both **zero bytes**. Actual pinned bytes returned to zero
on every sequence retirement. The active tiny GPU weight-plus-KV payload was
3,240–3,264 bytes and pinned staging was 192 bytes when enabled. After unload,
LibTorch still reported **260 MiB allocated and 706 MiB reserved**; native test
process RSS was about 1.27 GiB. Allocation traces identify persistent cuBLAS
workspaces and the caching allocator. These are framework allocations, distinct
from the released model/request payload, and must be included in physical fit
assessment. The measured native RSS range was at most 8 KiB after warm-up.

Separate native-worker process measurements ran 12 load/request/unload cycles,
with three warm-up cycles. Both workers' model reports returned to zero after
unload. Final CPU RSS was 24.5–24.9 MB and CUDA worker RSS was 1.019–1.029 GB;
post-warm-up RSS ranges were at most 339,968 bytes. The test allows a 32 MiB
range to accommodate platform variability. This RSS check is separate from the
native allocator test, which warms the complete stream pool. Finite plateau
checks on these workloads do not prove bounded memory for arbitrary models.

Reproduce the fault suite with `ctest --test-dir build/cuda -R
MixedFailureQualification --output-on-failure`, the native measurements with
`ctest --test-dir build/cuda -R CudaNumericalParity -V`, and process RSS with
both worker binary environment variables set and `uv run pytest
tests/cuda/test_failures.py -k worker_rss_plateau -q -s`.

### CUDA memory checking

Compute Sanitizer 2026.2.0 ran on the RTX 3060 Ti:

- `--tool memcheck --leak-check full --error-exitcode 99` on
  `hllm_cuda_transfer_tests`: all three cases passed, zero errors, zero leaked
  bytes. This includes pending-copy destruction after injected event-recording
  failure and allocation rollback.
- `--tool memcheck --leak-check no --error-exitcode 99` on
  `hllm_cuda_tests --gtest_filter='CudaLifecycleTest.*'`: both cases passed,
  zero memory-access errors, including the isolated 32-stream warm-up/plateau
  test. Instrumented RSS ranges remained below the native test's 8 MiB bound.
- An initial lifecycle run with full leak checking reported 748,815,373 bytes
  in 71 retained allocations, with LibTorch caching-allocator/cuBLAS allocation
  traces. It also exposed the insufficient warm-up described above. The final
  lifecycle result is **not a claim of whole-process leak-check cleanliness**;
  its retained framework allocations are explicitly measured and reported.

No racecheck, initcheck, full RPC sanitizer run, full-checkpoint inference,
or cross-machine qualification is claimed. The next step is a physical-memory
fit assessment; the 64 MiB host and 128 MiB device reservation caps are not
process/container memory limits.

## PR review validation (2026-09-06)

The independent [review of PRs 5–9](../code-quality-review-cuda.md) found and fixed
large pinned-boundary transfers and combined host-stage/transport accounting.
The new large-transfer regression failed on the original CUDA implementation,
then passed after copying through bounded chunks. Final allocation checks confirm
that 8 MiB + 1 byte and 16 MiB + 17 byte transfers retain only 8 MiB pinned staging,
that small transfers still work afterward, and that retirement releases staging.

A fresh CPU-only macOS build passed all 43 CTest entries (42 native cases plus
13 CPU process cases). All 37 Python unit tests, Ruff, Pyright, generated-binding
reproducibility, and whitespace checks passed. A fresh Linux CUDA-enabled build
on the RTX 3060 Ti passed all 48 CTest entries in 325 seconds: 42 common/CPU native
cases, 13 CPU process cases, four CUDA transfer cases, 10 CUDA numerical/lifecycle
cases, eight mixed pipeline cases, 68 mixed fault/memory cases, and five CUDA
worker cases. The deadline tests now keep the client alive five seconds beyond
the application deadline, so their asserted status comes from the native server.

The Linux build reused the documented GCC 14/CUDA/LibTorch dependencies in an
isolated source/build directory. Its packaged NVIDIA dependency libraries were
added to the build/test environment's library search path; no shared workload
or existing checkout was modified. Full-checkpoint and cross-machine limits
remain as described above.

Compute Sanitizer memcheck with full leak checking passed all four final transfer
cases: **zero errors and zero leaked bytes**. This run includes the new multi-chunk
payload case and the existing pending-copy/event-failure cleanup regression.
