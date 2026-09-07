# MLX qualification — 2026-09-06

Validated on Apple M3 Pro, 18 GiB unified memory, macOS 26.5.1, AppleClang 17,
with the official MLX/MLX-Metal 0.32.2 wheel's C++ headers and dynamic libraries.
The native project used the Debug configuration, with builds limited to two jobs.
Tests use tiny deterministic models and local loopback transport. Existing applications
were left running; these results do not establish performance under an idle machine.

## PR refresh — 2026-09-07

The Milestone 3 branch was rebased onto `main` at
`a643dc1` (the runtime/controller review fixes). MLX now uses the common typed
runtime errors for invalid requests, incompatible weights/metadata, and exhausted
capacity. Native tests assert those categories. This refresh excludes the later
full-checkpoint probes, redundant tied-head loader change, and cross-machine report.

Fresh checks on the scoped PR tree passed:

| Check | Result |
| --- | --- |
| Mac MLX-enabled CTest, including CPU and MLX process suites | 48/48 entries, 237.11 seconds |
| Linux CUDA-enabled CTest, including shared failure contracts | 51/51 entries, 324.20 seconds |
| Python unit/property tests | 42 passed |
| Ruff, Pyright, generated bindings, diff whitespace | Passed |
| CPU executable MLX/Torch/CUDA dependency isolation | Passed |

The Mac refresh used Python 3.13; native MLX remained 0.32.2. Linux used the existing
RTX 3060 Ti / CUDA 13.3 / LibTorch 2.13 toolchain in an isolated source/build directory.
These fresh counts include tests added by the runtime review on `main`. The initial
Milestone 3 measurements below remain useful historical evidence; no fresh sanitizer
run or separate Mac CPU-only build is claimed by this refresh.

## Initial results — 2026-09-06

| Check | Result |
| --- | --- |
| MLX-enabled CTest | 46/46 entries passed, 180.49 seconds |
| Native CPU/common cases in that build | 43 passed |
| CPU real-process cases | 13 passed |
| Native MLX numerical/lifecycle cases | 11 passed |
| MLX-only and CPU/MLX process cases | 41 passed, 145.47 seconds |
| Separate CPU-only build and CTest | 44/44 entries passed |
| Python unit/property tests | 38 passed |
| Ruff, Pyright, generated-binding reproducibility, diff whitespace | Passed |

CTest groups multiple pytest or GoogleTest cases into some entries; entry counts and
individual case counts are deliberately shown separately. The CPU executable built
alongside MLX has no MLX, Torch, or CUDA dynamic-library dependency.

The wheel CMake package emits link warnings for its producer's unavailable Xcode SDK
framework-search path and for a dylib built for macOS 26.2 versus the local compiler's
26.0 deployment target. Linking, the real Metal startup probe, and every hardware test
passed on macOS 26.5.1. Older macOS deployments were not tested. Use an appropriate
MLX source installation when targeting another SDK or deployment version.

## Numerical and execution coverage

- Independent Transformers Qwen3 fixture: prefill and incremental decode layer outputs,
  per-layer KV caches, and last-row logits match at absolute tolerance `3e-5` for F32
  and `5e-3` for F16.
- Llama layer outputs match the CPU reference at the same dtype-specific tolerances.
- Llama and Qwen3 generated tokens match CPU for F32/F16/BF16 checkpoint storage,
  tied/untied heads, and F32/F16 MLX execution. Sampling ties select token zero when
  all logits are equal.
- Direct split stages match CPU FP16 boundaries within `5e-3` and match sampled tokens.
- Real CPU/MLX processes match CPU/CPU in both stage orders and every valid split of
  the two-layer Qwen3 and four-layer Llama fixtures. The matrix covers storage formats,
  tied heads, repeated requests, stop IDs, 256 generated tokens, and the actual
  prepare → plan → generate CLI flow.
- MLX-only RPC generation matches unsplit CPU in F32 and F16 for both model families,
  including explicit reservation, cancellation, repeated load/generate/unload cycles,
  unified-domain reporting, and allocator telemetry.

These tolerances apply to these tiny fixtures. They are not acceptance thresholds
for arbitrary checkpoints, contexts, or model families. CPU/MLX mixed plans use F32
execution/KV throughout; only the transport boundary is FP16.

## Failure and admission coverage

A test-only gRPC relay holds prefill or decode at the stage boundary, after both workers
have reserved state. The 24-case fault matrix covers client/control cancellation,
application deadlines, disconnection, and loss of either worker in both CPU/MLX orders.
Assertions require the expected status, no false completion or replay, survivor cleanup,
and successful fresh generation after recoverable failures. Active unload is rejected.

Eight additional shared-contract cases cover malformed/stale stage messages, idle
cancellation/deadlines, partial-load rollback, and downstream admission rejection in
both worker orders. The relay implementation was extracted from the CUDA suite into
`tests/process_contracts.py`; both backend suites now invoke the same contract.

Native tests reject unsupported metadata/revisions, invalid context/positions, foreign
or poisoned sequence state, malformed and non-finite boundaries, non-finite weights,
F16 weight overflow, and insufficient load/sequence budgets. Common-runtime tests verify
unified exact-fit admission and one-byte-short rejection before sequence allocation,
overflow rejection, and totals that count each byte once. Planner tests check combined
stage and transport admission for both host and unified memory.

The private completion-wrapper test submits real MLX asynchronous work, then injects
known allocator exception messages, checking resource-exhaustion translation and
subsequent stream usability. Unrelated execution and input errors preserve their class.
This is controlled exception-path coverage, not induced physical OOM or a fatal Metal
fault. No fault-injection switches are exposed through worker arguments or RPCs.

## Memory observations

The native allocator test performs 12 equivalent full-stage load/generate/unload cycles
per dtype, using a three-token prompt, eight generated tokens and 16-token cache
capacity. It discards three warm-up cycles and samples nine steady-state cycles.
For both F32 and F16 in the final native run:

| Post-unload MLX allocator measurement | Bytes |
| --- | ---: |
| Active | 0 |
| Cached | 20,446 |
| Active range after warm-up | 0 |
| Cached range after warm-up | 0 |

The separate RPC suite also runs 12 cycles for each Llama/Qwen3 × F32/F16 combination.
Model-owned reports return to zero after unload. After three warm-up cycles, allocator
active memory is stable and cached-memory range stays within the test's 1 MiB bound.
These are framework allocator observations, not OS RSS, hard process limits, or proof
of bounded memory for arbitrary workloads. Transport/framework metadata and macOS
headroom remain outside the model payload report.

## Reproduction and limits

See [Milestone 3 build instructions](../milestone-3.md). With the MLX-enabled build:

```bash
uv run ctest --test-dir build/native/mlx --output-on-failure
uv run ctest --test-dir build/native/mlx -R MlxNumericalParity -V
HLLM_CPU_WORKER="$PWD/build/native/mlx/cpp/hllm-worker-cpu" \
HLLM_MLX_WORKER="$PWD/build/native/mlx/cpp/hllm-worker-mlx" \
  uv run pytest tests/mlx -q -s
```

These tests do not claim full-checkpoint inference, optimized attention or throughput,
MLX/CUDA cross-machine execution, or Metal validation/sanitizer coverage. The Linux
refresh above qualifies the tiny-model CUDA regression suite. The next step is
a physical-memory assessment of a concrete checkpoint and context before Milestone 4
cross-machine qualification.
