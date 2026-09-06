# Milestone 2 — CUDA backend

The optional Linux CUDA worker now executes dense Llama and Qwen3 stages with LibTorch/ATen.
Tiny-model numerical and single-worker process tests establish the initial execution path.
Mixed CPU/CUDA loopback correctness now passes in both orders and all tiny-fixture splits.
Opt-in bounded pinned transfers and failure/memory qualification are implemented.
Full-checkpoint inference remains next.
See the [mixed validation report](validation/mixed-cpu-cuda.md). The initial CUDA PRs established backend selection, memory domains and model execution;
the qualification PRs build on those interfaces.

## Backend and model boundaries

`ControlService` receives a `BackendFactory`; the shared worker runtime has no CPU or
LibTorch dependency. The executable selects `hllm-worker-cpu` or `hllm-worker-cuda`. CPU
builds require no Torch/CUDA installation. Torch-facing code is compiled separately from
RPC-facing code to avoid the Protobuf headers bundled in some Torch distributions.

CPU and CUDA share `model::inspect_dense_stage`: the architecture registry, revision and
feature validation, tensor ownership/shapes, source metadata and path checks. It returns
plain metadata and a source that reads one tensor at a time. It does not instantiate a
CPU backend or retain a full host copy of GPU weights. Dense configuration lives in the
model layer; CPU numerical kernels remain independent of CUDA execution.

`StageBackend::execute` accepts tokens on the first stage or owned little-endian FP16
boundary bytes on the final stage. It returns boundary bytes or a sampled token. Embedding,
owned layers and sampling remain inside that call. Production CUDA execution retains
weights, intermediate activations and KV state on the selected GPU. CPU float arrays and
Torch tensors never appear in the common execution contract. The wire protocol is unchanged.

The CUDA-only `ReferenceStage::execute_traced` interface exposes layer/cache/logit snapshots
for tiny numerical tests. Production requests do not call it. Its diagnostic host copies
are outside production workspace accounting.

## Execution and supported semantics

CUDA advertises `llama.v1` and `qwen3.v1`, revision 1 semantics, with F32 or F16 execution.
F32/F16/BF16 checkpoint payloads convert to the selected resident weight dtype. KV caches
use that same dtype. BF16 execution, quantization, scaled RoPE, sliding attention, biases,
MoE, and multimodal inputs are not implemented. Existing architecture and shape validation
rejects unsupported configurations before accepting a stage.

The initial path implements embeddings, RMS normalization, Qwen3 per-head Q/K normalization,
RoPE, grouped-query causal attention, residuals, SiLU gated feed-forward layers, final
normalization/projection, and greedy sampling. Tied heads use the resident embedding tensor;
split tied embeddings are loaded independently where assigned. Sampling uses the first
maximum on ties. Only the final row is projected to logits in production.

RMS statistics, RoPE arithmetic, attention scores and softmax use F32. Other activations and
matrix projections use the selected dtype. TF32 and reduced-precision FP16 GEMM reduction
are disabled. The implementation uses explicit dense attention, not fused attention or
custom kernels. Prefill therefore has quadratic attention workspace; this reference path
prioritizes correctness and does not establish full-model latency or memory efficiency.

Each stage owns a guarded CUDA stream and opaque sequence state. Positions must be contiguous
and fit the reserved context. Operations complete before returning control to the runtime;
error paths drain the stream before propagating the failure. A state interrupted during an
operation is marked unusable, since some cache layers may already have been written.
Cancellation is checked before/between layers and before returning output. It does not
preempt an individual CUDA kernel. Transfers default to blocking pageable copies, with opt-in bounded pinned staging
as described below. Transfer overlap is not implemented.

## Memory admission and reporting

Host, device and pinned-host amounts have independent caps. Pinned memory also counts toward
host usage and is counted once in legacy host-plus-device totals. Pageable mode reserves zero pinned bytes; pinned mode charges sequence staging. Resident GPU weights and exact per-layer KV payloads are reported in
the device domain; production retains no host weight or KV payload.

Before loading, validate all assigned metadata, then check resident device weights plus the
largest conversion/upload temporary. Host admission includes the largest checkpoint payload,
conversion/upload temporaries and a fixed allowance. Tensors are read, validated and uploaded
one at a time. Non-finite weights and weights overflowing F16 are rejected. Load failure
cannot publish a partial stage.

Before allocating sequence state, check weights plus cache plus workspace in every domain.
Workspace reserves dense projections, worst-case attention score/softmax matrices, logits,
transport copies and a fixed device allowance. Size arithmetic checks overflow. LibTorch
out-of-memory exceptions become allocation failures so the control/execution services return
resource-exhaustion errors. A rejected reservation does not retain KV state.

These are payload and conservative reservation estimates, not measured RSS, CUDA allocator
reserved memory or hard process limits. CUDA context, framework, allocator and RPC overhead
need operating-system/device headroom. Configured planner profiles remain estimates; their
workspace and transfer assumptions must be reconciled during mixed-backend qualification.
The GPU allocator may retain freed blocks for reuse after unloading, even though reported
model/request payload usage is zero.

## Linux build and tests

Prerequisites are a CUDA-capable NVIDIA driver/device, CUDA toolkit, CUDA-enabled LibTorch
with the modern C++11 ABI, and ABI-compatible Protobuf/gRPC development packages. Install
LibTorch's transitive CUDA libraries as well. Set its package prefix according to the
[official LibTorch CMake instructions](https://docs.pytorch.org/cppdocs/installing.html).
Older pre-C++11-ABI Torch distributions are rejected. Choose a host compiler supported by
your CUDA toolkit; set `CMAKE_CUDA_HOST_COMPILER` if the system compiler differs.

```bash
uv sync
uv run cmake -S . -B build/native/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DHLLM_ENABLE_CUDA=ON \
  -DCMAKE_PREFIX_PATH="/path/to/native-dependencies;/path/to/libtorch" \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/path/to/supported/g++ \
  -DTORCH_CUDA_ARCH_LIST=8.6
uv run cmake --build build/native/cuda -j 2
uv run ctest --test-dir build/native/cuda --output-on-failure
```

Use your GPU's architecture (8.6 is the RTX 3060 Ti). `CudaNumericalParity` and
`CudaBackendSmoke` require working CUDA hardware and fail rather than silently skip if it
is unavailable. CPU tests also run in a CUDA-enabled build.

Start a CUDA worker with an existing local checkpoint directory:

```bash
build/native/cuda/cpp/hllm-worker-cuda --listen 127.0.0.1:50053 \
  --worker-id cuda-a --model-root /models \
  --memory-limit-bytes 536870912 --device-memory-limit-bytes 1073741824 \
  --pinned-host-memory-limit-bytes 67108864 --device-id 0
```

The host budget keeps its existing `--memory-limit-bytes` name. The device budget is
required; `--device-id` defaults to zero. The optional pinned budget defaults to zero.
CPU workers reject CUDA-only flags. Startup probes a real ATen CUDA operation; capabilities
then advertise implemented model execution. As on CPU, health indicates a serving-capable
worker even before a stage is loaded. Deploy through `DeploymentSession` with a one-stage
F32/F16 plan for the current qualified flow; the planner still enumerates two-stage plans.
The CUDA process tests construct and exercise those one-stage plans against a tiny checkpoint.

## Validation

The second PR uses the same RTX 3060 Ti environment as the integration PR: GCC 14.4,
CUDA toolkit 13.3, C++11-ABI LibTorch 2.13.0+cu130, gRPC 1.83, Protobuf 7.35.1 and driver
610.57.04. Native dependencies and the checkout are isolated from existing workloads.
The system GCC 16 is unsupported by the CUDA toolkit. The installed Abseil shared package
has a Debug mutex-destructor link issue; Linux validation uses RelWithDebInfo.

Numerical tolerances are absolute and apply to the tiny fixture:

- Qwen3 prefill and incremental decode layer outputs, KV caches and last-row logits match
  the independent Transformers fixture at `3e-5` for F32 and `5e-3` for F16.
- Llama layer outputs match the CPU reference at the same dtype-specific tolerances.
- Llama/Qwen3 generated tokens match CPU for F32/F16/BF16 storage, tied/untied heads and
  F32/F16 execution. Both assigned CUDA stages match CPU boundaries within `5e-3` and produce
  matching sampled tokens. This is a direct backend test, not mixed-device RPC qualification.
- CUDA worker tests cover capabilities, unsupported architectures, device selection, required
  budgets, reservation accounting/cancellation, repeated generation and unload/reload in
  F32 and F16. Native tests cover invalid metadata/context/state, non-finite or overflowing
  weights, and host/device budget rejection.

Validation on 2026-09-05 passed 35 Python unit tests, 42 native CPU/common tests and 13 CPU
process cases on macOS; the Linux CUDA-enabled build passed those 42 native tests and 13
CPU process cases plus seven CUDA native tests and five CUDA process cases (45 CTest
entries). Ruff, Pyright, generated-binding reproducibility and whitespace checks passed.

For subsequent mixed-process, fault and sanitizer results, see the
[mixed qualification report](validation/mixed-cpu-cuda.md). Full-checkpoint
parity/performance and asynchronous transfer overlap remain unverified.

## Next steps

See the [mixed CPU/CUDA qualification plan](milestone-2-qualification-plan.md) for the
three-PR sequence, constraints and acceptance criteria.

Review and land the CUDA stack in dependency order. Next, assess a full-checkpoint
workload against available physical memory before scheduling inference. Account for
CPU F32 resident weights, global F32 compute/KV in mixed plans, dense attention
workspace, and measured framework/context overhead. Qwen3-4B-Base is still the
project target; choose a smaller compatible checkpoint or shorter context if needed.

The [model extension boundaries](model-extensibility.md) remain in force. Additional model
families require explicit decisions rather than a generic graph engine. MLX is Milestone 3,
cross-machine qualification is Milestone 4, and measured placement/performance is Milestone 5.

## Opt-in pinned boundary staging

`hllm-worker-cuda --boundary-transfer-mode pinned` requires a positive
`--pinned-host-memory-limit-bytes` cap (also bounded by the host cap).
The default remains `pageable`. Use `workers-cpu-cuda-pinned.yaml` for the pinned
qualification profile and the same CPU/link/workload/planner settings as the baseline.

Each reserved sequence owns one `cudaHostAlloc` buffer and a completion event.
Its size is `min(maximum_tokens * hidden_size * 2, 8 MiB)` for a split stage,
and zero for a stage owning the complete model. Admission charges these bytes
to both host workspace and its pinned subset; total usage sums host and device
only. The planner checks both budgets and rejects a pinned profile lacking a host budget.
CUDA's existing conservative device workspace covers the FP16 conversion tensor.

Boundaries larger than staging are copied in chunks through the same allocation,
including a final partial chunk. The 8 MiB cap limits pinned storage, not the
accepted boundary payload. Copies use the stage stream and wait for the completion event before staging is
reused or output bytes are published. Exception cleanup drains outstanding work.
The sequence frees staging on retirement; no cross-request pinned cache is used.
The completed stage interface, owned byte-vector boundary and protobuf remain
unchanged. Serialization and host copies still occur. This implementation does
not overlap decode steps, and pinned mode has no assumed latency advantage.

## Failure and memory qualification

The mixed fault suite uses a test-only gRPC relay to hold a prefill or decode
boundary while both native workers own their reservations. It covers client and
control cancellation, deadlines, disconnects, and loss of either worker, in
both transfer modes and stage orders. Unload rejects active work. Recoverable
faults must release reservations and permit reload plus a fresh request;
worker loss must release the survivor. Malformed/stale stage messages and
admission/load rollback reuse the CPU suite against mixed workers.

Generation deadlines first cancel the peer call and give the handler up to
100 ms to return its deadline status. Forcing `ServerContext::TryCancel()`
immediately can replace that status with `CANCELLED`; a bounded fallback is
retained for blocked writes. An idle downstream stage still uses transport
cancellation to unblock a synchronous read. Non-preemptible kernels or stalled
client writes can require that fallback; CUDA kernels cannot be forcibly stopped
at the application deadline.

Native tests inject pinned allocation/event failures through private CUDA call
wrappers and inject a reservation failure after allocating a real CUDA sequence.
No fault control is exposed through worker arguments or RPCs. A failed transfer
buffer cannot be reused. Its destructor drains pending stream work before freeing
staging, including when event recording failed. Recoverable allocation failure
is reported as resource exhaustion. Other CUDA errors retain their error class;
fatal device/context failures can require a worker restart and are not induced
by exhausting the shared machine.

Test-only diagnostics sample LibTorch allocator allocated/reserved/peak bytes,
actual pinned allocations, and `/proc` RSS during repeated request/unload cycles.
They are isolated from protobuf-facing translation units; the public metrics
schema and stage contract remain unchanged. Framework/context memory is distinct
from model reservation accounting. Reserved allocator cache may remain after
unload; qualification checks its plateau after warm-up.
