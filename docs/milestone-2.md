# Milestone 2 — CUDA backend

Milestone 1 is merged. Milestone 2 starts with a backend integration PR, followed by CUDA
model execution and then CPU/CUDA pipeline qualification. The integration target probes
LibTorch CUDA on a real device and exposes control RPCs. It **does not execute models**:
no architectures or execution dtypes are advertised, `Health.serving` is false, and
`LoadStage` rejects requests with an explicit unsupported-execution detail.

## First PR: integration boundaries

`ControlService` receives a `BackendFactory`; the shared worker runtime has no CPU or
LibTorch dependency. Torch-facing code is compiled separately from RPC-facing code,
avoiding collisions with the Protobuf headers bundled in some Torch distributions. The executable selects its factory: `hllm-worker-cpu` or the optional
Linux `hllm-worker-cuda`. CPU builds require no Torch/CUDA installation.

`StageBackend::execute` accepts token IDs on the first stage or owned little-endian FP16
boundary bytes on the final stage. It returns boundary bytes or a sampled token. Embedding,
owned layers, and sampling run within that call, so a CUDA backend can retain intermediate
activations and sequence state on the device. CPU-only reference methods remain available
to numerical tests. Neither Torch tensors nor CPU float activation arrays appear in the
common execution contract. The existing activation wire protocol remains unchanged.

Sequence state remains opaque and owned by the backend. Returned host data must be ready
for transport. Pending device work must finish before an operation returns or its buffers
are released, including exceptional/cancelled execution. Stream/event ownership and actual
GPU sequence allocations are implementation work for the next PR.

Memory admission checks three independently capped amounts: host, device, and pinned host.
Pinned bytes are a subset of host usage, with an additional cap; they are counted once in
legacy host-plus-device totals. Factories must validate peak load/conversion allocations
before loading. These runtime checks are authoritative; configured planner profiles remain
estimates, and pinned-transfer estimates must be reconciled during mixed-backend qualification.
Before sequence allocation, the runtime checks weights plus cache plus
workspace in every domain, with overflow checks. Allocation failure cannot publish a
partial stage or reservation.

`MemoryReport.domain_usage` adds per-domain weight/cache/workspace accounting while
retaining existing total fields. This is payload and reservation accounting, not RSS,
CUDA allocator-reserved memory, or a hard process limit. CUDA context, allocator pools,
LibTorch, and RPC overhead need headroom outside configured model budgets. The startup
probe allocates a tiny tensor and completes a CUDA reduction; it does not claim model
memory usage or inference readiness.

## Linux build and smoke test

Prerequisites are a CUDA-capable NVIDIA driver/device, CUDA toolkit, CUDA-enabled LibTorch
with the modern C++11 ABI, and ABI-compatible Protobuf/gRPC development packages. Install
LibTorch's transitive CUDA libraries as well. Set its package prefix according to the
[official LibTorch CMake instructions](https://docs.pytorch.org/cppdocs/installing.html).
Older pre-C++11-ABI Torch distributions are rejected. Choose a host compiler supported by
your CUDA toolkit; set `CMAKE_CUDA_HOST_COMPILER` explicitly if the system compiler differs.

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

Use the architecture for your GPU (8.6 is the RTX 3060 Ti). `CudaBackendSmoke` is registered
only when CUDA is enabled and requires working hardware; it fails rather than silently
skipping when the device/runtime is unavailable. It starts a real worker, checks its
capabilities, memory report, health and load rejection, and exercises invalid startup flags.
The CPU tests also run in a CUDA-enabled build.

Start an integration worker with an existing model-root directory:

```bash
build/native/cuda/cpp/hllm-worker-cuda --listen 127.0.0.1:50053 \
  --worker-id cuda-a --model-root /models \
  --memory-limit-bytes 536870912 --device-memory-limit-bytes 1073741824 \
  --pinned-host-memory-limit-bytes 67108864 --device-id 0
```

`--memory-limit-bytes` remains the host budget. The CUDA device budget is required;
`--device-id` defaults to zero. The optional pinned budget defaults to zero (no pinned
allocations admitted). CPU workers reject CUDA-only flags. A successful CUDA process
startup means the driver, selected device and linked ATen CUDA operation passed the probe;
use `Health` to check model-serving readiness.

## Next PRs

1. Implement CUDA dense Llama/Qwen3 loading, embedding, normalization, RoPE, attention/KV,
   feed-forward layers, final projection and greedy sampling with ATen/LibTorch. Keep
   intermediates and KV on the selected GPU. Validate architectural revisions, features,
   shapes and storage/compute dtypes explicitly. Establish tiny-model parity against the
   CPU/Transformers fixtures before advertising executable capabilities.
2. Implement and qualify boundary transfers, pinned staging, stream/event cleanup and
   memory reservations under real CUDA execution. Test both CPU/CUDA stage orders,
   repeated requests, cancellation, deadlines, allocation failure and peer loss. Only
   then attempt a full checkpoint and record its measured limitations.

These changes preserve the [model extension boundaries](model-extensibility.md). Additional
model families remain separate decisions; this PR adds no general model graph engine,
MoE routing, recurrent state implementation or multimodal input contract. MLX is Milestone 3,
cross-machine qualification is Milestone 4, and measured placement/performance is Milestone 5.

## Validation record — 2026-09-05

- macOS CPU Debug build: 42 native tests and all 13 real-process integration cases passed
  (43 CTest entries). The new tests cover independent memory rejection before allocation,
  accounting overflow, pinned inclusion, per-domain reporting/release, and the opaque
  prefill/decode contract against the CPU reference.
- Linux RTX 3060 Ti: the CUDA-enabled RelWithDebInfo build passed all 44 CTest entries:
  42 native tests, 13 CPU process cases, and three CUDA startup/control smoke cases.
  The CUDA process completed an ATen allocation, reduction and host result on device 0.
  Invalid device IDs and missing device budgets fail startup. The CPU binary has no
  Torch/CUDA dynamic-library dependency even in this build.
- 35 Python unit tests, Ruff, Pyright (including CUDA smoke tests), generated-binding
  reproducibility and whitespace checks passed locally.

The Linux check used GCC 14.4, CUDA toolkit 13.3, the available C++11-ABI LibTorch
2.13.0+cu130 build, gRPC 1.83 and Protobuf 7.35.1, with driver 610.57.04. Native build
dependencies were installed in an isolated environment; existing workloads were retained.
The system GCC 16 was unsupported by the CUDA toolkit, so nvcc used GCC 14 explicitly.
The installed Abseil 20260526 shared package failed Debug linking on its mutex destructor;
RelWithDebInfo passed. That package/toolchain limitation is not Debug validation.
No CUDA model parity, mixed-device inference, full-checkpoint run or sanitizer coverage
is claimed by this integration PR.
