# Milestone 3 — native MLX worker

The optional Apple Silicon worker executes dense Llama and Qwen3 stages through the
MLX C++ API on Metal. It implements the existing `BackendFactory` / `StageBackend`
contract: token IDs or owned FP16 boundary bytes in, boundary bytes or a sampled token
out. The Python controller and native execution transport contain no MLX model code.
CPU and CUDA builds do not require MLX.

## Build and run

Use Apple Silicon macOS with a Metal-enabled MLX C++ installation, plus the existing
C++ Protobuf/gRPC dependencies. CMake requires exactly MLX 0.32.2 because allocator
error translation is qualified against that release. CMake accepts its
installed package prefix; see the [official build instructions](https://ml-explore.github.io/mlx/build/html/install.html).

The official 0.32.2 Python wheels also bundle C++ headers, CMake metadata, the dynamic
library, and `mlx.metallib`. This isolated dependency environment was used for local
validation; inference itself runs entirely in the native executable:

```bash
uv venv build/mlx-deps
uv pip install --python build/mlx-deps/bin/python 'mlx==0.32.2'
uv run cmake --preset dev -B build/native/mlx \
  -DHLLM_ENABLE_MLX=ON \
  -DCMAKE_PREFIX_PATH="$PWD/build/mlx-deps/lib/python3.12/site-packages/mlx"
uv run cmake --build build/native/mlx -j 2
uv run ctest --test-dir build/native/mlx --output-on-failure
```

Keep the dependency prefix available at runtime. The wheel library finds its adjacent
Metal library; a source installation must also make its `mlx.metallib` available as
specified by MLX's build instructions. Startup evaluates a real GPU operation and fails
if Metal or its kernels cannot be loaded. It never advertises a CPU fallback as MLX.

For a tiny CPU/MLX demo, prepare the fixture and plan using the commands in the
[CPU demo](milestone-1.md), substituting
`examples/profiles/workers-cpu-mlx-local.yaml` for the worker profile. Start the CPU
worker on port 50051 and the MLX worker on port 50053:

```bash
build/native/mlx/cpp/hllm-worker-cpu --listen 127.0.0.1:50051 \
  --worker-id cpu-a --model-root build/cpu-demo --memory-limit-bytes 536870912
build/native/mlx/cpp/hllm-worker-mlx --listen 127.0.0.1:50053 \
  --worker-id cpu-b --model-root build/cpu-demo --memory-limit-bytes 134217728
```

Use the MLX worker profile again with `hllm generate`. The logical worker ID `cpu-b`
identifies the second process; its advertised backend is MLX. The example is a tiny
qualification profile, not a full-checkpoint memory recommendation.

## Semantics and ownership

MLX advertises `llama.v1` and `qwen3.v1`, revision 1, with F32 or F16 execution and
matching KV dtype. F32/F16/BF16 checkpoint storage converts to the selected dtype.
The shared dense loader validates architecture, unsupported features, assigned tensor
ownership, shapes, revisions, and source paths before loading payloads. It reads and
evaluates one weight at a time; no complete CPU weight replica is retained. Non-finite
weights and F16 overflow are rejected before publishing a stage.

Supported operations include embedding, RMS normalization, Qwen3 Q/K head normalization,
RoPE, grouped-query causal attention, residuals, SiLU gated feed-forward layers, final
normalization/projection, and greedy sampling. Tied heads share their assigned embedding
array; split embedding duplication follows the existing manifest contract. Greedy ties
choose the first maximum. Only the last hidden row is projected to logits.

Normalization statistics, RoPE arithmetic, attention scores and softmax use F32.
Matrix projections and persistent activations/cache use the selected execution dtype.
The initial attention path is explicit and dense, with quadratic prefill workspace.
BF16 execution, quantization, scaled RoPE, sliding attention, biases, MoE, and multimodal
models remain unsupported and fail validation.

A serialized process-wide GPU stream is reused across stage loads and gRPC threads.
Every MLX operation, including the startup probe, runs with that stream selected.
The calling thread retains this shared GPU stream as its default; only the previous
default device is restored afterward. Querying an uninitialized thread's previous
GPU stream would create another permanent stream in MLX 0.32.2, so the backend never
does that. This matches the one-request-per-worker runtime. Each layer
explicitly evaluates its output and updated KV arrays, bounding lazy graph retention.
Sequence allocation materializes its reserved cache before admission succeeds. Cache
updates may copy storage; workspace admission does not assume in-place optimization.

Boundary input is decoded from owned little-endian FP16 bytes, with finite/shape checks.
Output is evaluated and copied into an owned, contiguous FP16 byte vector before return.
No MLX array crosses the common stage interface or network. Diagnostic traces expose
host snapshots only to numerical tests and are outside production workspace accounting.

Invalid inputs, incompatible checkpoints, and exhausted capacity use the common
categorized runtime errors, preserving their wire status through the worker services.
Unclassified framework failures remain internal errors.

Cancellation is checked before/between layers and before publishing output. It cannot
preempt a Metal kernel. Stream completion is awaited on success and attempted on every
exception path before buffers can retire. An execution failure poisons the sequence;
subsequent execution requires a fresh reservation. Known MLX 0.32.2 allocator exceptions
are translated to `std::bad_alloc`, preserving resource-exhaustion status; unrelated
execution errors retain their class. Fatal Metal failures can require worker restart.

## Unified memory and metrics

For MLX, `--memory-limit-bytes` is a single usable unified-memory budget. Choose it after
leaving macOS, other applications, framework metadata, and allocator headroom. It is not
a process RSS limit, and the worker does not automatically subtract macOS headroom from
this already-usable budget. CUDA-only device and pinned-transfer arguments are rejected.

Runtime `MemoryAmounts` adds a disjoint unified domain. MLX weights, source conversion
buffers, KV caches, conservative dense workspace, possible cache-update copies, and
transport allocations are charged there. The control service requires a single unified
budget for unified backends, checks the combined reservation before allocation, and
reports unified usage in both domain and legacy totals exactly once. Existing host,
device, and pinned-subset accounting is preserved. The planner now includes host
transport storage in unified-stage workspace, as it already does for CPU host stages.

`GetMemoryReport` reports model payloads and conservative reservations. The additive
optional `GetMetrics().allocator` message reports process-wide MLX active, cached, and
peak allocation bytes, tagged with the unified domain. These numbers are not added to
payload totals: doing so would double-count live arrays. A metrics scrape attempts
the device lock without waiting. If loading or execution holds it, the response
still identifies the worker but omits allocator telemetry; callers should retry a
later scrape rather than treating absence as zero. Older backends omit this optional
field too; the execution protocol is unchanged.

The worker also sets MLX's allocator guideline to the configured budget and caps its
cache guideline at `min(budget / 8, 64 MiB)`. MLX memory limits are guidelines, not hard
physical limits; runtime admission and system headroom remain necessary. Cached blocks
may remain after unloading. Reusing one stream avoids accumulating a new stream per
load. See [MLX memory management](https://ml-explore.github.io/mlx/build/html/python/memory_management.html)
for the distinction between active and cached allocations. The fixed cache guideline
is conservative for this milestone's fixtures; larger workloads may incur allocation
churn. Profile cache reuse against real workspace sizes before changing it in measured
placement work. Admission reservations alone do not establish a suitable cache cap.

## Qualification

See [the MLX validation report](validation/mlx.md) for final results and limitations.
The golden suite covers MLX-only execution and CPU/MLX execution in both orders.
Mixed CPU/MLX plans use global F32 execution/KV because the CPU backend requires F32;
FP16 remains the boundary representation. F16 is separately qualified on MLX-only
paths and direct split-stage numerical tests. Mixed F32/F16 stage execution is not added.

Hardware tests fail when the MLX binary or working Metal device is missing. Run only
MLX tests with `ctest --test-dir build/native/mlx -L mlx --output-on-failure`.

## Next steps

Milestone 4 will qualify MLX/CUDA across the two Tailscale machines in both stage
orders, over direct and relayed connections, with at least 256 generated tokens
while neither worker holds the complete model. First assess a concrete checkpoint
and context against available physical memory and framework overhead.
Qwen3-4B-Base remains the target. Tiny-model tests do not establish full-checkpoint
fit, numerical tolerances or latency. Measured placement remains Milestone 5.
Profile before adding fused attention, compiled functions, asynchronous evaluation,
transfer overlap or more model families.
