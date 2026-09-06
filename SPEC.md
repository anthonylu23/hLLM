# hLLM Runtime — Project Specification

**Status:** Draft 0.2  
**Date:** 2026-09-01  
**Working name:** hLLM Runtime  
**Repository name:** `hllm-runtime`

## 1. Executive summary

hLLM Runtime is a heterogeneous inference system that hosts one decoder-only language model across multiple machines and accelerator platforms. It partitions the model into contiguous layer ranges, assigns each range to a native worker, transfers hidden-state activations directly between workers, and exposes the deployment through one OpenAI-compatible API.

The first supported path is:

```text
Apple Metal through MLX  ↔  NVIDIA CUDA through LibTorch/ATen
```

The system uses:

- **C++ native workers** for model execution, KV-cache ownership, activation transport, and the autoregressive hot path.
- **Python control plane** for the public API, tokenization, worker registry, placement planning, deployment management, and observability aggregation.
- **Protobuf and gRPC** as the language-neutral process boundary.
- **Tailscale** for private identity, addressing, encryption, NAT traversal, and routing between machines.
- **Safetensors** as the canonical weight format.

Python never relays hidden-state activations and does not issue one RPC per stage per generated token. After request setup, native workers drive the decode loop directly. Python receives sampled token events for client streaming and may cancel the native request asynchronously.

## 2. Project name

`hLLM` naturally expands to **heterogeneous LLM** and communicates the core idea clearly. Use **hLLM Runtime** in prose to distinguish it from unrelated projects already using HLLM.

Technical names:

- Repository: `hllm-runtime`
- Python package: `hllm_control`
- CLI: `hllm`
- Native worker binaries: `hllm-worker-cpu`, `hllm-worker-mlx`, and `hllm-worker-cuda`

The public brand should be revisited before a broad release because `HLLM` is not globally unique.

## 3. Problem statement

Open-weight models often exceed the memory of any one locally available accelerator. A user may nevertheless own several machines whose combined memory is sufficient—for example, a Mac Studio and a Linux CUDA workstation.

Existing high-performance inference runtimes generally assume a homogeneous distributed environment. hLLM Runtime targets the case where participating machines differ in accelerator platform, memory capacity and architecture, per-layer speed, supported kernels, and network performance.

Three constraints drive the architecture:

1. Every generated token traverses every pipeline stage, so network and conversion latency affect inter-token latency.
2. The slowest stage limits steady-state pipeline throughput, so equal layer counts are rarely optimal.
3. Model weights are only one part of memory use; KV cache, workspace, buffers, allocator overhead, and operating-system headroom must also be reserved.

## 4. Scope

### 4.1 MVP goals

- Host one Llama-compatible dense decoder model across one MLX worker and one CUDA worker.
- Load only the assigned layer range and required special tensors on each machine.
- Support both stage orders: MLX → CUDA and CUDA → MLX.
- Support prefill and autoregressive decode.
- Keep every layer's KV cache on the worker that owns the layer.
- Transfer framework-neutral FP16 activations directly between workers.
- Run the native autoregressive loop without Python in the stage-to-stage critical path.
- Use Tailscale for private cross-machine connectivity.
- Automatically choose a memory-feasible stage order and split point.
- Refine placement with measured compute, conversion, and link performance.
- Stream output through an OpenAI-compatible API.
- Expose stage, memory, network, and request metrics.

### 4.2 Later goals

- ROCm through a LibTorch/ATen backend build.
- More than two pipeline stages.
- Continuous batching and multiple in-flight microbatches.
- Quantized weights and optional compressed boundary activations.
- Replicated stages and request recovery.
- Additional dense decoder architectures.
- vLLM-derived kernels or a vLLM-backed worker where a stable integration boundary exists.

### 4.3 MVP non-goals

- Training or gradient synchronization.
- Tensor parallelism across heterogeneous devices.
- Mixture-of-Experts models.
- Arbitrary Hugging Face architectures or remote model code.
- Internet-scale untrusted workers.
- Mid-request KV-cache migration.
- Dynamic layer movement while requests are active.
- Seamless recovery after a worker loses its KV cache.
- Cross-backend bitwise-identical results.
- Matching vLLM's throughput or model coverage.

## 5. Design principles

1. **Native hot path.** Activations, KV caches, sampling, and the decode loop remain in native workers.
2. **Thin control plane.** Python coordinates deployments and clients but does not proxy model tensors.
3. **Framework-neutral boundaries.** No Torch, MLX, NumPy, pickle, or language-specific objects cross the wire.
4. **Memory before performance.** A placement is optimized only after it is proven feasible with headroom.
5. **Measure real hardware.** Placement uses observed backend and link profiles instead of advertised FLOPS.
6. **Static request ownership.** A request remains on the same stages for its lifetime.
7. **Replaceable data transport.** gRPC is the MVP transport; the stage protocol survives a later data-plane replacement.
8. **One architecture first.** Numerical correctness on one Llama-compatible model family takes priority over broad coverage.

## 6. Technology stack

### 6.1 Native runtime

- C++20.
- CMake with CMake Presets and Ninja.
- MLX C++ API on Apple Silicon.
- LibTorch/ATen and CUDA on NVIDIA systems.
- gRPC C++ and Protobuf.
- Buf for Protobuf linting, generation, and compatibility checks.
- `spdlog` for native structured logging.
- OpenTelemetry C++ for tracing.
- `prometheus-cpp` for native metrics.
- GoogleTest and Google Benchmark.
- Clang sanitizers for CPU-side development.

### 6.2 Python control plane

- Python 3.12 or later.
- FastAPI and Uvicorn.
- Pydantic and Pydantic Settings.
- Asynchronous `grpcio` clients and servers.
- Hugging Face `tokenizers`.
- `transformers` for supported model configuration parsing only.
- Safetensors tooling for model inspection and manifest preparation.
- NumPy for placement and benchmark calculations.
- SQLite for profiles, deployment plans, and benchmark history.
- Typer for the CLI.
- Prometheus client, OpenTelemetry SDK, and `structlog`.
- `uv`, Ruff, Pyright, pytest, pytest-asyncio, and Hypothesis.

### 6.3 Network substrate

- Standard Tailscale daemon on every machine.
- MagicDNS for per-worker addresses.
- Tailscale Grants for least-privilege reachability.
- Tailscale Serve raw TCP forwarding for the initial deployment experience.
- Direct-connection qualification before activating a performance deployment.

### 6.4 Deliberately excluded from the MVP

- Rust.
- Ray and Kubernetes.
- Redis, NATS, Kafka, or a distributed database.
- A custom TCP, QUIC, or RDMA protocol before profiling gRPC.
- Custom CUDA or Metal kernels before the reference path is correct.

## 7. System architecture

```text
                                Control plane
                     ┌─────────────────────────────┐
Client ── HTTP/SSE ─▶│ Python controller          │
                     │ OpenAI-compatible API       │
                     │ tokenizer                   │
                     │ worker registry             │
                     │ placement planner           │
                     │ deployment manager          │
                     │ metrics and traces          │
                     └──────────────┬──────────────┘
                                    │ gRPC control and token events
                                    ▼
                                Native data plane

┌─────────────────────────────┐  activations  ┌─────────────────────────────┐
│ C++ stage worker A          │──────────────▶│ C++ stage worker B          │
│ MLX or CUDA backend         │               │ CUDA or MLX backend         │
│ embedding + layers [0,m)    │◀── token ID ──│ layers [m,L) + LM head      │
│ local KV cache              │               │ sampling + local KV cache   │
└─────────────────────────────┘               └─────────────────────────────┘
             ▲                                            ▲
             └────────────── Tailscale tailnet ───────────┘
```

### 7.1 Process model

```text
hllm-controller      Python control plane and public API
hllm-worker-cpu      Native reference and protocol worker
hllm-worker-mlx      Native MLX worker for macOS
hllm-worker-cuda     Native LibTorch/CUDA worker for Linux
```

Workers are separate processes rather than Python extension modules. This provides backend isolation, independent restarts, simpler dependency boundaries, and a production-realistic protocol from the beginning. Optional Python bindings may be added for local tests, but they are not the production architecture.

## 8. Control plane

The Python controller owns:

- OpenAI-compatible HTTP endpoints.
- Prompt tokenization and output detokenization.
- Worker registration and heartbeat state.
- Capability, memory, benchmark, and link profiles.
- Layer placement and deployment-plan versioning.
- Model preparation manifests.
- Request admission and initial capacity reservation.
- Request cancellation and deadlines.
- Token-event streaming to clients.
- Metrics aggregation and deployment diagnostics.

The controller does not own hidden-state buffers, backend tensors, per-layer KV caches, stage-to-stage token scheduling, or the native sampling loop.

## 9. Native worker runtime

### 9.1 Common runtime responsibilities

- Receive and validate immutable deployment plans.
- Load only the assigned weights.
- Own backend-specific tensors and persistent buffers.
- Allocate, index, and release local KV-cache blocks.
- Maintain per-request native state machines.
- Execute prefill and decode.
- Transfer activations directly to the next stage.
- Return sampled token IDs directly to stage zero.
- Emit token and terminal events to the controller.
- Apply downstream backpressure.
- Enforce sequence numbers, deadlines, and cancellation.
- Report memory, execution, and transport metrics.

### 9.2 Common interfaces

The runtime shares protocol and buffer abstractions, not a universal device tensor type.

```cpp
struct ActivationView {
    std::span<const std::byte> data;
    std::vector<std::int64_t> shape;
    DataType dtype;
    Layout layout;
};

class StageBackend {
public:
    virtual ~StageBackend() = default;
    virtual Capabilities capabilities() const = 0;
    virtual LoadReport load_stage(const StageLoadSpec&) = 0;
    virtual void unload_stage(DeploymentId) = 0;
    virtual void allocate_sequence(const SequenceSpec&) = 0;
    virtual StageOutput prefill(const StageInput&) = 0;
    virtual StageOutput decode(const StageInput&) = 0;
    virtual void release_sequence(RequestId) = 0;
    virtual MemoryReport memory_report() const = 0;
};
```

The MLX backend converts a boundary buffer into `mx::array`. The CUDA backend converts it into `torch::Tensor`. Backend tensor lifetimes do not leak into the common transport layer.

### 9.3 Buffer ownership

Use RAII-owned, pooled buffers with explicit capacity, logical size, alignment, memory kind, and release-to-pool callback.

Initial memory kinds:

- Ordinary host memory.
- Page-locked CUDA host memory.
- MLX-compatible host-visible memory where supported.
- CUDA device memory owned only by the CUDA backend.
- MLX arrays owned only by the MLX backend.

No buffer may be reused until every backend event and network send that depends on it has completed.

## 10. Backend adapters

### 10.1 CPU reference backend

The CPU backend is the executable specification for transformer-block ordering, RoPE, RMSNorm, grouped-query attention, causal masks, KV-cache indexing, sampling, and RNG semantics. It prioritizes determinism and inspectability over speed.

### 10.2 CUDA backend

Use LibTorch/ATen initially.

```text
network receive buffer
  → pooled pinned host buffer
  → asynchronous H2D copy
  → torch::Tensor on CUDA
  → assigned blocks and local KV cache
  → asynchronous D2H copy
  → pooled pinned host buffer
  → native gRPC send
```

Requirements:

- Dedicated transfer and compute streams where useful.
- CUDA events for safe buffer reuse and send readiness.
- Double-buffered input and output storage.
- Explicit allocator and global memory reporting.
- Deterministic reference mode.
- Final normalization, LM head, and on-device sampling when assigned the last stage.
- No CUDA graph, FlashAttention-specific, or custom-kernel requirement for the first correct implementation.

Later optimizations may include CUDA graphs, fused attention, custom kernels, or vLLM-derived components when justified by profiles.

### 10.3 MLX backend

Use the official MLX C++ API.

```text
network receive buffer
  → host-visible input
  → mx::array
  → assigned blocks and local KV cache
  → explicit MLX evaluation
  → contiguous host-visible output
  → native gRPC send
```

Requirements:

- Direct loading of assigned Safetensors weights or conversion from the prepared native manifest.
- Explicit evaluation before a boundary buffer is transmitted.
- Exact agreement with reference RoPE, RMSNorm, mask, GQA, and cache semantics.
- MLX active, cached, and peak-memory reporting.
- Configurable unified-memory headroom for macOS.
- Final normalization, LM head, and sampling when assigned the last stage.
- Asynchronous evaluation and compiled functions only after correctness is established.

## 11. Model and weight format

The canonical source is a supported Hugging Face-style model directory containing `config.json`, tokenizer files, a Safetensors index, and sharded Safetensors files.

### 11.1 Model preparation

The Python `hllm prepare` command:

1. Validates the architecture and configuration.
2. Reads the Safetensors index without loading the full model.
3. Maps logical components and layers to tensor names and files.
4. Records shapes, dtypes, sizes, and shared weights.
5. Emits a versioned model manifest.
6. Optionally verifies file hashes.

### 11.2 Native loading

Workers consume the manifest and a strict native Safetensors reader. They map or read only tensors required by their stage.

Requirements:

- Python pickle is never accepted.
- Remote model code is never executed.
- Tensor lengths, offsets, shapes, and dtypes are validated before allocation.
- Backend-specific packed or transposed copies are measured and reported.
- Any duplication of tied embeddings is explicit in the deployment plan and memory model.

### 11.3 Initial validation models

The first real-model integration target is `tiiuae/Falcon3-3B-Base`, pinned to an immutable
repository revision. Its configuration declares `LlamaForCausalLM`, and its indexed BF16
Safetensors shards exercise GQA, explicit head dimensions, selective shard loading, and
asymmetric embedding and LM-head placement.

Automated tests use a generated tiny Llama model plus a sparse Falcon3-shaped metadata
fixture. The sparse fixture reproduces the real tensor dimensions and logical byte sizes but
does not substitute for validating the pinned upstream weight bytes.

The next benchmark target is `Qwen/Qwen3-4B-Base` at revision
`906bfd4b4dc7f14ee4320094d8b41684abff8539`, using the `qwen3.v1` adapter.
Falcon results remain a historical baseline. Qwen3 extends the shared dense decoder
operations with per-head query/key RMSNorm before RoPE and attention projection widths
independent of hidden size. Validation includes a full-size sparse metadata fixture and
a tiny two-layer Transformers oracle; see [Qwen3 support](docs/qwen3.md).

## 12. Partition rules

- Only contiguous transformer-block ranges are supported.
- Stage zero owns token embeddings and layers `[0, m)`.
- The final stage owns layers `[m, L)`, final normalization, LM head, and sampling.
- Tied embeddings stay on one stage unless the plan explicitly duplicates them.
- KV-cache entries remain on the worker that owns their layer.
- A deployment plan cannot change while it has active requests.
- The final stage returns sampled token IDs and requested top-k log probabilities, not complete logits.
- Stop token IDs and maximum-token termination execute natively.
- Stop strings may be detected by the Python detokenizer, which then sends asynchronous cancellation.

## 13. Request execution

### 13.1 Setup

1. Python validates and tokenizes the request.
2. Python admits it against deployment capacity.
3. Python opens or uses a generation stream to stage zero.
4. Native workers reserve request and KV-cache state.
5. Python sends token IDs and generation parameters once.

### 13.2 Prefill

1. Stage zero embeds tokens and executes its blocks.
2. It sends FP16 hidden states directly to the next native stage.
3. Each later stage executes its blocks and retains local KV entries.
4. The final stage computes logits and samples the first output token.

### 13.3 Decode

1. The final stage sends the sampled token ID directly to stage zero.
2. Stage zero emits a token event to Python for client streaming.
3. Stage zero starts the next native decode step.
4. Activations pass through the stage chain.
5. The final stage samples the next token.
6. The loop continues until a native stop condition, cancellation, deadline, or failure.

Python does not schedule each stage invocation.

### 13.4 Cleanup

- The native driver broadcasts terminal state to all stages.
- Every stage releases request buffers and KV blocks.
- Python closes the client stream and records terminal metrics.
- Cleanup is idempotent.

## 14. Internal protocol

Use Protobuf schemas with generated C++ and Python bindings.

### 14.1 Control service

```protobuf
service WorkerControl {
  rpc GetCapabilities(Empty) returns (Capabilities);
  rpc QualifyLink(LinkQualificationRequest) returns (LinkProfile);
  rpc LoadStage(LoadStageRequest) returns (LoadStageResponse);
  rpc UnloadStage(UnloadStageRequest) returns (Empty);
  rpc ReserveRequest(ReserveRequestMessage) returns (ReserveResponse);
  rpc CancelRequest(CancelRequestMessage) returns (Empty);
  rpc GetMemoryReport(Empty) returns (MemoryReport);
  rpc GetMetrics(Empty) returns (WorkerMetrics);
  rpc Health(Empty) returns (HealthResponse);
}
```

### 14.2 Generation service

The controller communicates with stage zero through a streaming RPC:

```protobuf
service Generation {
  rpc Generate(GenerationRequest) returns (stream GenerationEvent);
}
```

Events include token IDs, optional top-k log probabilities, prefill completion, usage, and terminal state.

### 14.3 Stage data service

Adjacent native workers maintain long-lived bidirectional streams:

```protobuf
service StageExecution {
  rpc Execute(stream StageMessage) returns (stream StageMessage);
}
```

For two workers, one native channel can carry forward activations and reverse token feedback. More-stage deployments may use a direct final-stage-to-stage-zero feedback channel.

### 14.4 Tensor envelope

```text
TensorEnvelope
  protocol_version
  deployment_id
  deployment_version
  request_id
  microbatch_id
  sequence_number
  phase: PREFILL | DECODE
  first_position
  sequence_lengths[]
  cache_slot_ids[]
  shape[]
  dtype
  layout
  payload_length
  optional_checksum
  payload
```

MVP boundary representation:

- Dense, contiguous, row-major, little-endian FP16.
- Prefill shape `[batch, sequence, hidden_size]`.
- Decode shape `[batch, 1, hidden_size]`.
- Protobuf `bytes` payload.
- gRPC compression disabled.

### 14.5 Error classes

- `INVALID_REQUEST`
- `INCOMPATIBLE_WORKER`
- `STALE_DEPLOYMENT`
- `OUT_OF_ORDER`
- `RESOURCE_EXHAUSTED`
- `DEADLINE_EXCEEDED`
- `WORKER_UNAVAILABLE`
- `BACKEND_ERROR`
- `TRANSPORT_ERROR`

## 15. Tailscale integration

### 15.1 Responsibilities delegated to Tailscale

- Tailnet device authentication.
- Stable private IPs and MagicDNS names.
- WireGuard encryption.
- NAT traversal.
- Direct, peer-relayed, or DERP-relayed connectivity.
- Port-level reachability through Grants.

### 15.2 Responsibilities retained by hLLM

- Worker registration and capabilities.
- Deployment identity and versioning.
- Model-stage semantics and tensor framing.
- Request ordering and flow control.
- Cancellation, deadlines, and error handling.
- Application-level deployment credentials.
- Metrics and tracing.

### 15.3 MVP deployment

- Install the standard Tailscale client on each machine.
- Run each native worker on localhost and expose its gRPC port privately with Tailscale Serve, or bind directly to its Tailscale interface.
- Address workers by individual MagicDNS names.
- Do not put stateful stage workers behind a load-balanced Tailscale Service.
- Have native workers register outbound with the Python controller.
- Do not require the Tailscale administration API for ordinary discovery.

Example:

```bash
tailscale serve --tcp=50051 tcp://127.0.0.1:50051
```

### 15.4 Access policy

Recommended tags:

```text
tag:hllm-controller
tag:hllm-worker
```

Authorized clients reach only the controller API. The controller reaches worker control and generation ports. Workers reach other workers' stage-data ports. Other tailnet devices cannot reach native worker ports.

### 15.5 Link qualification

Before deployment, hLLM must:

1. Warm each required connection.
2. Determine whether it is direct, peer-relayed, or DERP-relayed.
3. Measure p50 and p99 RTT.
4. Measure bandwidth at representative prefill and decode payload sizes.
5. Measure sender and receiver conversion overhead separately.
6. Persist a versioned `LinkProfile`.
7. Warn on relayed paths and reject them when `require_direct_connection` is enabled.

## 16. Memory model

Memory feasibility is a hard constraint.

Memory is budgeted by domain: host memory, pinned host memory, CUDA device memory,
or Apple unified memory. In the CPU/CUDA runtime, host usage includes pinned allocations;
pinned usage has an additional independent cap and is not added again to total bytes.
For worker `j` and memory domain `d`:

```text
usable_memory[j,d]
  = configured_memory_limit[j,d]
  - runtime_reserve[j,d]
  - safety_margin[j,d]
```

Every allocation is charged to its actual domain. Backend weights, KV cache, and device
workspace are normally charged to CUDA device memory or MLX unified memory; transport
staging is additionally charged to host or pinned-host memory where applicable. For a
candidate stage and domain:

```text
required_memory
  = backend_weight_memory
  + reserved_KV_memory
  + measured_peak_workspace
  + activation_and_transport_buffers
  + allocator_fragmentation_allowance
```

### 16.1 Weight memory

Store backend-specific measurements per tensor or layer. Include parameters, packed or transposed copies, embeddings, LM head, tied-weight duplication, and persistent kernel metadata. Theoretical tensor bytes are used only before a backend dry-load profile exists.

### 16.2 KV-cache memory

```text
KV_bytes_per_layer_per_cached_token
  = 2 × num_kv_heads × head_dim × kv_dtype_bytes

raw_KV_memory
  = layers_on_stage
  × configured_total_cached_tokens
  × KV_bytes_per_layer_per_cached_token

reserved_KV_memory
  = raw_KV_memory × kv_overhead_factor
```

The initial configurable `kv_overhead_factor` is `1.10`. Capacity uses cached tokens across all active sequences, not one request's context length.

### 16.3 Workspace and buffers

Measure workspace using representative prefill and decode buckets:

1. Load a candidate stage.
2. Reset backend peak-memory accounting.
3. Exercise supported prompt and batch buckets with populated caches.
4. Record peak memory above steady-state weights and KV allocations.
5. Add fragmentation and safety allowances.

CUDA profiles include global device memory and LibTorch allocator metrics. MLX profiles include active, cached, and peak memory. MLX unified-memory budgets must leave explicit macOS headroom.

## 17. Placement planner

The Python planner creates a versioned deployment plan. Native workers independently validate that the plan is loadable before it becomes active.

### 17.1 Inputs

- Ordered model layers and special tensors.
- Backend-specific tensor sizes.
- Worker memory budgets.
- Configured total KV-cache capacity.
- Per-layer or representative prefill/decode profiles.
- Tensor conversion profiles.
- Link RTT, bandwidth, and connection type.
- Target prompt length, output length, concurrency, and batch distribution.
- Objective weights for TTFT, ITL, pipeline period, and memory pressure.

### 17.2 Two-worker enumeration

For workers `A` and `B` and `L` blocks:

1. Evaluate `A → B` and `B → A`.
2. Enumerate every split `m` in `[1, L-1]`.
3. Add embeddings to the first stage and final norm plus LM head to the last stage.
4. Calculate complete memory use for both stages.
5. Reject candidates exceeding either budget.
6. Predict prefill, decode, and pipeline performance.
7. Add conversion, link, relay, and memory-pressure penalties.
8. Select the lowest-scoring feasible plan.
9. Retain rejected and runner-up plans for explanation and fallback.

### 17.3 Performance estimates

```text
stage_prefill_time
  = sum(profiled_layer_prefill_times) + backend_fixed_overhead

stage_decode_time
  = sum(profiled_layer_decode_times) + backend_fixed_overhead

boundary_time
  = measured_fixed_link_latency
  + activation_bytes / measured_bandwidth
  + sender_conversion_time
  + receiver_conversion_time

TTFT
  ≈ sum(stage_prefill_times)
  + prefill_boundary_times
  + first_sampling_time

ITL
  ≈ sum(stage_decode_times)
  + decode_boundary_times
  + sampling_time
  + token_feedback_time

pipeline_period
  ≈ max(stage_service_times, serialized_boundary_service_times)
```

Objective:

```text
score
  = w_ttft   × predicted_TTFT
  + w_itl    × predicted_ITL
  + w_period × predicted_pipeline_period
  + w_memory × memory_pressure_penalty
  + relay_penalty

memory_pressure_penalty
  = max(used_memory[j] / usable_memory[j])²
```

### 17.4 More than two workers

For a fixed worker order, use dynamic programming over workers and placed layers. Small worker sets may enumerate orders; larger sets use capability-based ordering followed by local search.

## 18. Scheduling

### 18.1 MVP

- One active inference microbatch is sufficient for the first end-to-end milestone.
- The native driver at stage zero controls prefill/decode progression.
- Downstream workers apply bounded backpressure.
- The Python controller controls admission, cancellation, and client streaming.

### 18.2 Continuous batching milestone

- Separate prefill and decode queues.
- Native shape and phase buckets.
- Multiple in-flight microbatches occupying different stages.
- Fairness so long prefills do not indefinitely delay decode.
- Per-request deadlines and cancellation.
- No migration of active request state or KV-cache ownership.

## 19. Public API

Initial endpoints:

```text
POST /v1/chat/completions
POST /v1/completions
GET  /v1/models
GET  /v1/deployments
GET  /health
GET  /metrics
```

Initial generation support:

- Streaming and non-streaming output.
- Greedy, temperature, top-p, and top-k sampling.
- Stop token IDs and stop strings.
- Maximum generated tokens.
- Optional limited top-k log probabilities.

Unsupported OpenAI parameters return an explicit error and are never silently ignored.

## 20. Configuration

```yaml
model:
  id: meta-llama/example-model
  path: /models/example-model
  dtype: fp16
  architecture: llama

network:
  require_tailnet: true
  require_direct_connection: true
  worker_port: 50051

capacity:
  total_cached_tokens: 32768
  kv_overhead_factor: 1.10
  memory_safety_fraction: 0.10

placement:
  mode: auto
  objective: interactive
  weights:
    ttft: 0.25
    itl: 0.50
    pipeline_period: 0.20
    memory_pressure: 0.05

workers:
  - name: mac-studio
    endpoint: mac-studio.example.ts.net:50051
    backend: mlx
    memory_limit_gib: 48
  - name: cuda-box
    endpoint: cuda-box.example.ts.net:50051
    backend: cuda
    memory_limit_gib: 20
```

Defaults and objective weights are embedded in the deployment plan for reproducibility.

## 21. Observability

Controller metrics:

- Requests, active requests, queue depth, and cancellations.
- TTFT and inter-token latency histograms.
- End-to-end tokens per second.
- Worker and deployment health.
- Predicted versus actual placement performance.

Worker metrics:

- Stage prefill and decode duration.
- Tensor materialization and host/device copy duration.
- Native queue and backpressure time.
- Bytes sent and received.
- KV-cache tokens and bytes.
- Active, cached, peak, free, and reserved memory where supported.
- Buffer-pool utilization.

Link metrics:

- Direct, peer-relayed, or DERP-relayed state.
- RTT distribution and payload-specific throughput.
- Reconnects and stream errors.

Every operation carries a trace, deployment, request, microbatch, token-position, stage, worker, and sequence identifier. One token must be traceable from stage zero through sampling and feedback.

## 22. Security

- Workers listen only on localhost plus a private Tailscale exposure, or directly on the Tailscale interface.
- Tailnet Grants restrict controller, worker, and client reachability.
- Application-level deployment credentials supplement tailnet identity.
- Pickle and arbitrary object deserialization are prohibited.
- Model paths and registries are explicitly allowlisted.
- Tensor lengths and dimensions are validated before allocation.
- Maximum batch, sequence, shape, and message sizes are enforced.
- Protocol and deployment versions are mandatory.
- Prompt-derived activations are treated as sensitive data.
- Native parsers and control surfaces receive fuzz testing.

## 23. Failure behavior

### 23.1 MVP

- Missed heartbeats stop new admissions.
- Loss of a stage stream fails its active requests.
- The native driver sends best-effort cleanup to surviving stages.
- Requests that emitted no output may restart from prefill after recovery.
- Requests that emitted tokens fail explicitly and are not silently replayed.
- Worker re-registration does not imply KV-cache recovery.
- Cleanup is idempotent.

### 23.2 Later

- Replicated stages and standby plans.
- Prefix-cache checkpointing.
- Request restart from a safe token boundary.
- Worker draining and replanning between requests.

## 24. Testing strategy

### 24.1 Unit and native quality tests

- Buffer ownership and reuse.
- Tensor envelope parsing and validation.
- Layer-range tensor selection.
- KV-cache indexing and cleanup.
- Memory formulas and placement scoring.
- Protocol compatibility and errors.
- AddressSanitizer, UndefinedBehaviorSanitizer, and applicable ThreadSanitizer tests.
- Fuzzing for manifests and tensor envelopes.
- Google Benchmark for buffers, conversions, and stage operations.

### 24.2 Numerical tests

Use a tiny deterministic Llama configuration with fixed weights and prompts. Compare CPU reference, CUDA, MLX, CPU/CUDA, CPU/MLX, and MLX/CUDA in both orders and at every valid split point.

Metrics:

- Absolute and relative error.
- Cosine similarity.
- Logit KL divergence.
- Top-k overlap.
- Greedy token agreement.

### 24.3 Integration and performance tests

- Two native CPU workers on one machine.
- Python controller with generated gRPC clients.
- Direct Mac-to-CUDA tailnet execution.
- Relayed-connection warning and rejection.
- Cancellation during prefill and decode.
- OOM rejection without process termination.
- Worker restart and stale-deployment rejection.
- Artificial latency, bandwidth, disconnects, and malformed envelopes.
- Prompt lengths 128, 512, 2K, and 8K where supported.
- Decode batches 1, 2, 4, and 8.
- Both worker orders and every feasible split.
- Conversion time isolated from link time.
- Predicted versus measured TTFT, ITL, throughput, and peak memory.

## 25. Repository layout

```text
hllm-runtime/
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
├── uv.lock
├── proto/
│   ├── control.proto
│   ├── execution.proto
│   ├── model.proto
│   └── telemetry.proto
├── cpp/
│   ├── include/hllm/
│   ├── runtime/
│   ├── transport/
│   ├── weights/
│   ├── backends/
│   │   ├── cpu/
│   │   ├── mlx/
│   │   └── cuda/
│   └── workers/
├── python/hllm_control/
│   ├── api/
│   ├── controller/
│   ├── planner/
│   ├── prepare/
│   └── cli/
├── tests/
│   ├── cpp/
│   ├── python/
│   ├── numerical/
│   └── integration/
├── benchmarks/
├── deploy/
│   ├── systemd/
│   ├── launchd/
│   └── tailscale/
└── examples/
```

## 26. Milestones

### Milestone 0 — protocol, model manifest, and planner

- Define Protobuf schemas.
- Implement `hllm prepare` for one Llama-compatible model.
- Compute per-layer weight and KV-cache estimates.
- Record worker budgets and link profiles.
- Enumerate and explain feasible two-worker plans.

**Exit criterion:** The controller emits a versioned placement plan without running distributed inference.

### Milestone 1 — native CPU pipeline

Implemented for tiny Llama and Qwen3 validation models; see the
[implementation, limits and local demo](docs/milestone-1.md). Model adapters, backend
implementations and common runtime responsibilities follow the
[extension boundaries](docs/model-extensibility.md).

- Implement common C++ runtime and buffer abstractions.
- Implement the CPU reference backend.
- Run two native worker processes through the Python controller.
- Support prefill, greedy decode, token streaming, and cleanup.

**Exit criterion:** Split native CPU execution matches the unsplit reference within defined tolerances.

### Milestone 2 — CUDA worker

- Implement LibTorch/ATen Llama stages.
- Add CUDA KV cache, pinned buffers, streams, events, and sampling.
- Add native metrics and memory reporting.

**Exit criterion:** CPU/CUDA split execution passes numerical, lifecycle, and memory tests.

### Milestone 3 — MLX worker

- Implement equivalent MLX C++ stages.
- Add explicit evaluation and unified-memory reporting.
- Validate MLX full-model and split execution.

**Exit criterion:** CPU/MLX and MLX-only paths pass the golden suite.

### Milestone 4 — MLX ↔ CUDA over Tailscale

- Run both stage orders on separate tailnet machines.
- Qualify direct and relayed connections.
- Use persistent native stage streams.
- Run the autoregressive loop without Python stage scheduling.

**Exit criterion:** The deployment streams at least 256 generated tokens while neither machine holds the complete model.

### Milestone 5 — measured automatic placement

- Add backend dry-load memory measurements.
- Profile layer compute, conversion, and link transfer.
- Select stage order and split automatically.
- Compare predicted and actual metrics.

**Exit criterion:** The chosen plan is memory-safe and within 15% of the best measured feasible split for the target workload.

### Milestone 6 — continuous batching

- Add native prefill and decode queues.
- Pipeline multiple microbatches.
- Add fairness, backpressure, cancellation, and admission limits.
- Complete the OpenAI-compatible serving behavior in scope.

**Exit criterion:** A defined multi-request soak test completes without KV-cache or buffer leaks.

### Milestone 7 — ROCm and additional stages

- Build the LibTorch backend for supported ROCm systems.
- Generalize placement to more than two workers.
- Validate that no control or tensor protocol changes are required.

**Exit criterion:** A third backend or stage can be added through the existing native backend contract.

## 27. MVP acceptance criteria

- One supported model runs across one MLX Mac and one CUDA Linux machine.
- Each machine loads only its assigned stage and required special tensors.
- Both MLX → CUDA and CUDA → MLX orders work.
- Workers communicate through private tailnet endpoints.
- Direct versus relayed link state is recorded.
- Hidden-state activations pass directly between native workers.
- Python does not proxy activations or schedule each stage per token.
- KV caches remain local and are released on completion or cancellation.
- At least 256 decode steps complete without cache corruption or buffer leaks.
- Cross-backend logits satisfy documented numerical tolerances.
- The controller streams output through an OpenAI-compatible endpoint.
- Placement accounts for weights, KV cache, workspace, buffers, fragmentation, and safety margin.
- Automatic placement is benchmarked against all feasible two-worker splits.
- TTFT, ITL, throughput, stage times, bytes transferred, and peak memory are observable.

## 28. Risks and mitigations

### C++ complexity and memory safety

**Risk:** Native ownership errors cause leaks, corruption, or races.  
**Mitigation:** RAII, narrow interfaces, sanitizers, fuzzing, bounded pools, explicit event ownership, and a CPU reference backend.

### Cross-backend numerical divergence

**Risk:** MLX and LibTorch produce meaningfully different hidden states or tokens.  
**Mitigation:** FP16 boundaries, one architecture, per-layer golden tests, reference mode, and explicit tolerances.

### Tensor conversion dominates decode

**Risk:** Materialization and host/device copies cost more than wire transfer.  
**Mitigation:** Measure conversion independently, preallocate and double-buffer, overlap operations, and pursue zero-copy only after correctness.

### Tailscale relay fallback

**Risk:** A relayed path adds unacceptable per-token latency.  
**Mitigation:** Warm and qualify links, expose connection state, and optionally require direct connectivity.

### MLX unified-memory pressure

**Risk:** The runtime starves macOS or becomes unstable.  
**Mitigation:** Explicit configured limits, measured peaks, OS headroom, and rejection of near-limit plans.

### Slow-stage bottleneck

**Risk:** A memory-feasible plan has poor throughput.  
**Mitigation:** Use memory only as feasibility and measured stage service time as the optimization input.

### Premature transport optimization

**Risk:** A custom protocol consumes effort before kernels and conversions are understood.  
**Mitigation:** Start with native gRPC, instrument copies and serialization, and replace only the data path when profiles justify it.

## 29. Architectural decisions

1. hLLM Runtime uses C++ native workers and a Python control plane.
2. Rust is not part of the initial architecture.
3. Native workers communicate directly; Python never relays activations.
4. The native stage-zero worker drives the autoregressive loop.
5. MLX uses the official C++ API.
6. CUDA uses LibTorch/ATen before custom kernels.
7. Workers are standalone processes, not Python extensions.
8. Tailscale is the secure network substrate, not the model-stage protocol.
9. Protobuf and gRPC define the control and MVP data protocol.
10. Safetensors is the only MVP weight format.
11. FP16 is the MVP boundary dtype.
12. Sampling occurs on the final native stage.
13. KV caches remain with their owning layers.
14. Model partitions are contiguous and immutable while requests are active.
15. Memory feasibility precedes performance optimization.
16. The Python planner uses measured worker and link profiles.
17. vLLM integration is a later backend or kernel source, not the foundation.

## 30. Open questions

- Which native Safetensors reader or internal audited implementation should be standardized?
- Should stage zero or the controller own the canonical sampling RNG seed sequence?
- What cross-backend numerical thresholds define official support?
- What unified-memory headroom should be recommended for each Mac memory tier?
- When should the runtime duplicate tied embeddings to improve stage balance?
- Does native gRPC remain acceptable after conversion costs are isolated?
- What workload presets should ship first: interactive chat, long-context chat, or batch generation?
- When should quantized weights be introduced, and may adjacent stages use different weight quantizations?

## 31. References

- [PyTorch C++ frontend](https://docs.pytorch.org/cppdocs/frontend.html)
- [Using the PyTorch C++ frontend](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html)
- [MLX C++ operations](https://ml-explore.github.io/mlx/build/html/cpp/ops.html)
- [MLX C API](https://ml-explore.github.io/mlx-c/)
- [MLX function export and C++ import](https://ml-explore.github.io/mlx/build/html/usage/export.html)
- [MLX custom extensions](https://ml-explore.github.io/mlx/build/html/dev/extensions.html)
- [MLX distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)
- [Safetensors](https://huggingface.co/docs/safetensors/index)
- [gRPC Python and streaming](https://grpc.io/docs/languages/python/basics/)
- [gRPC performance guidance](https://grpc.io/docs/guides/performance/)
- [Tailscale connection types](https://tailscale.com/docs/reference/connection-types)
- [Tailscale Serve](https://tailscale.com/docs/reference/tailscale-cli/serve)
- [Tailscale Grants](https://tailscale.com/docs/features/access-control/grants)
- [vLLM parallelism and scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [vLLM plugin system](https://github.com/vllm-project/vllm/blob/main/docs/design/plugin_system.md)
- [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- [Petals paper](https://arxiv.org/abs/2209.01188)
