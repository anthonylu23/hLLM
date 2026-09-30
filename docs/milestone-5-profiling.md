# Measured profiles and native probes (M5.1–5.4)

Measured-profile contracts and isolated memory, compute/conversion and directional
transport profilers are implemented. The [placement workflow](milestone-5-placement.md)
consumes these profiles for automatic selection and independent qualification.
WAN performance acceptance is [deferred](validation/milestone-5-wan-closeout.md).

## Profile contracts

`hllm_control.profiling.models` defines separate, versioned JSON artifacts for memory,
compute, conversion and directional transport measurements. Existing worker/link YAML
and configured planner behavior are unchanged. Artifacts remain separate from
configured planner inputs; additive diagnostic RPC fields support the link probe.

Each artifact includes a canonical SHA-256 digest, schema/profiler versions, the
prepared manifest digest, hashes of the actual config and all checkpoint payloads,
stage ownership, workload content and digest, dtypes, input generator identity,
transport mode, device identity, binary digest, source revision, compiler, backend,
driver and allocator configuration. Measurement conditions record an aware timestamp,
warmups, measured repetitions, process policy and competing workloads.

Files are created exclusively. The runner preserves the native log, raw samples,
input specification, preflight decision and physical-fit assessment alongside the
sealed artifact. It refuses to overwrite an existing run. Failed native runs retain
partial samples and an error; a preflight rejection writes its decision without
starting a native process. A failure before backend initialization retains logs but
cannot produce a complete backend identity or sealed measurement.

`check_compatibility()` distinguishes missing, incompatible, out-of-range, incomplete
and compatible profiles. Matching initially requires exact assignments and workloads;
there is no interpolation or configured fallback. Timing-profile lookups additionally
require the complete measurement `scope` (the measurement JSON without `samples_ms`),
so direction, payload, component, context and endpoint environment cannot be ignored.
Callers may impose an explicit maximum measurement age. These are compatibility
primitives used by measured placement; `hllm plan --profile-bundle` consumes a sealed
bundle of compatible evidence. See the placement workflow for bundle construction.

The original workload is `examples/workloads/milestone-5.yaml`: 512 prompt tokens,
256 generated tokens, concurrency 1, 768-token capacity and F16 execution/KV/boundary.
The objective fixture uses measured mode with TTFT weight 1 and ITL weight 255.
The approved acceptance target uses `milestone-5-mixed.yaml` and
`milestone-5-mixed-objective.yaml` instead: F16 resident weights, F32 execution/KV,
and F16 boundaries, with the same request dimensions and objective weights.
`interactive.yaml` retains its separate 32,768-token capacity target.

## Native process probe

CMake builds `hllm-profile-memory-cpu`, and optionally `hllm-profile-memory-mlx` or
`hllm-profile-memory-cuda`. They use the same backend factory, shared dense loader,
sequence allocation and stage execution as workers. No model executes in Python.

A fresh process probes one assignment, then optionally repeats it without flushing
caches. Each cycle records baseline, dry-load, allocation, prefill, all subsequent
decode steps, sequence cleanup and model unload. Warmup cycles are retained in raw
memory observations because cold-load peaks still matter for safety. Timing summaries
exclude warmups; timing artifacts retain raw samples from every cycle.

The native harness uses token ID zero for first stages and zero-valued F16 boundary
inputs for later stages. It exercises real dense operations, parameter loading, tied
head verification, cache allocation and the requested shapes. It does not run a
cross-worker continuation or establish token correctness. Backend numerical and
process regression tests cover execution correctness separately. Keep this input
identity distinct from future reference-driven profiles.

CUDA and MLX synchronize before resetting allocator peaks, solely inside this owned
profiling process. Phase peaks include active memory at phase entry and exit: MLX's
reset counter alone can otherwise report zero for a phase that only frees memory.
Ordinary `GetMetrics` retains allocator lifetime-peak semantics and never resets peaks.
Unsupported CUDA allocators, including `cudaMallocAsync`, report unavailable counters.

RSS high-water marks are process-lifetime observations, not resettable per-phase host
peaks. The parent also samples RSS and CUDA process usage independently. Those samples
are lower bounds on physical peaks and can miss short transients. Memory-probe durations
include observation overhead and must not be used as performance profiles.

### Process physical footprint

Memory artifacts with native process observations use schema **1.3**, profiler
**0.3.0**, for both uniform and mixed precision. Schema 1.0–1.2 artifacts still
validate with their original hashes; missing observations are not backfilled.
Every phase records `process_memory` immediately after reading allocator counters:
PID, `observed_at_unix_ns`, RSS, RSS lifetime peak, physical footprint and physical
footprint lifetime peak. The legacy RSS fields use that same observation.

On macOS, `proc_pid_rusage(RUSAGE_INFO_V4)` supplies resident size and physical
footprint, including OS-accounted compressed memory. `getrusage` supplies RSS high
water. On Linux, RSS comes from `/proc/self/statm` and `getrusage`; footprint is
unavailable. Failed or unsupported counters are JSON `null` / absent optional
protobuf fields, never an invented zero. These reads do not reset OS peaks, flush
allocator caches or synchronize device execution.

Workers expose the same `process_memory` message in `GetMetrics`, including before
load and after unload. The serving soak already saves these responses alongside
allocator and reservation snapshots during requests and after unload. HTTP `/metrics`
exports present counters as `hllm_worker_process_{rss_bytes,rss_lifetime_peak_bytes,
physical_footprint_bytes,physical_footprint_lifetime_peak_bytes}` plus
`hllm_worker_allocator_{active_bytes,cached_bytes,peak_bytes}`, labeled by stage.
Older workers without the new message remain supported.

These are adjacent reads, **not an atomic snapshot**. OS peaks cover the entire
process lifetime, including earlier loads and requests; they are not per-phase
peaks. Timestamped current values help compare load, request cleanup and unload,
while lifetime peaks retain transients between observations. A current value after
unload can include retained caches and runtime state even when logical reservations
are zero. Footprint is not additive with RSS or MLX allocator bytes, and their
differences alone do not establish allocation ownership or total system pressure.

The conservative fit gate remains the default. The [instrumented 4B reload run](validation/qwen3-4b-footprint.md)
includes system pressure/swap, available memory and retained-state evidence.
An explicit footprint policy is now implemented as described below; its fresh
full-checkpoint qualification is in the [overnight backlog](overnight-backlog.md).
The [instrumentation validation](validation/process-memory.md) covers small native
CPU/MLX fixtures and Linux counter availability; it is not a new 4B qualification.

## Admission and physical envelopes

`assess_fit()` evaluates native admission and physical memory separately. It returns
`safe`, `unsafe` or `unknown` for the recorded assignment and explicit physical budgets.

- A completed load/run proves admission only under its recorded caps. Lowering a cap
  requires a new probe; observed reservation values alone cannot prove a lower load cap.
- CPU physical observations use RSS, including its process high-water mark.
- CUDA uses process GPU observations separately from allocator counters and host RSS.
  Missing physical GPU observations or phase allocator peaks make fit unknown.
- MLX defaults to `conservative-v1`: RSS plus allocator residency, an intentionally
  overcounted upper bound. Opt-in `footprint-v1` uses the maximum of OS lifetime
  footprint, RSS and allocator envelopes when complete telemetry qualifies it.
  The raw views remain separate in the artifact.
- Every physical envelope adds a safety fraction and explicit overhead for costs
  absent from the isolated probe, including gRPC buffers. It must leave the requested
  headroom within current available memory. Pinned bytes remain a subset of host bytes.

Before launch, the runner conservatively checks the configured load/run caps plus
physical allowances against current availability. Linux uses `MemAvailable`; macOS
uses free plus inactive pages as a conservative availability observation. CUDA uses
`nvidia-smi` free device memory. Neither nominal capacity nor successful 0.6B runs
establish that a larger checkpoint will fit. No existing applications are stopped.

A `safe` assessment applies to the observed conditions and specified allowances; it
is not a new runtime RSS/VRAM cap or a guarantee about later competing workloads.
Refresh headroom before deployment. M5.5 must resolve compatible profiles and apply
these gates before ranking candidates.

### Opt-in MLX footprint policy

Pass `--mlx-fit-policy footprint-v1` to `hllm profile-memory`, or
`mlx_policy=MlxFitPolicy.FOOTPRINT` to `assess_fit()`. The opt-in requires a complete
schema-1.3 fresh-process/reload profile: positive current/lifetime RSS and footprint
counters in every phase, consistent process identity, nondecreasing observation
timestamps and lifetime peaks, and allocator observations including execution phase
peaks. A peak must cover its adjacent current value. Missing, zero, inconsistent,
old-schema or incomplete evidence falls back to the conservative calculation and
records the reason. Incomplete execution and missing required allocator peaks still
produce `unknown` under the existing rules.

The selected physical peak is:

```text
max(OS footprint lifetime peak, RSS observations/high water,
    allocator peak, allocator active + cached)
```

The same safety fraction, extra overhead and headroom are then applied. Native
reservations, configured-cap preflight and the requirement to reprofile lower
admission caps remain unchanged. CPU and CUDA calculations are unaffected.
Fit reports record `requested_mlx_policy`, the actual `policy` (for example
`mlx-footprint-max-v1` or `mlx-rss-plus-allocator-v1`) and any `policy_notes` explaining
fallback. An opt-in request alone is not evidence that the footprint formula applied.

Measured bundle worker bindings and native sweep worker configurations accept
`mlx_fit_policy: "footprint-v1"`. It is included in their content digests and reused
at fresh activation and independent exclusion validation. Omitted policy fields
retain the conservative default and historical bundle serialization. New fit fields
are optional when reading historical fit reports. Changing policy requires resealing
the bundle/replanning or freezing a new sweep; do not edit frozen identities in place.

Tiny native CPU/MLX checks cover the implementation; no new full-checkpoint fit,
larger-context, reverse-order or concurrency qualification is claimed yet.

## Reproduce

Build as documented in the CPU, MLX or CUDA milestone instructions. Prepare an actual
checkpoint and a valid explicit two-stage plan. For example, on the machine owning
the first MLX stage:

```bash
uv run hllm profile-memory \
  --manifest build/qualification/manifest.json \
  --plan build/qualification/mlx-cuda.plan.json --stage-index 0 \
  --workload examples/workloads/milestone-5.yaml \
  --model-root build/models/Qwen3-0.6B \
  --binary build/native/mlx/cpp/hllm-profile-memory-mlx --backend mlx \
  --unified-bytes 4294967296 \
  --source-revision '<revision plus source-snapshot digest used for this binary>' \
  --concurrent-load '<observed other workloads>' \
  --cycles 2 --output build/qualification/mlx-first-memory.json
```

The example cap is subject to the live preflight; select a cap that covers native
load admission and fits current physical availability. The recorded qualification
uses tighter caps, documented in its report. CUDA uses `--host-bytes` and
`--device-bytes`; pinned mode additionally needs `--pinned` and `--pinned-bytes`.
CPU requires F32 execution/KV in its plan/workload. All native boundaries require F16.

Repeat for stage 1 on its owning machine, then repeat for the reversed plan. Give
all runs distinct outputs. `--warmup-cycles`, `--cycles`, `--timeout`, physical headroom
and extra overhead are explicit controls. The command exits 2 for an incomplete
measurement and 3 for a completed measurement whose physical-fit assessment is not
safe. Inspect the saved reports to distinguish unknown measurements from exclusions.

Run contract tests through the normal Python suite. CTest registers
`MemoryProfile-cpu`, `MemoryProfile-mlx` and `MemoryProfile-cuda` when those backends
are built. They cover both stage roles, reloads, cleanup and allocation failures;
the CUDA test also exercises unavailable telemetry with `cudaMallocAsync`.

See the [M5.1–5.2 qualification report](validation/milestone-5-memory.md) for initial
full-checkpoint memory evidence and the larger checkpoint's preflight assessment.

## Paired native compute and conversion (M5.3)

`hllm profile-compute` takes the same assignment, checkpoint, workload, backend,
capacity and source-identity arguments as `profile-memory`. Select the matching
`hllm-profile-compute-{cpu,mlx,cuda}` executable. It defaults to two warmup cycles
and five measured cycles. Output consists of the sealed JSON artifact, native
JSONL records, spec, preflight, process log and `.summary.json`.

The native process loads the assignment once. Each cycle runs an ordinary pass and
an instrumented pass with separate sequence states and identical synthetic inputs;
pass order alternates. Boundary bytes or sampled tokens must match exactly at every
step. Sequence allocation is recorded separately from execution. Every decode
position is measured, from context 512 through 766 for the initial workload; there
is no interpolation. The prefill call produces output token one, followed by 255
decode calls. Capacity remains 768.

CPU, CUDA and MLX expose an opt-in `execute_profiled` collector. It records input
embedding or incoming F16 conversion/copy, each owned transformer layer by global
index, final norm/head/sampling or outgoing F16 conversion/copy, and runtime
validation/overhead. CUDA synchronizes its execution stream; MLX materializes lazy
outputs and synchronizes before recording completed work. Ordinary execution adds
no profiling clocks, record allocation or per-layer synchronization.

Schema 1.1 / profiler 0.2.0 `compute-run` artifacts retain all paired raw samples,
including warmups. Summaries exclude warmups and retain each context's distributions
and ordinary-versus-instrumented comparison. Component sums reconcile with the
instrumented stage wall time, including an explicit residual. The overhead report
compares that sum against ordinary stage time. Synchronized component measurements
are diagnostic inputs; they cannot be treated as additive uninstrumented costs
without checking this overhead and the later end-to-end validation. Whole-stage
measurements already include GPU boundary conversion; never add it twice.

Physical telemetry runs separately: the compute parent does not poll RSS or
`nvidia-smi` during execution. Preflight and native capacity checks still apply.
A compute profile proves neither current physical fit nor model correctness; pair
it with memory qualification and the independent numerical suites. No serving
worker can enable timing or reset allocator peaks through an RPC.

## Directional native gRPC exchange (M5.4)

Build `hllm-profile-link` on each host and start an isolated probe process. For
example, on the source host, using numeric addresses reachable by its peer:

```bash
uv run hllm serve-link-probe \
  --binary build/native/dev/cpp/hllm-profile-link \
  --worker-id source --listen 100.64.0.1:50543 \
  --peer target=100.64.0.2:50543 \
  --source-revision '<revision+source-snapshot>' --config build/source-probe.json
```

Run the analogous server on the target, with worker IDs and addresses reversed.
Each server is a foreground process; Ctrl-C stops it. The config is created
exclusively and includes the executable hash. The server adds compiler, gRPC,
Protobuf, OS and host identity. Its peer allowlist resolves target IDs; callers
cannot supply an arbitrary destination address. It never loads a model.

Serialize directional measurements that share either probe. One probe can perform
only one source or target measurement at a time; contention returns
`RESOURCE_EXHAUSTED`. Bad request limits/payloads return `INVALID_ARGUMENT`.
The source preserves downstream RPC status codes and details, including transport
`UNAVAILABLE`, instead of labeling failed exchanges as bad requests. A missing or
malformed response/termination acknowledgment returns `DATA_LOSS`; a mismatched
peer identity returns `FAILED_PRECONDITION`. An application timeout returns
`DEADLINE_EXCEEDED`, and explicit cancellation releases the active probe slot.
Direct target stream I/O may require gRPC transport cancellation to unblock; the
source still classifies its own elapsed application deadline precisely.

On the source host, measure the actual model width (`1024` for Qwen3-0.6B, `2560`
for Qwen3-4B-Base), then repeat on the other host with the direction reversed:

```bash
uv run hllm profile-link \
  --source-endpoint 100.64.0.1:50543 --target-endpoint 100.64.0.2:50543 \
  --target-worker-id target --hidden-size 1024 \
  --prompt-tokens 512 --output-tokens 256 \
  --warmup-cycles 2 --cycles 5 --timeout 120 \
  --tailscale /path/to/tailscale \
  --concurrent-load 'Describe observed competing work' --output build/link.json
```

The existing `WorkerControl.QualifyLink` signature is extended additively with
bounded workload parameters and a typed `LinkProfile.qualification` result.
`GetLinkProbeInfo` identifies each isolated probe. Ordinary serving workers leave
these diagnostics unimplemented. Legacy latency/bandwidth/conversion scalars in a
qualification response are unmeasured; the controller requires the qualification
field and does not convert the response into a configured planner profile.

All timed traffic is native `StageExecution.Execute`: the same `SequenceOpen`,
F16 `TensorEnvelope`, `SampledToken` and termination messages as inference. The
shared boundary codec performs the production payload copy and validation on both
paths. Synthetic finite F16 values and deterministic token-zero replies replace model
compute at the receiver; serialized activation and feedback sizes are retained. The
controller requests the run and collects its result; Python never carries timed
activation traffic.

Attribution is explicit:

- Sender encode includes production boundary validation and the Protobuf payload
  copy. It excludes device-to-host conversion, which belongs to compute profiling.
- Native RPC round trip starts after sender encode and ends after validating the
  sampled-token reply. It includes gRPC serialization/deserialization, transfer,
  receiver boundary decoding/copy/validation, and return feedback. It is not a
  one-way latency and must not be divided by two or charged feedback a second time.
- Channel readiness is measured once. Each persistent stream has separate setup
  and teardown samples. Setup uses a diagnostic initial-metadata readiness barrier
  after `SequenceOpen`; production overlaps open with its first computation and
  has no such wait. This setup sample must not be added blindly to predicted TTFT.
- Host-to-device conversion and all transformer/head work are excluded from the
  link exchange. The profile does not infer an independent bandwidth or fixed
  latency from two payload sizes.

Requests are capped at 32 total cycles, 1,024 output steps, 8 MiB per activation,
256 MiB aggregate activation traffic and a 120-second deadline. Only one exchange
can occupy a probe process. A watchdog bounds stalled reads and forwards control
cancellation to the peer. Invalid identities, payload shapes, phases, non-finite
activations and incomplete sequences are rejected.

Qualification observes the route on the native source host. Tailscale ping and
peer status are captured before and after the run, with status observations every
0.5 seconds during it. The Mac collector uses `posix_spawn`-compatible subprocess options so route queries
do not fork live gRPC threads. Direct endpoint changes, mixed connection types, inactive
peers, unknown ping formats and observation failures invalidate reuse. Ping is
path evidence only, never an application timing sample. It can warm NAT traversal;
channel readiness therefore covers a new gRPC connection on the qualified path,
not cold tailnet discovery. Sampling cannot rule out a
transition shorter than its interval; raw observations and the interval remain in
the artifact. Numeric loopback runs have an explicit, separate path scope.

`LinkArtifact` is a separately sealed schema 1.1 artifact: transport-only
measurements bind to both host builds, endpoints, exact payload/workload, framing,
stream policy and qualified path, rather than a checkpoint they never load.
`check_link_compatibility` also checks age and direction. Changed builds,
workloads, payload widths or qualified routes cannot reuse an old measurement.
Completed-but-path-invalid and failed runs remain reviewable, with summaries and
error evidence; they are not usable planner inputs.
