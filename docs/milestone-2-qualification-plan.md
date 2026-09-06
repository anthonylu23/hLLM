# Plan: mixed CPU/CUDA qualification

Status: implemented on the qualification stack, 2026-09-06. PR #7 establishes
mixed execution; PR #8 adds pinned staging; `codex/cuda-failure-qualification`
adds failure/memory qualification. See the [validation report](validation/mixed-cpu-cuda.md)
for measured results and limitations. Upstream PR #5 (integration) and PR #6
(CUDA execution) remain dependencies. Review and merge in stack order.

## Outcome and scope

Establish that two native processes can execute one model across CPU and CUDA, in either
stage order, with correct outputs, bounded memory and predictable cleanup. Use the existing
stage interface and wire protocol. Start with both processes on the Linux CUDA machine over
loopback, so backend/transport failures can be distinguished from cross-machine issues.
Cross-machine/Tailscale qualification remains Milestone 4.

Use tiny Llama and Qwen3 fixtures, one sequence and one active request per worker. Keep the
existing model registry, opaque sequence state, native decode loop and Python controller.
No new model family, per-stage precision schema, batching scheduler, generic graph engine,
public serving API or custom kernels belongs in this phase.

## Constraints established from the current implementation

- `DeploymentPlan.execution_dtype` is global. CPU supports F32 computation; CUDA supports
  F32/F16. Mixed deployments therefore use **F32 execution and F32 KV on both workers**,
  with FP16 boundary bytes. F16 CUDA execution remains covered by standalone CUDA tests.
  Using CPU F32 with CUDA F16 requires a separate precision-design decision.
- `StageBackend::execute` returns completed host data or a token. Preserve that contract.
  An internal asynchronous copy/event does not imply overlap across decode steps, since
  each step depends on the downstream sampled token.
- `BoundaryActivation` owns a byte vector and gRPC serializes its payload. Backend-owned
  pinned staging can be added without changing the transport API, but host copies remain.
  Do not describe this as zero-copy transport or assume it improves tiny decode latency.
- CUDA attention currently materializes dense score/softmax matrices. Memory estimates
  must include quadratic workspace and conversion/staging temporaries.
- The planner uses configured workspace and an independently configured KV dtype. Its
  pinned-budget check currently substitutes for the host check. Qualification profiles
  must match runtime F32 KV, and pinned usage must pass both host and pinned budgets.

## PR A — establish mixed-process correctness

Generalize the existing process-test launcher to select a binary and arguments per worker,
while preserving the CPU-only suite. Reuse checkpoint, plan, event and cleanup helpers.
Register a hardware-required mixed suite only in CUDA-enabled builds.

Run CPU → CUDA and CUDA → CPU for every valid contiguous split of the tiny four-layer Llama
and two-layer Qwen3 fixtures. Test F32 execution, FP16 boundaries, F32/F16/BF16 storage,
and tied/untied heads using a bounded matrix rather than multiplying every lifecycle test
by every storage combination. Include prefill, single-token decode, stop IDs, repeated
requests and a 256-token generation case with adequate fixture context capacity.

Compare against a CPU/CPU deployment with the same split and FP16 boundary to isolate the
backend change. Retain the existing unsplit oracle as a separate reference; split rounding
must not be confused with a GPU regression. Require matching greedy tokens on the selected
fixtures. Diagnose any disagreement with the numerical trace and existing tolerances;
do not automatically widen tolerances to make the process tests pass.

Add runnable loopback CPU/CUDA profiles and instructions for prepare → plan → generate.
Use F32 KV estimates and conservative CUDA workspace settings. Test the actual CLI flow,
not only hand-constructed plans. Retain explicit warnings in reports that these profiles
are qualification estimates, not full-checkpoint performance measurements.

Likely files: `tests/integration/` shared helpers, a new mixed suite under `tests/cuda/`,
`tests/cpp/CMakeLists.txt`, `examples/profiles/`, and the Milestone 2 documentation.

Acceptance: both orders and all fixture splits match the reference; generation drives the
native RPC path; requests release reservations; the original CPU and CUDA suites remain green.

## PR B — bounded pinned boundary transfers

Add an explicit CUDA boundary-transfer setting, proposed CLI spelling
`--boundary-transfer-mode pageable|pinned`. Keep pageable as the initial default. Pinned
mode requires a configured pinned budget and rejects insufficient capacity rather than
silently changing transfer mode. The CPU executable remains independent of CUDA.

Keep buffer/event ownership inside the CUDA backend. Allocate a bounded staging buffer
for the active sequence at reservation time, sized to the maximum legal boundary payload
for that reservation, capped by the existing transport limit. A single-stage deployment
needs no activation-boundary staging. Reuse the sequence buffer across decode calls and
release it when the sequence is retired. Prefer explicit lifetime/accounting initially;
a cross-request pinned-buffer cache is unnecessary for correctness.

For input: copy validated wire bytes into pinned staging, enqueue host-to-device transfer,
and order model execution after the transfer. For output: convert on device to FP16,
enqueue device-to-host transfer, wait for its completion event, then create the owned
boundary payload. Start with the existing guarded stream; add a separate transfer stream
only if a measured use case warrants the additional dependencies.

Never overwrite/release staging while an event still refers to it. Cancellation and
exceptions must drain outstanding work before retiring sequence state. Preserve completed
`execute()` results and the runtime's ownership rules. No background operation may outlive
its stage/request accidentally.

Charge staging to both host and pinned workspace, in addition to the device and pageable
transport/conversion copies. Validate peak memory before allocation. Fix the planner to
check pinned transport against both applicable budgets, with tests where the host cap is
smaller than the pinned cap. If a required budget is absent, reject the pinned qualification
configuration rather than assume unlimited memory. Keep runtime admission authoritative.

Likely files: new CUDA-local staging/event classes, `cpp/src/cuda/stage.cpp`, factory and
worker argument wiring, planner budget checks, CUDA transfer tests and qualification profiles.
The common stage API and activation protobuf remain unchanged.

Acceptance: PR A passes in both modes; exact-fit/one-byte-short budgets behave correctly;
allocation failure rolls back; repeated use preserves payload bytes; pending copies prevent
buffer reuse/release; reported pinned usage is included once in total host-plus-device usage.
Record prefill/decode transfer timings for context, with no required speedup threshold.

## PR C — failure handling and memory qualification

Extend the mixed harness with controlled synchronization points for fault tests. Prefer
barriers/test-only hooks or a controlled peer over timing-only sleeps. Production behavior
must not depend on test hooks, and fault-injection controls must not be exposed by normal
worker RPCs.

Exercise client cancellation, control cancellation and deadlines during prefill/decode;
downstream admission rejection; disconnects and worker loss in either position; malformed
or stale messages; unload during active work; repeated requests; and reload after recoverable
failure. Use deterministic allocation-failure injection for rollback tests instead of
exhausting the user's GPU. Distinguish recoverable allocation/request failures from fatal
CUDA-context errors, for which restarting the worker may be necessary.

For each case assert the expected gRPC status, no replay or false completion, cleanup on the
surviving worker, and a successful subsequent request where recovery is supported. Once
execution has stopped, active requests and cache/workspace reservations return to zero;
weights remain until unload. Hardware kernels need not be preemptible, so cleanup deadlines
must account for draining submitted work.

Run repeated request/unload cycles and capture host RSS, device payload usage, pinned bytes,
and CUDA allocator allocated/reserved memory. Compare allocator values after warm-up;
reserved cache need not return to zero, but must not grow without bound across equivalent
cycles. Separate model-owned accounting from context/framework/allocator overhead. Add only
the diagnostic counters needed for these assertions; a general profiling service remains
Milestone 5 work.

Reconcile qualification profiles against runtime reservation formulas and observed peaks,
including quadratic attention workspace and F32 KV. Document workload/context, toolchain,
GPU, budgets, test counts and limitations in a reproducible validation report.

Acceptance: both worker orders pass the failure matrix; pinned lifetimes are verified;
recoverable failures permit a new request; no unbounded memory growth is observed; all
existing CPU/CUDA numerical and process suites stay green. Run CUDA memory checking on
small transfer/lifecycle cases if the supported tool is available, and report its actual
coverage separately from ordinary tests.

## Completion and next decision

This phase is complete when mixed execution is reproducible, both transfer modes have
correct ownership/admission, failure cleanup is qualified, and documentation includes the
measured limitations. It establishes a small-model heterogeneous runtime, not a claim of
full-model throughput or arbitrary checkpoint support.

Then choose a full-checkpoint workload that fits the currently available CPU/GPU memory,
leaving headroom for F32 resident CPU weights and dense attention. Qwen3-4B-Base remains the
project target, but this plan does not presume it will fit the shared Linux host. A smaller
compatible checkpoint or a deliberately short context may be the appropriate first run.
Do a read-only memory/workload assessment before scheduling that run. MLX and cross-machine
execution retain their existing milestone boundaries.
