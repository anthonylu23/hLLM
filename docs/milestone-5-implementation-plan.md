# Milestone 5 implementation plan

Status: M5.1–M5.4 implemented, 2026-09-08; M5.5–M5.6 remain planned. The
[profiling workflow](milestone-5-profiling.md) and
[memory report](validation/milestone-5-memory.md) and
[timing report](validation/milestone-5-timing.md) record the delivered behavior
and limitations. See [current status](milestone-5.md) and [SPEC](../SPEC.md).

## Scope and acceptance contract

The agreed initial workload is **512 prompt tokens + 256 generated tokens,
concurrency 1, with capacity for 768 cached tokens**. Use F16 execution, KV and
boundary activations, fixed tokenized prompts, and disabled early stopping.
Create a dedicated workload fixture; preserve the existing 32,768-token capacity
example as a separate, unqualified capacity target.

Use the pinned Qwen3-0.6B checkpoint to develop and qualify the measurement pipeline.
Keep Qwen3-4B-Base as the larger target, gated on physical-fit measurements. Report
results per checkpoint: success on 0.6B does not establish 4B completion or fit.
The first acceptance sweep uses the direct MLX/CUDA Tailscale path. Relayed links
require separate profiles and qualification; do not reuse direct-path results.

Proposed performance objective: minimize request-start-to-last-token latency with
the model already loaded, including admission, stream setup and token feedback.
Record terminal cleanup latency separately and require cleanup to succeed.
For this fixed workload:

```text
predicted_generation_ms = predicted_TTFT_ms + sum(predicted_ITL_ms[255 decode steps])
selection_regret = measured_selected_generation_ms / measured_best_generation_ms - 1
pass when selection_regret <= 0.15, with memory and correctness gates satisfied
```

Use a dedicated objective configuration: TTFT weight 1, average ITL weight 255,
pipeline-period weight 0, memory-pressure weight 0. Memory remains a hard feasibility
gate and a deterministic tie-breaker. Preserve existing configured planner modes
and their objective settings. Pipeline throughput belongs to Milestone 6; the current
native loop waits for each downstream sampled token before the next decode step.
Report TTFT, average/p95 ITL and prediction error separately. A 15% selection-regret
pass is not a claim that every prediction is accurate within 15%.

## Current gaps

| Area | Existing implementation | Work required |
| --- | --- | --- |
| Enumeration | Both worker orders, every interior split | Retain and apply measured costs |
| Memory | Tensor/KV formulas and configured allowances | Assignment-specific load and execution envelopes, physical headroom |
| Performance | Directional latency/bandwidth plus fixed conversion estimates | Native compute, payload-specific conversion and transport, feedback/setup |
| Profiles | Worker/link schemas with a provenance label and timestamp | Compatibility keys, sample distributions, measurement identity |
| Telemetry | CUDA/MLX allocator snapshots; CUDA lifetime peak | Isolated measurement windows with explicit peak semantics |
| Qualification | Explicit split runners and reload checks | Uninstrumented timing, exhaustive sweep, automatic-plan comparison |

The existing `checkpoint_run.py` and `checkpoint_memory.py` query memory during
generation. Reuse their plan, correctness and cleanup logic, but do not use their
current timings as placement benchmarks. `QualifyLink` is declared in the protocol
but has no native control-service implementation.

## Implementation slices

### 5.1 — Profile contract and workload fixtures (implemented)

Define versioned memory, compute, conversion and directional transfer artifacts
under a new `python/hllm_control/profiling/` package. Keep immutable JSON artifacts
and raw samples first; a database/catalog is not required to finish this milestone.

Every artifact should record its digest, schema and profiler versions, source/build
revision, checkpoint/manifest identity, device identity, backend/toolchain/driver,
allocator configuration, execution/KV/wire dtypes, stage range and endpoint tensor
ownership, prompt/cache/batch shape, transport mode, warmup/repetition policy,
measurement time and concurrent-load conditions. Link records additionally identify
both endpoints, direction, connection type, payload size and stream policy.

Implement strict compatibility checks and explicit missing/incompatible/out-of-range
results. Missing counters are unavailable, never zero. Measured mode must not silently
substitute configured estimates. Start with exact shape/assignment matching; introduce
interpolation only within validated buckets, with its coverage recorded.

Add the 512/256/768 workload and the proposed fixed-workload objective. Validate
that workload capacity covers requested tokens and that measured mode initially
supports concurrency 1 and the runtime's supported dtypes. Store the actual workload
content/digest, not only its human-readable ID.

Likely files: `models.py`, `profiling/`, planner configuration, `examples/workloads/`,
`examples/profiles/`, and profile/placement protobufs plus `wire.py` where artifacts
cross the worker boundary. Keep legacy files readable and version changed contracts.

Acceptance: round-trip artifacts; reject wrong checkpoint, device, dtype, ownership,
cache shape, unsupported version and incomplete measurements; preserve existing
configured planning behavior. Test NaN/infinite timing inputs and deterministic
artifact identity. Do not add execution instrumentation in this slice.

### 5.2 — Assignment memory profiling and physical-fit gate (implemented)

Build an opt-in native profiling harness using the existing backend factory and
stage-loading path. Run in dedicated owned processes so peak resets cannot affect
ordinary requests or the documented lifetime semantics of `GetMetrics`.

For each candidate assignment, measure startup baseline, dry-load transient peak,
post-load residency, reservation/allocation, 512-token prefill, decode through the
target context, request cleanup and unload. Track load-time host staging and tied-head
verification scratch as well as device/unified memory. Distinguish cold fresh-process
runs from warm repeated-assignment behavior. Synchronize at measurement boundaries;
record resettable backend peaks locally, with external RSS/device observations and
their sampling limitations recorded separately.

Keep reservation accounting, allocator active/reserved bytes, process physical usage
and global available memory as distinct views. An observed whole-process peak is an
envelope check, not an extra term to add to weights/KV/workspace. Establish a documented
component accounting convention before replacing configured workspace allowances.
Pinned memory is also host memory; unified allocations share macOS capacity.

Feasibility must pass both the worker's admission accounting and conservative load/run
physical-envelope checks, including explicit OS/driver headroom and current available
memory. Sampled physical peaks are lower bounds, so retain a safety allowance. Label
unsupported telemetry and unmeasured assignments as unknown rather than safe.

After the 0.6B harness works, assess 4B candidate load/run fit using current host and
device availability. Existing nominal worker budgets and 0.6B RSS are insufficient.
Start from conservative candidates and stop on resource failure; do not repeatedly
attempt known unsafe assignments. A failed fit gate produces a capacity report and
leaves 4B qualification incomplete.

Acceptance: CPU fixture coverage for accounting/cleanup and peak isolation; MLX/CUDA
native checks for load and context peaks, reload stability, missing telemetry and
allocation failure. Produce one reproducible 0.6B memory artifact for each stage order
before expanding coverage across candidates.

### 5.3 — Native compute and conversion profiling

Extend the harness with optional timing collectors in CPU/CUDA/MLX stage execution.
Measure transformer layers or representative layer groups plus embedding, final norm,
LM head and sampling independently. Keep endpoint ownership explicit, including tied
embeddings. Retain whole-stage measurements to detect non-additive costs.

Measure prefill at 512 tokens and decode at cache lengths spanning the 255 subsequent
steps. Evaluate enough context buckets to validate any interpolation used in the
sum of decode costs. Separate native compute from F16 conversion, packing and
host/device boundary copies; cover the actual pageable/pinned transport configuration.
Keep sequence allocation and fixed setup outside recurring per-token compute costs.

Use backend-appropriate completed-work timing; MLX lazy evaluation and asynchronous
CUDA launches must not become apparent speedups. Detailed profiling must be opt-in.
Compare sums against whole-stage wall time without per-layer synchronization to
quantify instrumentation overhead before trusting the model.

Acceptance: distributions with warmups excluded, valid units and finite nonnegative
times; first/final ownership tests; backend numerical checks with profiling enabled;
whole-stage reconciliation and an instrumentation-overhead report on both devices.

### 5.4 — Directional native transport profiling

Implement a bounded profiling exchange using the same native gRPC serialization,
message framing and persistent stream behavior as inference. Complete or extend the
currently declared `QualifyLink` RPC with explicit target endpoint resolution,
payload sizes, repetition bounds, timeout/cancellation and result provenance.

Measure both MLX→CUDA and CUDA→MLX at the model's actual prefill and one-token payload
sizes, plus sampled-token feedback. Separate stream setup, conversion/copies,
serialization and transfer consistently with slice 5.3. Do not infer application
transfer cost from ping RTT or advertised bandwidth, or count an RTT again as a
separate feedback cost. Avoid cross-host timestamp subtraction; use local durations
and complete round-trip measurements with documented attribution.

Record path qualification before and after samples; invalidate mixed/path-changing
runs. Keep direct and relayed observations separate. Run memory observation and
transport timing as separate passes where sampling would affect results.

Acceptance: local CPU two-process test for payload handling, direction, timeouts and
cancellation, followed by actual tailnet measurements in both directions. A changed
payload, transport configuration or connection type must fail compatibility checks.

### 5.5 — Measured planner integration

Add explicit measured planning mode and profile-bundle input to `hllm plan`. Reuse
the existing two-order/all-split enumeration. Resolve compatible measurements for
every stage and boundary, apply admission/physical memory gates, then predict full
TTFT, context-dependent ITL and fixed-workload generation latency.

Treat measured, unknown and infeasible candidates distinctly in reports. Explain
missing coverage, rejection reasons, cost components, profile IDs/digests, runner-up
plans and deterministic ties. Attach the workload/profile bundle identities to the
versioned planning artifacts so the selected plan is reproducible. Account for any
new plan fields in hashing, serialization and native validation.

Refresh capabilities and physical headroom before activating a measured plan; an
offline profile does not guarantee future free memory. If validation or loading
fails, clean up and report the candidate failure without claiming successful placement.
Automatic fallback/replanning can be added later; avoid introducing a hidden retry loop.

Acceptance: synthetic profiles that deliberately change the winning split/order,
asymmetric endpoint/conversion costs, context-dependent decode, boundary limits,
physical-load failure despite feasible reservations, incompatible profiles, incomplete
coverage, no feasible plan, deterministic hashes and legacy-mode regressions.

### 5.6 — Exhaustive qualification and milestone report

Add a resumable sweep runner around the existing explicit-plan/session utilities.
For L layers there are `2 * (L - 1)` two-worker candidates. Independently account for
every candidate: benchmark feasible placements, retain supported memory exclusions,
and record unknown/OOM/correctness failures. Planner rejection alone cannot establish
the ground-truth feasible set; missing candidate evidence leaves qualification incomplete.

Freeze profiles and automatic selection before the validation sweep. Use independent
timing repetitions so the planner cannot simply select from its own validation data.
Use at least two warmups and five measured repetitions per candidate initially, with
seeded shuffled/interleaved candidate order and reference candidates repeated to
detect drift. Start each candidate's memory qualification from a fresh process; test
warm reload behavior separately. Preserve raw samples and resume only compatible runs.

Time requests without in-loop telemetry RPCs. Record client token arrival times and
native timing where available so buffering is visible. Use identical prompt fixtures,
budgets, dtypes, output length and load conditions across splits. Generate an independent
reference for the 512-token fixture; existing short-prompt oracle data does not cover it.
Predeclare numerical/token criteria using the existing backend qualification policy,
record divergence, and never count an incorrect or incomplete run as a speed win.

Compare median generation latency for the selected plan with the best measured feasible
candidate. Publish repeat counts, spread and uncertainty; increase repetitions when
noise makes the 15% threshold inconclusive. Report prediction errors separately for
TTFT, ITL, total latency and memory. Recheck correctness, request/model cleanup,
cancellation, deadlines and post-failure recovery for the selected deployment.

Acceptance: reproducible, redacted report with complete candidate coverage and regret
at most 15%, plus memory/correctness/cleanup gates. Qualify 0.6B first and repeat for
4B only after the fit gate passes. Keep checkpoint-specific completion claims explicit
in `README.md`, `SPEC.md` and the milestone status document.

## Delivery order and validation

Implement 5.1 first, then 5.2, 5.3, 5.4, 5.5 and 5.6 as reviewable changes.
Slices 5.3 and 5.4 share the timing-attribution contract from 5.1; planner integration
depends on all three kinds of measurements. Keep every slice usable through a small
fixture before running a complete hardware sweep.

For Python/schema changes run pytest, Ruff, Pyright and the generated-proto check.
For native changes run relevant CTest coverage and the affected backend numerical,
lifecycle and process tests. Full hardware timing sweeps are qualification artifacts,
not CI tests with hard speed thresholds. Update implementation status after each slice.

M5.5 measured integration and the M5.6 independent runner are implemented; see the
[placement workflow](milestone-5-placement.md) and
[validation/gates](validation/milestone-5-planner.md). Next are current-build profile
coverage, production setup calibration, and the full 0.6B hardware sweep.
M5.1–5.4 are implemented; their
[profiling workflow](milestone-5-profiling.md),
[memory evidence](validation/milestone-5-memory.md), and
[timing evidence](validation/milestone-5-timing.md) document scope and limitations.
The 4B endpoint-only MLX load-admission lower bound still exceeds observed Mac
availability, so its physical profiling and inference qualification remain pending
a successful preflight. Transport-only payload qualification does not establish 4B fit.
