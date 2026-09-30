# Qwen3-4B process-footprint soak — 2026-09-23

Status: the instrumented native MLX→CUDA soak completed **12 exact 256-token
continuations across three load/run/unload cycles** in the same worker processes.
The Mac's lifetime physical-footprint peak was **4.576 GiB**, with normal pressure
and no new swap-outs during the completed run. The production physical-fit rule
is unchanged; the measured overlap supports implementing a footprint-aware policy.

[Machine-readable evidence](qwen3-4b-footprint-results.json) contains per-cycle
snapshots, allowances, both attempts, guard summaries, source/binary identities,
cleanup checks and hashes of raw evidence. Raw records and reproduction scripts
are retained under `build/4b-footprint-20260923/`.

## Workload and identity

The user authorized a new memory window without a fixed deadline. Resource checks
preceded loading; configured caps plus the existing allowances passed preflight.
Each worker had an independent 20-minute timeout and the previous pressure/swap
guards. Other applications were left running.

- Pinned `Qwen/Qwen3-4B-Base`, revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`. All three shard hashes were rechecked on
  both machines and match the previous pinned payloads.
- F16 weights, execution, KV and boundaries; 15 MLX layers and 21 CUDA layers.
  Caps: 4.375 GiB MLX unified, 3.75 GiB CUDA host and 6.125 GiB CUDA device.
- 128 prompt tokens, 256 generated tokens, concurrency one, 384-token reservation.
  Four requests per deployment, then cancellation after three tokens, unload,
  and ten seconds idle. Repeat three times without restarting workers.
- Every full continuation matched the saved independent F16 oracle from the
  [previous qualification](qwen3-4b.md). No new oracle was generated in this pass.
- Native gRPC over bidirectional SSH forwards; this was not an HTTP load test,
  reverse-order run, context expansion or measured-placement qualification.
- Source base `66227433182b48f1d79fffd5ec112d33e47d44fc` plus the physical-memory
  instrumentation. All 65 compared native/protocol source files match across
  hosts. Worker-reported binary hashes match the binaries launched.

## Physical memory versus overlapping counters

All amounts below are GiB. The OS peaks are lifetime values, not reset per cycle.

| Deployment cycle | Exact continuations | Footprint lifetime peak after cycle | Footprint after unload and idle | MLX active after unload |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 / 4 | 4.575 | 0.879 | 0 |
| 2 | 4 / 4 | 4.576 | 0.890 | 0 |
| 3 | 4 / 4 | 4.576 | 0.892 | 0 |

All model/request reservations retired on both workers. Mac footprint after unload
increased by about 13.4 MiB from the first cycle to the third, mostly between the
first two. This is bounded reload evidence, not proof of an indefinite plateau.

| Mac accounting view | Peak / requirement |
| --- | ---: |
| OS physical-footprint lifetime peak | 4.576 |
| RSS lifetime peak | 3.970 |
| MLX allocator envelope (maximum of active+cache and allocator peak) | 3.635 |
| Existing RSS-plus-allocator requirement, including allowances/headroom | 9.616 |
| Diagnostic maximum-of-views requirement, including identical allowances/headroom | 6.284 |
| Available before the completed run | 8.344 |

Both requirements retain 10% safety, 0.25 GiB extra overhead and 1 GiB headroom.
The diagnostic comparison uses
`max(footprint lifetime peak, RSS lifetime peak, allocator envelope)`, followed by
those allowances. It is **not** a changed production acceptance result. The old
rule would still reject this serving evidence; the diagnostic comparison leaves
about 2.06 GiB beyond its existing headroom allowance under this run's initial
availability.

Apple documents physical footprint as accounting for dirty, compressed and swapped
pages, including accessed CPU and GPU resources on Apple Silicon; reclaimable clean
pages can remain in RSS without belonging to footprint. Thus footprint is not
simply another count of resident pages, and adding MLX allocations to it would
double-count accounted resources. See [Apple's memory accounting explanation](https://developer.apple.com/videos/play/wwdc2022/10106/).
The comparison retains RSS and allocator maxima as conservative cross-checks.

During the completed run, 280 Mac guard samples all reported normal pressure and
zero new swap-out bytes. Linux also recorded zero new swap-outs. Minimum sampled
availability was 3.519 GiB on the Mac and 2.898 GiB on Linux. These are sampled
system conditions, not guarantees against competing future workloads.

## What remained after unload

A `vmmap -summary` capture immediately after the third unload reported
**814.5 MiB of “Malloc Large (empty)”**, including 801.2 MiB resident/dirty and
13.2 MiB swapped. MLX reported no active bytes and approximately 65 MiB cached;
`vmmap` also showed about 68 MiB of IOAccelerator graphics regions. The snapshot
supports retained host allocator regions as the dominant residual footprint,
rather than live model reservations. It does not identify the allocating call
sites or establish that every retained byte can be released safely in-process.

The `vmmap` snapshot displayed a 970 MiB footprint versus about 913 MiB at the
subsequent idle snapshot. These were taken at different times after unload.
Keep the raw observations distinct; do not add potentially overlapping
VM categories together. The region breakdown is attribution evidence, not a
replacement peak measurement. Worker process exit releases the retained state.

## Sampling coverage and the first attempt

The first attempt generated one exact 256-token continuation, then deliberately
failed on a sampler error. `GetMemoryReport`/`GetMetrics` can wait behind the
control-service load lock, exceeding a five-second observation timeout. The
failure and logs are retained. Its guards did not trigger a resource stop.

The corrected diagnostic preserves load-time RPC gaps and continues sampling.
It recorded 1,259 observation rows, including 12 load-time timeout rows, with no
unexpected observation errors. An independent two-second OS sampler covers the
Mac worker during load and records its lifetime peak. RPC snapshots after load,
requests, cancellation and unload provide adjacent OS/allocator/reservation views.
Allocator fields may be unavailable while the MLX device mutex is occupied; they
are never substituted with zero. Observation reads are not atomic, and polling
intervals include RPC time. This pass does not change the production RPC locking.

The completed run lasted 572 seconds, including startup and cleanup. Final checks
found no hLLM model workers on either host and no CUDA compute processes. Both
owned workers were stopped normally by the diagnostic; no resource guard fired.

## Next implementation and remaining qualification

1. Add an explicit footprint-aware MLX fit policy for complete schema-1.3 evidence.
   Use OS lifetime footprint with RSS/allocator cross-checks and retain all current
   safety, overhead and headroom allowances. Preserve conservative handling of old,
   missing or incomplete telemetry. Record the chosen policy in fit results.
2. Keep fresh-worker deployment as the reliable way to return the entire allocation.
   If long-lived unloaded workers matter, separately test host allocator reclamation
   at unload against reload latency and correctness; the retained empty regions are
   now a concrete target. Do not flush caches during ordinary metrics collection.
3. Reprofile and qualify the chosen policy before expanding to 512/256, reverse-order
   serving or concurrency. This run covers three deployment cycles and one fixed
   prompt at concurrency one; it does not establish long-context or general capacity.

September 27 follow-up: the [opt-in policy and reusable tooling](footprint-policy.md)
are implemented and pass tiny-model checks. Fresh 4B policy acceptance and the
longer soak remain in the [overnight backlog](../overnight-backlog.md). The September
23 evidence above retains its original diagnostic interpretation and fit result.
