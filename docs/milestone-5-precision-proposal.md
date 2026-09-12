# M5 precision target — approved for implementation

The isolated diagnostic now matches all 256 reference tokens at all 54 placements
(13,824 teacher-forced outputs). The [evidence audit](validation/milestone-5-planner/kv-f32-complete.json)
verifies coverage, raw hashes, shared inputs, cached prefixes and configured caps.
It does **not** establish production acceptance. Current production remains at
51/54 exact placements in the diagnostic matrix.

## Approved target

The user approved an explicit mixed-precision mode for the Qwen3-0.6B M5 qualification:

| Component | Current target | Proposed target |
|---|---|---|
| Resident weights | F16 | F16 |
| Layer execution | F16, with existing F32 intermediates | F32 intermediates throughout |
| KV cache | F16 | F32 |
| Transmitted activations | F16 | F16 |
| Correctness | Exact F32-reference greedy tokens | Unchanged |
| Workload | 512 prompt + 256 output, concurrency 1, capacity 768 | Unchanged |
| Memory caps and safety margins | Existing fixed caps | Unchanged |

This explicitly replaces the F16 execution/KV target for this qualification. It
does not claim that F16 KV passed, relax token matching, introduce a tie tolerance,
change the checkpoint/reference, or qualify the separate 4B and 32K targets.
Existing F16 behavior should remain available; the new mode is explicit and opt-in.

## Memory and performance implications

For 28 layers, 8 KV heads, head dimension 128 and capacity 768, total KV allocation
across a complete partition increases from 84 MiB to 168 MiB. The incremental
84 MiB is distributed according to the split. MLX cache-copy workspace and backend
allocator overhead require separate accounting; this is not a peak-memory bound.

The diagnostic casts weight matrices to F32 for computation while retaining F16
resident weights. Transient casts, especially the output head, consume additional
workspace and bandwidth. F32 arithmetic may materially reduce throughput. Neither
its performance nor its physical peak memory is qualified yet. All configured
caps stay fixed; the implementation must reject configurations that cannot fit.

## Implementation and qualification work

1. Represent resident weight precision independently of execution precision.
   Add an explicit `weight_dtype` to settings, plans and profile identities;
   execution and workload KV precision remain F32 for this mode. Preserve legacy
   defaults and hashes when the new mode is absent. Use a versioned plan contract
   for mixed precision, and reject unsupported precision combinations rather than
   silently reinterpreting existing F16 plans.
2. Advertise and validate supported precision combinations in workers. Carry the
   new identity through protobuf conversion, canonical plan hashes, planning,
   activation, profile matching, native probes and frozen sweep execution.
   Old measurements cannot satisfy a request for the new mode.
3. Implement F16 weight residency, F32 computation and F32 KV allocation in MLX and
   CUDA. Account for actual weight, cache, cast and workspace allocations. Keep
   F16 transport. Do not promote the copied diagnostic binaries into production.
4. Add meaningful tests for mixed-precision serialization and hash integrity,
   capability rejection, resident-weight/cache accounting, stale-profile rejection,
   and numerical behavior against an independent small fixture. Run the affected
   Python/native suites and required checks, then full-checkpoint serving checks.
5. Collect fresh affected memory and compute profiles for both backends, request
   setup calibration, link/environment snapshots and a new measured selection.
   Freeze the resulting plan, reference, bundle and executor before independent
   acceptance timing. Retain every failed historical run as separate evidence.

## Acceptance and final audit

The diagnostic pass supports the approved implementation target. The
production acceptance run must still establish all 54 placements, exact tokens,
fresh independent memory qualification, two warmups per job, shuffled repeated
timings, selected cancellation/deadline/cleanup/recovery checks, reference drift
at most 10%, and regret with a seeded 95% bootstrap upper bound at most 15%.
Start with five rounds and extend to at most fifteen if uncertainty requires it.
No unsupported memory exclusion is permitted.

After those checks finish, conduct the user's requested separate M5.1–M5.6 audit:
implementation versus requirements, raw evidence and identities, precision claims,
coverage, memory and health results, timing/statistics, tests, documentation and
remaining risks. Mark M5 complete and create the PR only after that audit passes.

## Decision

The user’s “Proceed with the proposed target” authorizes implementing and
qualifying this explicit mixed-precision target. It does not approve milestone completion, a PR before the final audit,
increased memory caps, or any correctness relaxation. Production acceptance remains
pending until fresh qualification and the separate audit pass.
