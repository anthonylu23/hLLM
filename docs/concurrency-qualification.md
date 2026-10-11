# Concurrency-two qualification design

Status: implemented October 11, 2026; **no concurrency-two capacity is qualified**
until the bounded two-host experiment below runs and its record is accepted. The
record is `hllm_control.qualification.concurrent.ConcurrentServingEvidence`, the
harness is `scripts/validation/concurrent_soak.py`, and workers expose the opt-in
request-lifecycle observations with `--request-observations on` (see the
[tools README](../scripts/validation/README.md#concurrent-soak-concurrency-two)). The
single-request 4B baseline passed on October 10. Existing tiny concurrent pipeline
tests are behavior checks, not a physical capacity certificate.

## Evidence boundary

Keep `ProfileKey` and measured placement at concurrency one. Do not change that
validator, multiply a single-request fit result by two, or import this experiment
into a placement bundle. Introduce a separate versioned `ConcurrentServingEvidence`
record when implementing this design. It must bind the exact serving configuration
and measured overlap; a single-stage synthetic profile cannot represent that run.

The record will contain these fields, with canonical JSON content hashes and strict
validation (unknown fields rejected, finite nonnegative measurements):

| Group | Required content |
| --- | --- |
| Identity | schema version, evidence digest, source digest, plan/manifest/checkpoint hashes, independent reference hashes, worker/profiler binary hashes, OS/backend/driver/allocator identity, transport and worker order |
| Workload | two exact token-ID sequences and hashes; per-request prompt/output/reservation lengths; effective weight/execution/KV/wire precision; greedy or fully specified sampling; deadline |
| Configuration | per-worker admission caps, active-request limit, aggregate cached-token limit, decode batch limit, prefill chunk size, queue capacity and HTTP controller limit |
| Preconditions | passing baseline report hash, fresh compatible single-request profile hashes (at most 24 hours old), live prelaunch availability, unchanged safety/overhead/headroom, guard settings |
| Observations | per-worker process identity, monotonic timestamp and UTC correlation, OS/allocator samples, logical weights/cache/workspace, active request IDs/counts, request admission and completion times, cancellation and unload events |
| Results | token comparisons, confirmed overlap intervals, peak logical reservations and physical envelopes, guard outcomes, cleanup evidence, failure reason and acceptance status |

Do not record only a global `max(active_requests)`: both stages must demonstrate
at least two simultaneously admitted requests in their own observations. At least
one observation per worker must include both live request IDs and their allocated
KV reservations. Mere concurrent HTTP submission or queueing is insufficient.
The implemented evidence is the worker's own lifecycle log: each admission and
retirement event lists the other reservations whose allocation had completed at that
instant, so the first retirement of a pair is the worker's own proof that both held
allocated KV together; a polled snapshot with both allocated rows is the fallback.
Workers without `--request-observations on` fail the overlap claim as unknown. Do not
infer cross-host timing order from wall clocks alone. Per-worker monotonic timelines
and request IDs establish local overlap; report clock alignment uncertainty separately.

## Admission and physical memory

For two 128-prompt/256-output requests, reserve 384 tokens per request and configure
at least 768 aggregate cached tokens with `max_active_requests=2`. Calculate native
cache and workspace using the actual backend configuration. Count resident weights
once per worker; count per-request KV once per request. Include concurrent or batched
workspace according to its actual lifetime, and loader scratch when checking cold
load. The approximate 0.669 GiB extra MLX reservation from the backlog is a planning
estimate, never admission evidence.

Start with `max_decode_batch=1` to isolate overlap from batch-dependent numerical
changes. Treat a later batch size of two as a separate configuration requiring new
evidence. Record prefill chunk size and actual prompt token counts; the HTTP soak's
default prompts must not be relabeled as 128 tokens.

Before launching, require configured host/device caps plus the existing 10% safety,
256 MiB overhead and headroom to fit live availability (1 GiB host; 512 MiB CUDA
device). This is a conservative preflight, not a measured concurrency certificate.
Do not expand caps or reduce allowances automatically after failure.

During serving, compare peaks against the recorded prelaunch availability. Apply
the selected policy once to combined observations from each worker process:
MLX uses the maximum of coherent OS footprint lifetime peak, RSS and allocator
residency/peaks only when qualified telemetry is present; otherwise report the
conservative fallback explicitly. CUDA requires process-device observations as
well as allocator evidence. Missing samples, missing overlap, process restart,
incomplete requests or missing cleanup yield unknown/failed evidence, never safe.
Do not add OS footprint to RSS or add a fresh model allocation per request.

## Bounded experiment

1. Check source/checkpoint/binary identities and the passing baseline. Reprofile
   stale single-request inputs; these only bind prerequisites, not combined fit.
2. Launch one independently guarded worker on each host with a 30-minute ceiling.
   Preserve exclusive logs, fresh budgets and process IDs. No independent oracle or
   other substantial GPU workload may overlap this experiment.
3. Run one serial exact-reference request, then release two requests from a barrier.
   Require recorded overlap and exact outputs for both; if work finishes before
   overlap is observed, report unknown rather than relaxing the requirement.
4. Repeat three paired rounds. In a separate round, cancel one request after three
   tokens while the other continues to its exact completion. Require the canceled
   prefix to match and reservations to retire independently.
5. Unload and reload three times in the same processes, repeating the pair each
   time. Observe residual memory and report the trend without claiming an indefinite
   plateau. Stop at the first correctness, pressure, swap, capacity or cleanup failure.
6. Retire only owned workers/guards/tunnels. Save partial evidence on any failure,
   including unavailable peers. Process exit is the final release of retained memory.

Acceptance requires all prerequisites, native admission, real overlap on both
workers, exact requested continuations, safe combined physical envelopes, normal
guard outcomes and verified cleanup. A numerical mismatch is a failed correctness
check even if memory fits. Do not silently weaken exact matching for batching;
qualify any changed numerical criterion as a separate experiment.

## Implementation and test follow-up

- [x] Standalone evidence model/validator and immutable report writer; changed
  hashes, caps, worker order, sampling settings and unsupported versions are
  rejected, and the acceptance status is re-derived from the record on every load.
- [x] Opt-in per-worker request-lifecycle observations (`--request-observations on`)
  proving overlap and independent reservation release; unavailable counters are
  recorded explicitly and make the claim unknown.
- [x] CPU tests (`tests/python/test_concurrent_evidence.py`): queued-only and
  one-worker-only overlap, premature cancellation, incomplete streams, dropped
  telemetry, process changes, stale profiles, missing device observations, guard stops,
  unverified cleanup and tampered records are never accepted; shared weights are
  counted once and two KV reservations are summed. The tiny two-worker rehearsal
  (`tests/profiling/test_concurrent_cpu.py`) produces an accepted record.
- [x] Concurrency-one serialization/hashes and placement behavior are unchanged;
  `ProfileKey` still requires concurrency one.
- [ ] Run the guarded two-host measurement only when the GPU is free and the baseline
  passes. Larger contexts, higher concurrency and throughput claims remain separate.
