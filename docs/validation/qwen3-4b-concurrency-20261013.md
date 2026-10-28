# Qwen3-4B concurrency two: October 12–13, 2026 measurement

Continuation of the [October 11 concurrency night](qwen3-4b-concurrency-20261011.md)
in the same run directory (`~/Projects/experiments/hllm-4b-fit-20261011-044533/` on the
CUDA host, `build/4b-fit-20261011-044533/` on the Mac). The user freed Mac memory and
asked to proceed; the P5 measurement then ran to an **accepted** record after four
refused or unusable MLX profile attempts and five concurrency attempts that ended
earlier, each preserved unchanged. Two tooling defects found on the way are fixed in
PR #27 and both fixes were required for the accepted record. A review of PR #27 on
October 13 found that the first form of the MLX telemetry fix was itself a data race
and that the preflight's CUDA device sample was stale; both are corrected in the same
PR (see "Telemetry defect" below and the caveat it attaches to this record). Times are
UTC.

## Result

| Item | Value |
| --- | --- |
| Record | `concurrent.json`, schema 1.1, evidence digest `4a05d0f6f6bfd1202e523c144e983b6288ba1ae23f15b5478a7ee33197df725e` |
| Acceptance | `accepted`, no reasons, re-derived on a fresh load of the record |
| Window | 01:55:00Z–02:06:53Z, October 13, 2026 |
| Workload | 128 prompt / 256 output tokens, 384-token reservation per request, F16, forward split (MLX layers 0–14, CUDA layers 15–35), plan `1ed77abe…`, reference `329ea04b…` |
| Procedure | 3 cycles × (fresh fit gate on both workers, load, 1 serial request, 3 paired rounds released from a barrier, 1 cancellation round cancelled after 3 tokens, unload); 15 rounds, 27 requests |
| Exactness | 27/27 exact against the sealed reference; both members of every pair; victims a 3-token proper prefix |
| Overlap | 24 rows: proven on both workers for all 12 non-serial rounds from each worker's own retirement witnesses; peak concurrent requests 2 on both workers |
| Cancellation | all three rounds: victim retired `cancelled` while the survivor held its reservation; survivor retired `completed` afterwards |
| Caps | MLX unified 5.125 GiB (5,502,926,848 B), CUDA host 3.75 GiB / device 6.125 GiB; `--max-active-requests 2 --max-cached-tokens 768 --request-observations on` on both |
| Logical peaks | within admission on both workers |
| Guards | both `normal`, bound to the worker PIDs (MLX 14103, CUDA 3075942), 350 and 358 observations, no swap-out, no stop |
| Cleanup | verified on both workers after every cycle |
| Source | `59cfe606…` (commit `fbf4d0b`, PR #27 head); MLX worker `9e801e78…`, CUDA worker `577b9ca0…` (unchanged since `97aa6b0`) |

Per-cycle gates and physical envelopes (all `safe`):

| Cycle | Gate time | MLX available | MLX envelope (footprint-max) | CUDA host avail / envelope | CUDA device avail / envelope |
| --- | --- | --- | --- | --- | --- |
| 0 | 01:55:04Z | 7.08 GiB | 5435 MiB (peak 4708) | 9.11 GiB / 3311 MiB | 7.33 GiB / 5932 MiB (peak 5160) |
| 1 | 01:59:05Z | 8.68 GiB | 5436 MiB (peak 4709) | 7.99 GiB / 4210 MiB | 7.29 GiB / 5932 MiB |
| 2 | 02:03:00Z | 8.38 GiB | 5443 MiB (peak 4715) | 7.79 GiB / 4232 MiB | 7.29 GiB / 5932 MiB |

The MLX allocator sample is present in every one of the worker's 3,430 snapshots (the
attempt-2 gap below is closed); the CUDA worker has 22,612 snapshots and 1,350 device
samples from `memory_watch.py --cuda` inside the window, all bound to the worker PID.

Preconditions bound into the record:

- **Baseline:** `baseline-reload-soak.json`, scope `f16-reload-qualification`, 3 cycles,
  6/6 exact, 01:03Z–01:09Z, on the same plan, reference and worker binaries.
- **P1 profiles (fresh, same binaries):** CUDA `88aae48a…` measured 18:37:53Z October 12
  (safe, `cuda-rss-device-v1`); MLX `a9d80d45…` measured 01:02:01Z October 13 at the
  5.125 GiB cap (safe, `mlx-footprint-max-v1`, envelope 5.312 GiB, preflight sample
  7.36 GiB before the checkpoint read, guard gap ≤ 2 s).

What this record does **not** claim: any throughput or latency figure, any concurrency
above two, any workload other than 128/256, any cap other than the ones above, or
anything about the reverse order. The MLX cap of 5.125 GiB is the P3 value from
October 10 and was raised from 4.375 GiB only after the user approved it (below).

Results file: [`qwen3-4b-concurrency-20261013-results.json`](qwen3-4b-concurrency-20261013-results.json)
(`assemble_results_20261013.py` over the run directory, tracked since PR #31 as
`scripts/validation/assemble_two_host_results.py`: every profile attempt,
baseline and concurrency attempt with its hashes, gates, envelopes and notes).

## Timeline

### MLX profile attempts (cap 4.375 GiB unless noted)

| # | Launch | Outcome | Preserved as |
| --- | --- | --- | --- |
| 3 | 18:31:06Z | Refused at preflight: 5.91 GiB sampled against 6.06 GiB. The guard read 8.01 GiB at launch and 6.34 GiB four seconds later. | `mac/rejected-3/` |
| 4 | 18:37:08Z | Refused at preflight with the fixed sampler: 6.03 GiB before the checkpoint read (5.88 after), 30 MiB short. | `mac/rejected-4/` |
| unsafe-1 | 18:39:10Z | Preflight accepted (6.23 GiB after Discord and Spotify were quit); measurement completed; fit **unsafe**: envelope 5.668 GiB plus headroom exceeds 6.23 GiB (October 10 had 7.71 GiB). | `mac/mlx-profile-unsafe-1/` |
| interrupted-1 | 18:41:44Z | Mac entered Maintenance Sleep a minute after launch (dark wakes every five minutes); the suspended run finished at 19:00Z reading 5.28 GiB. Not used: measured across sleep, guard gaps of 138–300 s. | `mac/mlx-profile-interrupted-1/` |
| cap4375 | 00:22:52Z | Awake, `caffeinate -dis`; safe, envelope 5.277 GiB, guard gap ≤ 2 s. | `mac/mlx-profile-cap4375/` |
| cap5125 | 00:37:05Z | Cap raised to 5.125 GiB (user approval); safe, envelope 5.275 GiB. | `mlx-profile-cap5125-bin09a7/` |
| final | 01:01:57Z | Rebuilt MLX binaries (`fbf4d0b`); safe, envelope 5.312 GiB. | `mac/mlx-memory.*` |

The CUDA P1 profile from October 11 expired at 04:50:49Z; it is preserved in
`cuda-profile-expired-20261011/` and was re-measured at 18:37:46Z–18:38:55Z (page
cache dropped first, guard unchanged): safe, artifact `88aae48a…`.

Preflight defect (fixed in PR #27, `f4dcee5`). Importing the CLI costs nothing (6.00 GiB
before, 6.06 GiB after), but `run_memory_profile` digested the whole checkpoint before
sampling availability, and on macOS the pages it had just read sit on the active list,
outside the free-plus-inactive proxy. The runner now samples before that read and records
the post-read value beside it (preflight schema 1.1). The October 11 "launch allowance"
lesson is corrected in that report. The requirement and allowances are unchanged, and
attempt 4 above shows the fix does not make a short Mac pass.

### Baselines (single-request reload soak, 3 cycles × 2 requests)

| Cap / binaries | Window | Result | Preserved as |
| --- | --- | --- | --- |
| 4.375 GiB, `09a79a4c` | 00:24Z–00:30Z | 6/6 exact | `baseline-cap4375/` |
| 5.125 GiB, `09a79a4c` | 00:38Z–00:44Z | 6/6 exact | `baseline-cap5125-bin09a7/` |
| 5.125 GiB, `9e801e78` | 01:03Z–01:09Z | 6/6 exact | `baseline-reload-soak.json` (bound into the record) |

Each was needed because the harness (schema 1.1) binds the baseline to the worker
binaries under test, and the October 10 baseline was on the previous binaries.

### Concurrency attempts

| # | Window | Outcome | Preserved as |
| --- | --- | --- | --- |
| 1 | 00:31Z–00:33Z | **Failed at the 4.375 GiB cap.** Cycle 0 serial exact; in the first paired round the MLX worker admitted `c0-p0-a` (3630 MiB weights + 22 MiB cache + 663 MiB workspace = 4315 of 4480 MiB) and refused `c0-p0-b` with `RESOURCE_EXHAUSTED: allocation exceeds host, device, pinned-host, or unified memory budget`; CUDA would have fit (5442 of 6272 MiB). Also found: `watch-cuda.sh` had been handed the guard's PID, so device samples bound to another process (script fixed). | `concurrent-attempt-1/` |
| 2 | 00:44Z–00:57Z | **Unknown at 5.125 GiB.** 27/27 exact, all overlap and cancellation proven, CUDA safe, guards normal; the MLX worker returned no allocator sample in 2548 of 3449 busy snapshots, so the MLX envelope could not be sealed (conservative 5435 MiB). | `concurrent-attempt-2/` |
| 3 | 01:09Z–01:18Z | **Failed by the gate** on the rebuilt worker. Cycles 0–1 accepted-grade (18/18 exact, MLX envelopes 5435/5436 MiB safe, allocator present in all 2305 snapshots); the gate before cycle 2 saw 4.80 GiB on the Mac after a browser and an editor were opened there. | `concurrent-attempt-3/` |
| 4 | 01:47Z | No measurement: the remote launch of the MLX worker returned only "Current directory does not exist" and the driver could not connect; the same launch by hand at 01:52Z worked. | `concurrent-attempt-4/` |
| 5 | 01:53Z | No measurement: aborted by the harness's exclusive-create because attempt 4's observations file had not been moved (operator error). | `concurrent-attempt-5/` |
| 6 | 01:55Z–02:07Z | **Accepted** (table above). | `concurrent.json` |

Cap decision. Attempt 1 is honest evidence that the MLX stage cannot admit two 128/256
requests under the 4.375 GiB cap: each request reserves 685 MiB on top of 3630 MiB of
weights and the cap leaves 165 MiB after the first. The
[design](../concurrency-qualification.md) forbids expanding caps automatically after
a failure, so the agent stopped and asked; the user approved 5.125 GiB, the value P3 had
already used on October 10, and the whole chain (profile, baseline, soak) was redone at
that cap.

Telemetry defect (PR #27). `MlxBackendFactory::allocator_metrics()` took the device
mutex with `try_lock` and returned nothing when a decode step held it, which is exactly
when the peak occurs. The first fix (`fbf4d0b`, the binary this record was measured
with, MLX worker `9e801e78…`, profiler `bf229e22…`) read MLX's counters without the
device mutex on the claim that MLX guards them itself. That claim was wrong: in MLX
0.32.2 `get_active_memory`, `get_cache_memory` and `get_peak_memory` return plain
fields with no lock, while `malloc`/`free` update them under the allocator's mutex on
the scheduler thread, so a scrape taken mid-step was a data race (undefined behavior by
the standard; on arm64 an aligned 64-bit load does not tear, so the recorded values are
at worst a few microseconds stale, never torn). The PR review caught it. The corrected
worker publishes the counters at every execution boundary, where the device mutex is
held and the stream synchronized (load, each step, sequence release, unload), and a
scrape that cannot take the mutex reports that sample instead of reading live; a
scrape that can take it reads live, because every holder synchronizes before releasing.
Releasing caches and weights now also happens under the device mutex, which closes the
same race for the off-mutex frees the old `try_lock` version already had. The sample a
mid-step scrape reports is at most one step old and carries the same monotonic peak.

Caveat on this record. Its MLX worker (`9e801e78…`) contains the racy read. The
numbers it produced are not affected in any way that could change the result: the MLX
envelope at every gate was decided by the OS footprint lifetime peak (5435–5443 MiB)
from the independent process-memory sampler, not by the allocator peak (4708–4715 MiB),
and the allocator sample only had to be present and coherent. The record remains
`accepted` on re-derivation and the reviewer re-derived the same digest independently.
It is, however, a measurement taken with a worker whose telemetry path was not
race-free, and the corrected worker is a different binary, so a re-run on the corrected
binary is listed under Follow-ups; until then this record stands with this caveat. The
CUDA binaries are byte-identical to the October 11 build because no CUDA source
changed.

## Guard records and the hashes in the record

The record hashes each guard record as it stood at assembly (02:06:53Z); both guards kept
writing until cleanup stopped them twenty seconds later, so the files on disk now differ.
The exact prefixes the record hashed were reconstructed byte-for-byte and saved beside
the live files: `concurrent-cuda.guard.as-recorded.jsonl` (`d09ed85e…`, 360 rows) and
`mac/concurrent-mlx.guard.as-recorded.jsonl` (`1ad72f85…`, 352 rows); the Mac's final
record is `mac/concurrent-mlx.guard.final.jsonl`. A follow-up for the harness: stop the
guards, or snapshot their records, before hashing them. (Done in PR #31: the harness
hashes the bytes it parsed and saves them as a snapshot.)

## Host conditions

- **Mac:** the user quit Discord, Spotify and ChatGPT at the agent's request (each was
  asked for individually; the agent quit nothing on its own), later closed a browser and
  an editor, and kept the lid open; the agent held `caffeinate -dis` assertions during
  the runs and released them at the end. The unloaded free-plus-inactive reading was
  7.1–8.7 GiB during the accepted run.
- **CUDA host:** Minecraft service stopped (since October 8), page cache dropped before
  every launch, no competing GPU process. The CUDA profile is valid until 18:38Z on
  October 13.
- Both hosts were idle at the end: no workers, guards, tunnels or watchers; zero GPU
  compute processes.

## Follow-ups

*October 15, 2026: the harness follow-ups below (guard hashing, PID-bound device
watcher, fail-fast launches) closed in PR #31, which also tracks the launchers in
`scripts/validation/two_host/`. The P5 re-run and the workspace question stay open in
the [project status](../STATUS.md).*

*October 20, 2026: the P5 re-run on the corrected worker was accepted with the same
physical peaks ([report](qwen3-4b-concurrency-20261020.md)), which resolves the caveat
above. The workspace question is answered by sharing one workspace per stage (PR #35).*

- Re-run P5 on the corrected MLX worker (boundary-published allocator sample) with
  fresh same-day profiles on both hosts, so that an accepted record exists whose
  telemetry path is race-free. The procedure is unchanged:
  `scripts/validation/two_host/profile.sh ENV mlx` with the Mac memory gate met, then
  `run.sh ENV baseline` and `run.sh ENV concurrent`.
- Harness: hash guard records after stopping the guards (above).
- `run-concurrent.sh` relies on `pgrep -f` to find the worker PID for the device watcher;
  the worker should report its PID through the control API so the watcher is bound by
  the record, not by a shell pattern.
- The MLX per-request workspace reservation (663 MiB for a 384-token request) is what
  makes the 4.375 GiB cap too small for two requests; whether it can be shared across
  active requests is a design question, not a measurement one.
- The remote launch failure in attempt 4 ("Current directory does not exist") was not
  reproduced; the launcher should check the worker's listen port and fail fast instead
  of waiting out the 180 s loop.
