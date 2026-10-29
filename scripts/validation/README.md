# Qualification tools

Run these tools explicitly; none schedules work or starts automatically. Full model
work needs a fresh resource check on every host. Tiny fixture tests are suitable for
normal development. Ordered next steps are on [the status page](../../docs/STATUS.md);
the [two-host log](../../docs/overnight-backlog.md) records the September–October 4B
runs. Host-specific values live in a run directory's copy of
[`two_host/two-host.env.example`](two_host/two-host.env.example), never in the repo.

Run every tool from the repository root as a module (`uv run python -m
scripts.validation.<tool>`). The tools that import their siblings also accept the
file path form (`python scripts/validation/<tool>.py`) for older instructions.

## Resource guard

`resource_guard.py` is a standard-library-only wrapper for one owned process group.
It records system availability, new swap-outs and Mac pressure every approximately
two seconds. It stops on less than 1 GiB available, more than 256 MiB new swap-out
over a 60-second window, 16 seconds of non-normal Mac pressure, sampling failure,
or its time limit. Startup also rejects non-normal Mac pressure, raising
`PreflightRefused` before anything is spawned. TERM, INT and HUP retire the owned
group; a leader that outlives SIGKILL does not suppress the final `exit` record.
Exit status is 124 when a stop condition ended the run, otherwise the child's status.
Every log is exclusive: existing evidence is never replaced.

```bash
cc -O2 -Wall -Wextra -Werror scripts/validation/process_footprint.c -o build/process-footprint
uv run python scripts/validation/resource_guard.py \
  --seconds 1800 --record build/run/worker.guard.jsonl \
  --footprint-helper "$PWD/build/process-footprint" -- \
  build/native/mlx/cpp/hllm-worker-mlx \
  --listen 127.0.0.1:50291 --worker-id mlx \
  --model-root build/models/Qwen3-4B-Base \
  --memory-limit-bytes 4697620480 --max-cached-tokens 384
```

The optional helper is macOS-only and observes the **direct child PID**. Use it
when the child is a native worker. For an isolated `uv run hllm profile-memory`
command, omit the helper: the native probe records its own OS counters, while the
guard protects the entire descendant group. Do not mislabel Python/uv footprint as
model-worker footprint. Run one guard on each host; a Mac controller timeout alone
does not guarantee remote cleanup. Keep the owning guard PID and its recorded child
PID; stop those processes explicitly when a run ends and verify both hosts are clean.

## Resource gates

These apply to every model-bearing step, by hand or through the launchers.

1. Record Mac `vm_stat`, pressure, swap, load, disk and top processes, and Linux
   `free -h`, `nvidia-smi`, load, disk and relevant processes. Do not stop other
   applications; freeing memory is the operator's decision. Availability uses
   `host_available_bytes()`, the definition profiling and native activation use.
2. Native caps plus 10% safety, 256 MiB extra overhead and headroom (1 GiB host,
   512 MiB CUDA device) must pass live preflight: for the Mac, free plus inactive memory
   of about 6.06 GiB at a 4.375 GiB MLX cap and about 6.9 GiB at 5.125 GiB. Leave a
   margin, because launching uv and the guard reclaims inactive pages. Never reduce
   allowances, and never raise a cap without the user's approval.
3. Guard every model-bearing worker or probe with `resource_guard.py` on its own host,
   independent of SSH and controller lifetime. No unbounded model process.
4. One substantial GPU workload at a time. Never run the independent oracle and hLLM
   CUDA workers together. Respect Linux host memory as well as free VRAM.
5. On any gate failure, preserve the rejected report, clean up owned processes and stop
   that expansion. Missing evidence is `unknown`, not passing. Do not retry unchanged
   resource conditions repeatedly.

New binaries invalidate binary-bound profile evidence, and profiles expire after
24 hours, so every run needs fresh same-day profiles on both hosts from the binaries it
serves. Keep previous evidence immutable: each session gets a new run directory.

## Reload soak

`reload_soak.py` accepts a fully hashed manifest, explicit uniform-F16 plan, independent reference, worker configuration
and an exclusive output path. Workers and network forwarding must already be running
under guards. The tool never connects to arbitrary discovered endpoints.

```bash
uv run python -m scripts.validation.reload_soak \
  --manifest build/run/manifest.json --plan build/run/mlx-cuda.plan.json \
  --reference build/run/reference-f16.json --workers build/run/soak-workers.json \
  --cycles 6 --requests 4 --idle-seconds 10 --timeout 180 \
  --output build/run/reload.json
```

The worker configuration maps each exact plan worker ID to the following fields
(replace the digest and driver placeholders with independently verified values):

```json
{
  "mlx": {
    "endpoint": "127.0.0.1:50291",
    "memory_profile": "/absolute/path/to/mlx-memory.json",
    "binary_digest": "<64-character worker executable SHA-256>",
    "runtime_driver_version": "<GetQualificationState driver_version>",
    "mlx_fit_policy": "footprint-v1"
  },
  "cuda": {
    "endpoint": "127.0.0.1:50293",
    "memory_profile": "/absolute/local/path/to/copied/cuda-memory.json",
    "binary_digest": "<64-character CUDA worker executable SHA-256>",
    "runtime_driver_version": "<GetQualificationState driver_version>"
  }
}
```

Profiles must be complete, less than 24 hours old, match the exact assignment,
checkpoint, precision, prompt/output lengths and capacity, and carry the same
admission caps as the workers. Before every load, the tool checks binary, backend,
device fingerprint, driver API, allocator, transfer mode, current availability and
the selected fit policy (`footprint-v1` unless the worker entry says otherwise). It
refuses busy workers. Default headroom and extra overhead
are the production defaults (1 GiB host, 512 MiB CUDA device, 256 MiB extra, 10% safety).

Requests use the prefill chunk the profiles were measured at (the workload's
`prefill_chunk_tokens`, 0 for whole-prompt prefill), because each worker's
reservation depends on it. Every profile must share one chunk, each worker must
advertise chunked prefill when the chunk is nonzero, and the report records it as
`prefill_chunk_tokens`. Chunking bounds a long prompt's boundary payload and the
per-pass workspace that admission reserves.

For a CUDA worker the fresh device budget credits the worker's own **inactive cached**
allocator bytes on top of the driver's free bytes: after an unload the caching allocator
keeps the model's blocks reserved, `cudaMemGetInfo` no longer counts them as free, and the
next load in the same process reuses them. Active allocator bytes are never credited, and
each cycle's preflight records the reported, credited and effective values separately
(`device_availability`). Before this credit existed, the October 10 soak passed cycle 0
and was refused at cycle 1 with 2.5 GiB reported free and 4.78 GiB cached.

Each request must match the independent reference exactly. Each cycle includes
cancellation after three tokens, request cleanup, unload and idle snapshots. Phase
snapshots stream to `.observations.jsonl`; completed requests and fresh fit results
are saved incrementally in the report. Load-time OS coverage comes from the worker's
guard/helper, independent of potentially blocked control RPCs. Polling changes timing;
this tool is not a throughput benchmark. An interrupted run preserves partial results.

A completed soak establishes the recorded execution/cleanup checks only. Compare
the observed **serving** peaks with the prelaunch budgets before declaring physical
fit; an isolated profile can underestimate serving overhead. Check `assessment.policy`
to distinguish footprint evaluation from conservative fallback. No concurrent-request
or indefinite reload-plateau claim follows from this concurrency-one harness.

## Concurrent soak (concurrency two)

`concurrent_soak.py` runs the bounded two-request experiment from the
[concurrency-two design](../../docs/concurrency-qualification.md) and writes one
immutable `ConcurrentServingEvidence` record (`hllm_control.qualification.concurrent`).
Start each worker under its guard exactly as for the reload soak, plus
`--max-active-requests 2`, an aggregate `--max-cached-tokens` of at least two
reservations, and `--request-observations on`. That opt-in makes `GetMemoryReport`
carry each live request's own KV/workspace reservation and a bounded log of
admissions and retirements on the worker's steady clock, each listing the other
reservations whose allocation had completed at that instant. The harness refuses
workers that do not advertise it, admit fewer than two requests, or batch decodes.

```bash
uv run python -m scripts.validation.concurrent_soak \
  --manifest build/run/manifest.json --plan build/run/mlx-cuda.plan.json \
  --reference build/run/reference-f16.json --workers build/run/concurrent-workers.json \
  --baseline-report build/run/baseline-reload-soak.json \
  --source-digest "$HLLM_SOURCE_ID" --transport "loopback over ssh -L/-R forwarding" \
  --guard-description "1 GiB floor, 256 MiB/60 s swap-out, 40-minute ceiling" \
  --cycles 3 --pairs 3 --timeout 180 --output build/run/concurrent.json
```

`--baseline-report` must be the reload soak's report: before anything loads, the
harness checks that it completed with exact outputs, in the same scope (`f16-reload-
qualification`, or `cpu-rehearsal` under `--cpu-rehearsal`), for this plan digest,
against this reference file, on these worker binaries and at the profiles' prefill
chunk (a report without one sent whole-prompt prefill), and the record keeps that
summary. A baseline produced on other binaries is refused, so a rebuild needs a new
reload soak first. The worker configuration is the reload-soak format plus two
optional paths per worker: `guard_record` (that worker's `resource_guard.py` JSONL)
and `device_samples` (a `memory_watch.py --cuda` JSONL for the CUDA worker; without it
the device side of the physical envelope is unknown, never safe). The guard must have
spawned the worker process itself (the start event's PID is matched against the
worker's reported PID), started before the harness, and still be observing within 20 s
of the end; a stop reason, an early exit or a failing exit status is `stopped`, and
anything less than full coverage is `unavailable`. Device samples are recorded with
their PIDs and must belong to the worker process. Each cycle re-runs the fresh
single-request fit gate and records that cycle's live budget, loads, runs one serial
exact request, `--pairs` paired rounds released from a barrier, then a cancellation
round in which one request is cancelled after `--cancel-after` tokens (default three)
while the other completes, then unloads. Both members of every pair must match the
sealed reference exactly; the cancelled request must match a proper prefix of it, its
worker-side retirement must say `cancelled` while the survivor still held its
reservation, and the survivor's retirement must say `completed` afterwards.

Overlap is claimed per worker only from that worker's own events: a retirement of
either member must list the other as an allocated concurrent reservation (or, failing
that, a polled snapshot must carry both allocated rows); the interval starts at the
later admission. Two requests that reserve together list no peer in their admissions,
which is not evidence against overlap. Queued-only submission, a single-worker
overlap, a missed observation window, a worker restart, an unbound guard record or an
unverified cleanup make the record `unknown` or `failed`. Everything the record
concludes (overlap, logical peaks, one physical envelope per worker and cycle against
that cycle's budget, guard outcomes and the acceptance status with its reasons) is
derived from the record's contents and re-derived on every load, so an edited record,
or a stale safe envelope carried over observations that no longer fit, is rejected.
The record states one set of allowances (safety fraction, extra overhead, host and
device headroom), so the harness refuses workers configured with different ones, and
loading rejects a record whose fit gates used allowances other than those it states.

What a failure leaves depends on when it happens. Refusals of the inputs (profiles,
baseline, reference, plan) come before any worker is contacted and write nothing. A
failure while establishing the workers' identity (an unreachable peer, a worker
without `--request-observations on`) writes `<output>.failure.json` with the reason.
Any later failure writes the partial record with its failure reason; if even assembly
fails, a `.failure.json` note is written beside the output and the original error is
kept.

Guard records are still being written when the record is assembled (a guard must
outlive the experiment), so the harness reads each one exactly once: the recorded
`record_sha256`, every parsed field and `<output>.guard-<worker>.jsonl` all come from
those bytes, and that snapshot reproduces the hash later. An incomplete final line,
from a mirror copied mid-write, is not yet evidence and is skipped, in guard records
and in device samples alike. (The October 13 record predates this; see its report.)

The cancellation round is a race: the controller's cancel RPCs must
reach every worker before the victim completes naturally there, or that worker
retires it as `completed` and the record is `unknown` (the reason names the observed
retirement), so the reference must be long relative to `--cancel-after`. The CPU
rehearsal (`tests/profiling/test_concurrent_cpu.py`, run by the `CpuReloadRehearsal`
CTest entry) exercises the whole procedure on tiny workers with a 128-token reference
and `--cancel-after 3`, which is procedure evidence only. No throughput claim and no
concurrency above two follows from an accepted record.

## HTTP serving soak

`serving_soak.py` drives the real HTTP server at increasing levels (default 1 and 2)
against running, guarded workers. It binds to the supplied plan and to the same
worker specifications as the reload soak, rather than building a plan of its own.
Before each level loads, it checks that every worker admits the level (its
`--max-active-requests`, and an aggregate `--max-cached-tokens` that holds the level's
reservations) and re-runs the fresh single-request fit gate against each worker's
same-day profile; a refusal stops the run before loading, with the reason in the
report. A level passes only when every stage, on its own, was observed holding that
many active requests. Each round sends one request more than the level, so one is
queued by the server. Seeded sampling is batch-dependent, so compare sampled tokens
only within one level.

```bash
uv run python -m scripts.validation.serving_soak \
  build/run/manifest.json build/run/tokenizer build/run/serving.json \
  --plan build/run/mlx-cuda.plan.json --workers build/run/serving-workers.json \
  --levels 1 2 --rounds 5 --output-tokens 64 --timeout 180
```

It sends two built-in prompts (not a fixed 128 tokens) and sizes each reservation as the
longest prompt plus `--output-tokens`; record that workload rather than relabeling it.
With the Qwen3 tokenizer the longer prompt is 130 tokens, so against a 384-token
profile use at most 254 output tokens; the gate refuses anything larger before loading.
It does not check `--max-decode-batch`, so start the workers at batch 1 unless batching
is what the run measures. `--prefill-chunk-tokens` must match the plan's workload;
`--sampling-json` and `--minimum-duration-seconds` are optional. `run.sh` has no serving
role: start the guarded workers as for `concurrent`, then run this by hand.

## Two-host launchers

[`two_host/`](two_host/) holds the launch tooling derived from the October 2026 4B runs,
with the fixes those runs found, parameterized by one settings file (copy `two-host.env.example` into the run
directory and fill it in; it is the only place host names, SSH aliases, paths, caps
and limits go). Every script takes that file as its first argument. The Linux host
runs the controller and the CUDA worker; the Mac runs the MLX worker, reached over
SSH. Each Mac checkout needs `uv sync` and the MLX build at the same commit, plus
`build/process-footprint` (see above).

| Script | Host | Does |
| --- | --- | --- |
| `build-cuda.sh` | Linux | RelWithDebInfo CUDA build of the worker and probes, with hashes |
| `oracle.sh ENV PROMPT OUTPUT` | Linux | Guarded independent reference into `inputs/reference-PROMPT-OUTPUT-f16.json`; point `REFERENCE` at it |
| `profile.sh ENV cuda` | Linux | Guarded CUDA memory profile (`cuda-profile.sh`) |
| `profile.sh ENV mlx` | Linux to Mac | Copies settings and inputs, runs `mlx-profile.sh` on the Mac detached, copies results to `mac/`, exits with the profile's status |
| `run.sh ENV baseline` | both | Guarded workers, tunnel, reload soak, cleanup |
| `run.sh ENV concurrent` | both | The same plus the device watcher, the Mac guard mirror, the concurrency soak and results assembly |

The other scripts (`tunnel.sh`, `watch-cuda.sh`, `cuda-worker.sh`, `mlx-worker.sh`,
`cuda-profile.sh`, `mlx-profile.sh`, `lib.sh`) are called by these.

A run, in order: a new run directory with `source/` cloned from `main` and the same
commit built on the Mac; `two-host.env` filled in; inputs (manifest, plan, workload,
reference, `inputs/<role>-workers.json`) in place; `build-cuda.sh`; `profile.sh cuda`
and `profile.sh mlx` on the same day; `run.sh baseline`; then `run.sh concurrent`.
After each run, inspect fresh gates, every continuation, reservation cleanup, Mac
pressure and swap-outs, Linux availability and sampling errors, and compare serving
peaks against prelaunch availability with unchanged allowances. A report records
source, binary, checkpoint and raw-evidence hashes, all failures and fallbacks, the exact
scope, caps, policies and budgets, token comparisons, guard summaries, reload residuals
and cleanup evidence, as a Markdown report and machine-readable summary in
`docs/validation/`; bulky raw artifacts stay in the run directory.

`run.sh` refuses busy ports on either host, and any hLLM worker or memory probe
already running there on any port (matched by executable path with `pgrep -f`;
Linux truncates process names to 15 characters). Each launch waits for
its port while checking that the process is still alive, and fails at once with the
log tail if it died. The device watcher (`watch-cuda.sh`) samples the PID the CUDA
worker reports through its control API (`worker_pid.py`), never a command-line match.
Cleanup stops the watcher and both guards and waits for them to exit, so each guard
record ends with its exit event. It copies the final Mac records to `mac/final/`,
leaving the mirror the soak read untouched, and only then runs
`assemble_two_host_results.py`. The worker specifications are
`inputs/<role>-workers.json` in the reload-soak format; for `concurrent` the guard and
sampler paths must be the ones `run.sh` writes (`<run>/concurrent-cuda.guard.jsonl`,
`<run>/concurrent-cuda-device-samples.jsonl`, `<run>/mac/concurrent-mlx.guard.jsonl`),
which `run.sh` checks before launching. It also checks that each `binary_digest` is
the SHA-256 of the worker binary it will start (`CUDA_BUILD_DIR` here, `MLX_BUILD_DIR`
on the Mac), so specifications copied from an earlier run fail before anything
starts, with both digests in the log; set them to the new build's hashes. The scripts
resolve the settings path to an absolute one, so a relative path works. Mac processes start detached with their own
stdio, so the SSH call returns at once with the process's own PID. `profile.sh mlx`
propagates the Mac profiler's exit status (2: incomplete measurement, 3: physical fit
not safe), which `mlx-profile.sh` records in `mlx-profile.exit`; the profiler writes
its artifact before refusing, so the copied file alone is not success.

`source_identity.py` records a checkout's revision, tracked-file hashes, combined
source digest and binary hashes; profiles take that digest as `--source-revision` and
evidence as `--source-digest`. `assemble_two_host_results.py RUN OUTPUT` summarizes
profiles, baselines and every P5 attempt in a run directory from the preserved files
only, naming the directory by its base name. A baseline that ran no requests is not
`all_exact`. Each attempt also lists the final guard records (`final_guards`) and, in
`guard_stops`, every guard that stopped its worker itself rather than being stopped
by cleanup. When a worker stops answering, the concurrency soak waits up to 20 s for
that guard's exit event before reading its record, so a guard-stopped worker is
recorded as `stopped`, not `normal`. On the October 11–13 run directory it
reproduces the committed `docs/validation/qwen3-4b-concurrency-20261013-results.json`
except for one more profile directory it discovers (`cuda_rejected_1`).

## Independent reference

`checkpoint_reference.py` takes `--plain` for Base checkpoints and `--gpu-layers` for an
explicit Accelerate device map. Run it in a separate pinned
Torch/Transformers/Accelerate environment, under the Linux guard, with all hLLM
GPU workers stopped. It hashes the checkpoint and records tokenizer mode, precision,
library versions and placement. Example for the existing 4B model:

```bash
python scripts/validation/checkpoint_reference.py /path/to/Qwen3-4B-Base \
  /new/output/reference-512-f16.json \
  --plain --dtype f16 --gpu-layers 26 --prompt-tokens 512 --output-tokens 256
```

The previous pinned reference used Torch 2.13.0+cu130, Transformers 4.57.6 and
Accelerate 1.15.0. Verify that environment before reuse. The 26-GPU-layer mapping
is historical evidence, not an unconditional fit guarantee for a new context.
This oracle's CPU weight placement is distinct from hLLM runtime offloading.

## Publishing evidence

The repository is public. Before committing evidence, run `redact_report.py` over
every file of the report in one invocation, so each value gets the same placeholder
everywhere:

```bash
uv run python -m scripts.validation.redact_report --in-place \
  docs/validation/<report>/*.json docs/validation/<report>/*.json.gz
```

It replaces host names (`<cuda-host>`, `<mac-host>`), tailnet, public and LAN addresses
(`<cuda-tailnet-ip>`, `<mac-public-ip>`; unattributed ones become `<IPv4 address>`),
Tailscale node IDs and keys, reverse-DNS names, MAC addresses and home directories
(`~`). Roles come from link-probe worker IDs, their path observations and Tailscale
peer records. JSON formatting is preserved; a sealed `artifact_digest` is recomputed
with the profiler's canonical digest, and digests of rewritten archives cited by
other files in the same invocation (a summary's `archive_sha256` and
`profile_digest`) are updated. Digests that cover uncommitted raw files (for example
`evidence_digest` or `artifact_sha256` in planner diagnostics) are left as recorded.
`redact_report.py INPUT OUTPUT` still redacts a single report.

Host identifiers in already committed evidence were redacted on 2026-10-19 with this
tool; the affected link archive digests and the [timing summary](../../docs/validation/milestone-5-timing/summary.json)
were recomputed after redaction. The pre-redaction bytes remain in git history.

## Routine CI

The [workflow](../../.github/workflows/ci.yml) checks Python, types, lint, generated
bindings and native CPU tests, and runs the same CTest suite under the `asan` preset
(AddressSanitizer plus UndefinedBehaviorSanitizer on hLLM code only; the pinned
dependency prefix is uninstrumented, so leak and container-overflow detection are
disabled there, as the workflow comments explain). Pushes to `codex/**` run a deps-only
job that restores or builds and saves the dependency cache so stacked pull requests
based on those branches do not rebuild gRPC cold. Locally the `asan` preset needs
`libasan` and `libubsan`. The native dependency helper builds matching gRPC and Protobuf
from pinned gRPC v1.74.0 and caches the install prefix; its configure sets a CMake 3.5
policy floor because the pinned c-ares submodule otherwise fails with CMake 4. The
native job permits 90 minutes for a cold build and saves the dependency cache as soon as
that build succeeds. Hardware and full-model runs are separate. The recipe was first
validated in the [September 28 Linux run](../../docs/validation/qwen3-4b-fit-20260928.md)
and the [September 30 readiness report](../../docs/validation/cpu-readiness-20260930.md).
Sources for the workflow setup are the official
[setup-uv instructions](https://github.com/astral-sh/setup-uv) and
[gRPC build instructions](https://github.com/grpc/grpc/blob/master/BUILDING.md).

## CPU rehearsal while the GPU is busy

Use the built CPU worker and profiler; no checkpoint download or accelerator is needed:

```bash
HLLM_CPU_WORKER="$PWD/build/native/dev/cpp/hllm-worker-cpu" \
HLLM_MEMORY_PROFILER="$PWD/build/native/dev/cpp/hllm-profile-memory-cpu" \
uv run python -m scripts.validation.cpu_rehearsal
```

`CpuReloadRehearsal` runs the same entry point through CTest. Every invocation
creates a new directory under `build/cpu-rehearsal/`, retaining the resource guard
log, JUnit results and pytest artifacts (profiles, plans, tokens, worker logs,
soak reports and observations). The 300-second guard owns the pytest process group
and both tiny CPU workers; fixture teardown closes workers on ordinary failures.
The guard enforces the same availability, swap and pressure limits as other runs.
If it refuses to start, the wrapper exits 77 and CTest reports the entry as skipped
(`SKIP_RETURN_CODE 77`); any other wrapper failure exits 125 and fails. Both write
`result.json` with the reason. This is a serial, tiny-model procedure test, not a
4B memory benchmark.

The harness's explicit `--cpu-rehearsal` mode requires uniform F32, CPU profiles
and a reference labeled `producer.purpose="cpu-rehearsal"`. Reports carry
`scope="cpu-rehearsal"`. Its token baseline comes from the same CPU implementation
and is not an independent numerical oracle. The default harness still requires
F16 and retains the existing qualification checks. Fault tests cover stale/future
profiles, changed binary/cap/workload, token mismatch and a lost worker. Partial
request tokens, the cancellation probe's expected and observed prefix, and per-worker
cleanup observations remain in failed reports; an unreachable worker is not reported
as cleaned up.

Concurrency-two evidence is a separate [design](../../docs/concurrency-qualification.md),
not a relaxation of the current concurrency-one profile format.
