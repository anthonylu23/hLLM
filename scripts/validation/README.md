# Qualification tools

Run these tools explicitly; none schedules work or starts automatically. Full model
work needs a fresh resource check on every host. Tiny fixture tests are suitable for
normal development. The [overnight backlog](../../docs/overnight-backlog.md) contains
the ordered 4B work and current machine-specific paths.

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
  build/native/m6-mlx/cpp/hllm-worker-mlx \
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

## Reload soak

`reload_soak.py` replaces the fixed-path September 23 diagnostic. It accepts a fully
hashed manifest, explicit uniform-F16 plan, independent reference, worker configuration
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
the selected fit policy (`footprint-v1` unless the worker entry says otherwise, since
October 10, 2026). It refuses busy workers. Default headroom and extra overhead
are the production defaults (1 GiB host, 512 MiB CUDA device, 256 MiB extra, 10% safety).

For a CUDA worker the fresh device budget credits the worker's own **inactive cached**
allocator bytes on top of the driver's free bytes: after an unload the caching allocator
keeps the model's blocks reserved, `cudaMemGetInfo` no longer counts them as free, and the
next load in the same process reuses them. Active allocator bytes are never credited, and
each cycle's preflight records the reported, credited and effective values separately
(`device_availability`). The October 10 four-request first cycle was refused on its second
cycle before this credit existed, with 2.5 GiB reported free and 4.78 GiB cached.

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

## Independent reference

`checkpoint_reference.py` now supports `--plain` for Base checkpoints and
`--gpu-layers` for an explicitly chosen Accelerate device map. These promote the
previous 4B oracle's behavior into a reusable tool. Run it in a separate pinned
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

## Routine CI

The [workflow](../../.github/workflows/ci.yml) checks Python, types, lint, generated
bindings and native CPU tests, and runs the same CTest suite under the `asan` preset
(AddressSanitizer plus UndefinedBehaviorSanitizer on hLLM code only; the pinned
dependency prefix is uninstrumented, so leak and container-overflow detection are
disabled there, as the workflow comments explain). Pushes to `codex/**` run a deps-only
job that restores or builds and saves the dependency cache so stacked pull requests
based on those branches do not rebuild gRPC cold. The first local sanitizer run on
October 10, 2026 passed all 81 entries on Fedora 44 with GCC 16 after installing
`libasan`/`libubsan`. The native dependency helper builds matching gRPC
and Protobuf from pinned gRPC v1.74.0 and caches the install prefix. Hardware/full-model
runs are separate. The dependency configure explicitly sets a CMake 3.5 policy
floor because the pinned c-ares submodule otherwise fails with CMake 4.
The [September 28 Linux run](../../docs/validation/qwen3-4b-fit-20260928.md) built
these dependencies from source and passed all 80 CPU CTest entries. Hosted Actions
first ran on September 30 (see the [readiness report](../../docs/validation/cpu-readiness-20260930.md));
the native job permits 90 minutes for a cold build and saves the dependency cache
as soon as that build succeeds.
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
soak reports and observations). The 180-second guard owns the pytest process group
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
