# Milestone 5 — measured placement

Milestone 5 adds measured profiles (M5.1–5.4), measured placement (M5.5) and an
independent sweep runner (M5.6). This page covers CUDA memory accounting and
repeated-assignment measurement. See the [profiling workflow](milestone-5-profiling.md)
and [placement workflow](milestone-5-placement.md); evidence:
[memory](validation/milestone-5-memory.md), [timing](validation/milestone-5-timing.md),
[planner and gates](validation/milestone-5-planner.md) and the
[WAN closeout](validation/milestone-5-wan-closeout.md). Status is on
[the status page](STATUS.md).

## CUDA memory accounting

CUDA keeps one process-lifetime execution stream per device, including its startup
probe. Reassigning stages reuses that stream and its allocator/BLAS caches. Calls
still synchronize before returning, including on exceptions. Concurrent host-thread
submissions are supported by CUDA; stage state retains its own execution lock.

`GetMemoryReport` continues to report model weights and sequence reservations.
`GetMetrics.allocator` independently samples the process's LibTorch native allocator
on the selected device, without synchronizing GPU work or evicting caches:

- `active_bytes`: active blocks, including allocations awaiting safe release.
- `cached_bytes`: reserved minus active bytes. This is reusable allocator storage,
  not a guarantee that all bytes can satisfy a particular allocation or be returned
  to the driver immediately.
- `peak_bytes`: lifetime peak active bytes, including framework workspaces.

These values include allocations outside model accounting, but exclude CUDA context,
driver and other non-allocator allocations. They are not per-request measurements.
They must not be added to model reservations or NVIDIA process usage. Use external
process observations separately when assessing physical headroom. Allocator telemetry
is omitted for other allocator implementations (including `cudaMallocAsync`), whose
counters do not have these semantics. Missing telemetry means unavailable, not zero.

The change does not impose a new hard device-memory cap or clear the cache on unload.
Configured admission budgets remain separate from physical/framework memory usage.
See [the measured reload comparison](validation/cuda-memory-reuse.md).

## Repeated-assignment measurement

Start a freshly built worker using the [Milestone 4 instructions](milestone-4.md).
With its pinned prepared manifest and independent reference available:

```bash
uv run python scripts/validation/checkpoint_memory.py \
  build/qualification/manifest.json build/qualification/reference-f32.json \
  build/qualification/cuda-reloads.json \
  --workers cuda --cuda-endpoint 127.0.0.1:50163 --cycles 8 --new-tokens 16
```

The worker must start without a loaded model or active request. Each cycle loads the
same explicit F16 assignment, generates a reference-checked continuation, then
unloads it. Samples record both reservation and allocator reports after load, at the
first observed token, after request completion, and after unload. Sequence and model
cleanup must pass. An observed first-token sample can arrive after native execution
has advanced; it is not an exact prefill peak. Generation timings include telemetry
sampling overhead and are not compute or transport benchmarks.

For split assignments, use `--workers mlx cuda` or `--workers cuda mlx`, `--split`,
and both endpoints. Run `memory_watch.py` alongside the owned workers to collect RSS
and NVIDIA process usage. Compare fresh processes with the same checkpoint, workload,
execution dtype and allocator configuration; preserve the source revision and toolchain
in the measurement report. Do not extrapolate 0.6B results to 4B or arbitrary contexts.

## Acceptance

The acceptance workload is 512 prompt + 256 output tokens at concurrency 1 with
768-token capacity, using the [approved precision](milestone-5-precision-proposal.md).
The criteria are in
[Freeze and execute the sweep](milestone-5-placement.md#freeze-and-execute-the-sweep);
a restart follows the
[closeout restart gates](validation/milestone-5-wan-closeout.md#deferred-gates-and-restart-point).
The original slice plan is [archived](archive/milestone-5-implementation-plan.md).
Results are on [the status page](STATUS.md).
