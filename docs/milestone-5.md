# Milestone 5 — measured placement

The first implementation slice adds CUDA allocator telemetry, stable stream reuse,
and a repeatable assignment-memory probe. Automatic placement still uses configured
profiles. Layer compute, conversion and directional link profiling, measured split
selection, and the 15% comparison against the best feasible split remain unfinished.

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

## Next implementation steps

1. Add measured profile provenance and compatibility checks (checkpoint digest,
   backend/toolchain, dtype, context, stage ownership, and measurement conditions).
2. Measure dry-load and per-context peaks for candidate assignments while accounting
   separately for allocator residency and physical process overhead.
3. Profile native prefill/decode, conversion and directional boundary transfer, then
   feed those measurements into stage-order and split selection.
4. Compare predictions against measured feasible splits and establish the milestone's
   accuracy criterion. Assess the larger target's physical fit before running it.
