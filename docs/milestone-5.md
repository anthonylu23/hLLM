# Milestone 5 — measured placement

M5.1–5.4 implement measured-profile contracts, isolated assignment-memory probes,
paired native compute/conversion timing and directional gRPC qualification. See the
[profiling workflow](milestone-5-profiling.md),
[memory report](validation/milestone-5-memory.md), and
[timing report](validation/milestone-5-timing.md). Measured split selection (5.5) and the independent sweep runner (5.6) are now
implemented. The full 15% exhaustive acceptance comparison remains pending. See the
[placement workflow](milestone-5-placement.md) and
[current validation/gates](validation/milestone-5-planner.md).

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

The [implementation plan](milestone-5-implementation-plan.md) breaks the remaining
work into six reviewable slices:

1. **Implemented:** versioned measured profiles, compatibility checks and workload fixtures.
2. **Implemented:** assignment dry-load/context memory measurements and physical-fit checks.
3. **Implemented:** paired native prefill/decode and conversion profiles.
4. **Implemented:** payload-specific directional native transport profiles.
5. **Implemented:** measured automatic selection, explanations and activation checks.
6. **Runner implemented; acceptance pending:** independent exhaustive qualification against 15%.

The agreed initial workload is 512 prompt + 256 generated tokens at concurrency 1,
with 768-token capacity. The existing 32K cache-capacity example remains a separate
qualification target. Develop and qualify the profiling pipeline on 0.6B, then assess
the larger target's physical fit before attempting its qualification. The proposed
acceptance objective and detailed validation gates are recorded in the plan.

Next: collect production setup calibration and compatible current-build profiles for
all splits, freeze the automatic selection, then execute the 54-candidate 0.6B sweep
when conservative Mac headroom permits. The separate 4B fit gate and 32K-capacity
qualification remain pending; neither is implied by the tiny runner smoke test.
