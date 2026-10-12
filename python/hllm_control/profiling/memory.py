"""Conservative admission/physical envelopes; accounting views stay separate."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import Field

from hllm_control.models import Backend, NonNegativeInt
from hllm_control.profiling.models import (
    MemoryAmounts,
    MemoryMeasurement,
    ProfileArtifact,
    ProfileModel,
)


class MlxFitPolicy(StrEnum):
    CONSERVATIVE = "conservative-v1"
    FOOTPRINT = "footprint-v1"


# Evaluation default since October 10, 2026, after the two-host serving qualification met
# the promotion criterion. Stored bindings (measured bundles, frozen sweeps, native
# executor configurations) keep "omitted means conservative-v1", so historical digests
# and frozen identities are unchanged; new MLX bindings state footprint-v1 explicitly.
DEFAULT_MLX_FIT_POLICY = MlxFitPolicy.FOOTPRINT


class PhysicalBudget(ProfileModel):
    # Available to a fresh process, sampled before launch; not total machine capacity.
    available_bytes: NonNegativeInt | None
    headroom_bytes: NonNegativeInt
    # Includes gRPC/transport buffers absent from isolated-stage execution.
    extra_overhead_bytes: NonNegativeInt
    safety_fraction: float = Field(ge=0.05, lt=1, default=0.10)


class FitResult(ProfileModel):
    status: Literal["safe", "unsafe", "unknown"]
    reasons: tuple[str, ...]
    host_envelope_bytes: NonNegativeInt | None
    device_envelope_bytes: NonNegativeInt | None
    # Absent in historical results; do not change their serialized representation.
    policy: (
        Literal[
            "cpu-rss-v1", "cuda-rss-device-v1", "mlx-rss-plus-allocator-v1", "mlx-footprint-max-v1"
        ]
        | None
    ) = Field(default=None, exclude_if=lambda v: v is None)
    requested_mlx_policy: MlxFitPolicy | None = Field(default=None, exclude_if=lambda v: v is None)
    policy_notes: tuple[str, ...] = Field(default=(), exclude_if=lambda v: not v)


def _footprint_peak(artifact: ProfileArtifact, m: MemoryMeasurement) -> tuple[int | None, str]:
    """Only complete, coherent OS/allocator observations can replace the union bound."""
    if artifact.schema_version != "1.3" or not m.completed or not m.samples:
        return None, "footprint policy requires a complete schema-1.3 memory exercise"
    if artifact.conditions.process_policy != "fresh-process-then-reloads":
        return None, "footprint policy requires fresh-process/reload evidence"
    previous_time = previous_rss = previous_footprint = 0
    process_id: int | None = None
    for sample in m.samples:
        p = sample.process_memory
        if p is None or any(
            v is None or v <= 0
            for v in (
                p.rss_bytes,
                p.rss_lifetime_peak_bytes,
                p.physical_footprint_bytes,
                p.physical_footprint_lifetime_peak_bytes,
            )
        ):
            return None, "footprint policy requires positive OS counters in every phase"
        # The explicit checks also narrow optional values for static typing.
        assert p.rss_bytes is not None and p.rss_lifetime_peak_bytes is not None
        assert p.physical_footprint_bytes is not None
        assert p.physical_footprint_lifetime_peak_bytes is not None
        if (
            (process_id is not None and p.process_id != process_id)
            or p.observed_at_unix_ns < previous_time
            or p.rss_lifetime_peak_bytes < max(previous_rss, p.rss_bytes)
            or p.physical_footprint_lifetime_peak_bytes
            < max(previous_footprint, p.physical_footprint_bytes)
        ):
            return (
                None,
                "footprint policy requires coherent process identity, clock and lifetime peaks",
            )
        if sample.allocator is None or (
            sample.phase in ("load", "allocate", "prefill", "decode")
            and sample.allocator.peak_scope != "phase"
        ):
            return (
                None,
                "footprint policy requires allocator observations and execution phase peaks",
            )
        process_id = p.process_id
        previous_time = p.observed_at_unix_ns
        previous_rss = p.rss_lifetime_peak_bytes
        previous_footprint = p.physical_footprint_lifetime_peak_bytes
    return previous_footprint, ""


def assess_fit(
    artifact: ProfileArtifact,
    admission_capacity: MemoryAmounts,
    host: PhysicalBudget,
    device: PhysicalBudget | None = None,
    *,
    mlx_policy: MlxFitPolicy = DEFAULT_MLX_FIT_POLICY,
) -> FitResult:
    """Assess a compatible assignment with explicit, fresh physical budgets.

    Existing native load admission is authoritative. Without an exact load-peak
    formula in the artifact, decreasing any successful probe cap requires reprofiling.
    CPU uses process RSS. MLX defaults to the footprint policy: the maximum of OS
    footprint, RSS and allocator envelopes, only with complete telemetry; otherwise it
    falls back to the conservative RSS-plus-allocator sum and records why.
    CUDA uses process GPU observations independently.
    All physical observations retain safety and explicit non-profiled overhead.
    """
    m = artifact.measurement
    mlx_policy = MlxFitPolicy(mlx_policy)
    if not isinstance(m, MemoryMeasurement):
        raise ValueError("fit assessment requires a memory measurement")
    unsafe: list[str] = []
    unknown: list[str] = []
    if not m.completed:
        unknown.append("incomplete memory exercise: " + (m.error or "unknown error"))
    for name in MemoryAmounts.model_fields:
        if getattr(admission_capacity, name) < getattr(m.admission_capacity, name):
            unknown.append(f"{name}: lower admission cap requires a new load/run probe")
    for sample in m.samples:
        for name in MemoryAmounts.model_fields:
            required = sum(
                getattr(a, name) for a in (sample.weights, sample.cache, sample.workspace)
            )
            if required > getattr(admission_capacity, name):
                unsafe.append(f"{name}: native reservations exceed admission cap")
    host_values = [
        s.rss_lifetime_peak_bytes for s in m.samples if s.rss_lifetime_peak_bytes is not None
    ]
    host_values.extend(s.rss_bytes for s in m.physical_samples if s.rss_bytes is not None)
    device_values = [
        s.device_process_bytes for s in m.physical_samples if s.device_process_bytes is not None
    ]
    backend = artifact.key.environment.backend
    policy: Literal[
        "cpu-rss-v1", "cuda-rss-device-v1", "mlx-rss-plus-allocator-v1", "mlx-footprint-max-v1"
    ] = (
        "mlx-rss-plus-allocator-v1"
        if backend == Backend.MLX
        else "cuda-rss-device-v1"
        if backend == Backend.CUDA
        else "cpu-rss-v1"
    )
    policy_notes: tuple[str, ...] = ()
    unified_allocator_peak = 0
    if backend != Backend.CPU:
        if any(
            s.allocator is None or s.allocator.peak_scope != "phase"
            for s in m.samples
            if s.phase in ("load", "allocate", "prefill", "decode")
        ):
            unknown.append("backend phase peak telemetry unavailable")
        allocator_values = [
            max(s.allocator.peak_bytes, s.allocator.active_bytes + s.allocator.cached_bytes)
            for s in m.samples
            if s.allocator is not None
        ]
        if backend == Backend.MLX:
            unified_allocator_peak = max(allocator_values, default=0)
        else:
            # Still require physical CUDA process observations for context/driver cost.
            if not device_values:
                unknown.append("CUDA physical process observations unavailable")
            device_values.extend(allocator_values)
    host_peak = max(host_values, default=None)
    if host_peak is not None:
        host_peak += unified_allocator_peak
    if backend == Backend.MLX and mlx_policy == MlxFitPolicy.FOOTPRINT:
        footprint, reason = _footprint_peak(artifact, m)
        if footprint is None:
            policy_notes = (reason + "; using conservative-v1",)
        else:
            # Native RSS current values are included as an additional cross-check.
            host_peak = max(
                footprint,
                unified_allocator_peak,
                *host_values,
                *(s.rss_bytes or 0 for s in m.samples),
            )
            policy = "mlx-footprint-max-v1"
    device_peak = max(device_values, default=None)

    def envelope(name: str, peak: int | None, budget: PhysicalBudget | None) -> int | None:
        if peak is None or budget is None or budget.available_bytes is None:
            unknown.append(f"{name}: physical peak/current availability unavailable")
            return None
        needed = peak + math.ceil(peak * budget.safety_fraction) + budget.extra_overhead_bytes
        if needed + budget.headroom_bytes > budget.available_bytes:
            unsafe.append(f"{name}: physical envelope plus headroom exceeds current availability")
        return needed

    host_envelope = envelope("host/unified", host_peak, host)
    device_envelope = envelope("device", device_peak, device) if backend == Backend.CUDA else None
    return FitResult(
        status="unsafe" if unsafe else "unknown" if unknown else "safe",
        reasons=tuple(dict.fromkeys(unsafe + unknown)),
        host_envelope_bytes=host_envelope,
        device_envelope_bytes=device_envelope,
        policy=policy,
        requested_mlx_policy=mlx_policy if backend == Backend.MLX else None,
        policy_notes=policy_notes,
    )
