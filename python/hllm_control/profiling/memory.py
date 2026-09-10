"""Conservative admission/physical envelopes; accounting views stay separate."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from hllm_control.models import Backend, NonNegativeInt
from hllm_control.profiling.models import (
    MemoryAmounts,
    MemoryMeasurement,
    ProfileArtifact,
    ProfileModel,
)


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


def assess_fit(
    artifact: ProfileArtifact,
    admission_capacity: MemoryAmounts,
    host: PhysicalBudget,
    device: PhysicalBudget | None = None,
) -> FitResult:
    """Assess a compatible assignment with explicit, fresh physical budgets.

    Existing native load admission is authoritative. Without an exact load-peak
    formula in the artifact, decreasing any successful probe cap requires reprofiling.
    CPU uses process RSS. For MLX, RSS plus allocator residency is an intentionally
    overcounted upper bound: their overlap is unknown. It is not reported as measured
    physical usage. CUDA uses process GPU observations independently.
    All physical observations retain safety and explicit non-profiled overhead.
    """
    m = artifact.measurement
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
    )
