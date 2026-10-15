"""Concurrency-two serving evidence, separate from concurrency-one profiles.

A `ConcurrentServingEvidence` record binds one bounded two-request experiment: the
exact serving configuration, the single-request preconditions, every worker's own
request-lifecycle observations and the resulting token comparisons. Everything the
record concludes is derived from those observations by this module and re-derived on
every load: overlap intervals, logical peaks, per-cycle physical envelopes, guard
outcomes and the acceptance status. A record therefore cannot claim overlap, exact
output, a safe envelope, a protective guard or cleanup that its own contents do not
show. `ProfileKey` and measured placement stay at concurrency one; nothing here feeds a
placement bundle.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from hllm_control.models import Backend, DType, NonNegativeInt, PositiveInt
from hllm_control.profiling.memory import MlxFitPolicy, PhysicalBudget
from hllm_control.profiling.models import (
    AllocatorSample,
    Digest,
    Environment,
    Label,
    MemoryAmounts,
    ProcessMemoryObservation,
    ProfileModel,
    digest,
)

SCHEMA_VERSION = "1.1"
PROFILE_FRESHNESS = timedelta(hours=24)
REQUIRED_CANCEL_PREFIX = 3
# A guard samples every two seconds; a record mirrored from another host may lag a
# little more. Observations older than this at the end of the experiment do not cover it.
GUARD_STALENESS_SECONDS = 20.0
Status = Literal["accepted", "failed", "unknown"]
Policy = Literal[
    "cpu-rss-v1", "cuda-rss-device-v1", "mlx-rss-plus-allocator-v1", "mlx-footprint-max-v1"
]
BaselineScope = Literal["f16-reload-qualification", "cpu-rehearsal"]


def add_amounts(a: MemoryAmounts, b: MemoryAmounts) -> MemoryAmounts:
    return MemoryAmounts(
        **{name: getattr(a, name) + getattr(b, name) for name in MemoryAmounts.model_fields}
    )


def max_amounts(values: Iterable[MemoryAmounts]) -> MemoryAmounts:
    result = MemoryAmounts()
    for value in values:
        result = MemoryAmounts(
            **{
                name: max(getattr(result, name), getattr(value, name))
                for name in MemoryAmounts.model_fields
            }
        )
    return result


def fits(required: MemoryAmounts, capacity: MemoryAmounts) -> bool:
    return all(
        getattr(required, name) <= getattr(capacity, name) for name in MemoryAmounts.model_fields
    )


# ---------------------------------------------------------------------------- identity


class WorkerIdentity(ProfileModel):
    worker_id: Label
    stage_index: NonNegativeInt
    backend: Backend
    environment: Environment  # OS, backend, driver, allocator, source and profiler binary.
    worker_binary_digest: Digest
    endpoint: Label
    transport_mode: Literal["pageable", "pinned"]
    # Observed before the first load; every later process observation must match it.
    process_id: PositiveInt


class Identity(ProfileModel):
    source_digest: Label
    manifest_digest: Digest
    checkpoint_digest: Digest
    plan_id: Label
    plan_digest: Label
    reference_sha256: Digest
    transport: Label  # For example "loopback over ssh -L/-R forwarding"; not WAN acceptance.
    driver: Literal["native-controller"]  # Requests enter through the gRPC controller.
    worker_order: tuple[Label, Label]
    workers: tuple[WorkerIdentity, WorkerIdentity]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if tuple(w.worker_id for w in self.workers) != self.worker_order or tuple(
            w.stage_index for w in self.workers
        ) != (0, 1):
            raise ValueError("worker identities must follow the stage order")
        return self


# ---------------------------------------------------------------------------- workload


class RequestWorkload(ProfileModel):
    prompt_ids: Annotated[tuple[NonNegativeInt, ...], Field(min_length=1)]
    expected_ids: Annotated[tuple[NonNegativeInt, ...], Field(min_length=REQUIRED_CANCEL_PREFIX)]
    prompt_digest: Digest
    expected_digest: Digest
    reservation_tokens: PositiveInt

    @model_validator(mode="after")
    def hashed(self) -> Self:
        if (
            self.prompt_digest != digest({"ids": list(self.prompt_ids)})
            or self.expected_digest != digest({"ids": list(self.expected_ids)})
            or self.reservation_tokens != len(self.prompt_ids) + len(self.expected_ids)
        ):
            raise ValueError("request workload digests or reservation do not match the tokens")
        return self


class Workload(ProfileModel):
    requests: tuple[RequestWorkload, RequestWorkload]
    weight_dtype: DType | None = Field(default=None, exclude_if=lambda v: v is None)
    execution_dtype: DType
    kv_dtype: DType
    activation_dtype: DType
    sampling: Literal["greedy"]
    timeout_seconds: Annotated[float, Field(gt=0, le=3600)]


# ----------------------------------------------------------------------- configuration


class WorkerConfiguration(ProfileModel):
    worker_id: Label
    admission_capacity: MemoryAmounts
    maximum_active_requests: Annotated[int, Field(ge=2)]
    maximum_cached_tokens: NonNegativeInt  # Zero means memory-only admission.
    maximum_decode_batch: PositiveInt
    prefill_chunk_tokens: NonNegativeInt
    request_observations: bool  # Capability advertised by the running worker.


class Configuration(ProfileModel):
    workers: tuple[WorkerConfiguration, WorkerConfiguration]
    queue_capacity: NonNegativeInt  # HTTP queue; zero for the native controller driver.
    controller_limit: Annotated[int, Field(ge=2)]
    mlx_fit_policy: MlxFitPolicy
    safety_fraction: Annotated[float, Field(ge=0.05, lt=1)]
    extra_overhead_bytes: NonNegativeInt
    host_headroom_bytes: NonNegativeInt
    device_headroom_bytes: NonNegativeInt


# ----------------------------------------------------------------------- preconditions


class BaselineSummary(ProfileModel):
    """What the passing single-request report said, so the record can bind it."""

    scope: BaselineScope
    completed: bool
    plan_digest: Label
    reference_sha256: Digest
    worker_binary_digests: tuple[tuple[Label, Digest], ...]  # In stage order.
    cycles: PositiveInt
    requests: PositiveInt  # Completed requests across all cycles.
    all_exact: bool  # Every request and cancellation prefix matched the reference.
    finished_unix_time: Annotated[float, Field(ge=0)]


class ProfilePrecondition(ProfileModel):
    worker_id: Label
    artifact_digest: Digest
    measured_at: AwareDatetime
    requested_mlx_policy: MlxFitPolicy | None = Field(default=None, exclude_if=lambda v: v is None)


class CycleGate(ProfileModel):
    """The fresh single-request fit sampled before one cycle's load, with its budgets."""

    worker_id: Label
    cycle: NonNegativeInt
    sampled_at: AwareDatetime
    status: Literal["safe", "unsafe", "unknown"]
    reasons: tuple[str, ...]
    host_budget: PhysicalBudget  # Live availability sampled before this cycle's load.
    device_budget: PhysicalBudget | None = None


class Preconditions(ProfileModel):
    baseline_report_sha256: Digest  # The passing single-request serving report.
    baseline: BaselineSummary
    started_at: AwareDatetime
    profiles: tuple[ProfilePrecondition, ProfilePrecondition]
    gates: tuple[CycleGate, ...]  # Ordered by cycle, then stage order.
    guard: Label  # Guard settings in words; outcomes are in results.

    @model_validator(mode="after")
    def ordered(self) -> Self:
        keys = [(g.cycle, g.worker_id) for g in self.gates]
        if len(set(keys)) != len(keys):
            raise ValueError("one fit gate per worker and cycle")
        if [g.cycle for g in self.gates] != sorted(g.cycle for g in self.gates):
            raise ValueError("fit gates must be ordered by cycle")
        return self


# ------------------------------------------------------------------------ observations


class RequestReservation(ProfileModel):
    request_id: Label
    maximum_total_tokens: PositiveInt
    cache: MemoryAmounts
    workspace: MemoryAmounts
    admitted_at_monotonic_ns: NonNegativeInt
    allocating: bool
    running: bool
    cancelled: bool


class LifecycleEvent(ProfileModel):
    kind: Literal["admitted", "retired"]
    request_id: Label
    sequence: PositiveInt
    at_monotonic_ns: NonNegativeInt
    concurrent_request_ids: tuple[Label, ...]
    # Retirement only: completed, cancelled, expired, failed, allocation-failed.
    # Older workers recorded failures as completed and running expiry as cancelled.
    reason: str


class WorkerSnapshot(ProfileModel):
    phase: Label
    cycle: NonNegativeInt | None  # None before the first cycle's fit gate.
    unix_time: Annotated[float, Field(ge=0)]
    process_id: PositiveInt | None  # From request observations; None when unavailable.
    observed_at_monotonic_ns: NonNegativeInt | None
    observed_at_unix_ns: NonNegativeInt | None
    loaded_weights: MemoryAmounts
    reserved_cache: MemoryAmounts
    reserved_workspace: MemoryAmounts
    active_request_ids: tuple[Label, ...]
    request_observations_available: bool
    requests: tuple[RequestReservation, ...]
    events: tuple[LifecycleEvent, ...]
    events_total: NonNegativeInt
    peak_concurrent_requests: NonNegativeInt
    allocator: AllocatorSample | None
    process_memory: ProcessMemoryObservation | None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if not self.request_observations_available and (
            self.requests or self.events or self.process_id is not None
        ):
            raise ValueError("unavailable request observations cannot carry request rows")
        if self.request_observations_available and (
            self.process_id is None or self.observed_at_monotonic_ns is None
        ):
            raise ValueError("request observations require process identity and a steady clock")
        if {r.request_id for r in self.requests} != set(self.active_request_ids):
            raise ValueError("request rows must match the active request IDs")
        if len(self.events) > self.events_total:
            raise ValueError("more events than the lifetime total")
        return self


class DeviceSample(ProfileModel):
    """One out-of-band process-device observation (for example `memory_watch.py --cuda`)."""

    unix_time: Annotated[float, Field(ge=0)]
    process_id: PositiveInt
    cuda_process_bytes: NonNegativeInt | None  # None when the counter was unavailable.


class WorkerObservations(ProfileModel):
    worker_id: Label
    snapshots: tuple[WorkerSnapshot, ...]  # Possibly empty after an early failure.
    device_samples: tuple[DeviceSample, ...] = ()


class Observations(ProfileModel):
    workers: tuple[WorkerObservations, WorkerObservations]
    # Cross-host order is never inferred from wall clocks; say what is known.
    clock_alignment: Label


# ----------------------------------------------------------------------------- results


class Round(ProfileModel):
    cycle: NonNegativeInt
    index: NonNegativeInt
    kind: Literal["serial", "paired", "cancel"]
    request_ids: Annotated[tuple[Label, ...], Field(min_length=1, max_length=2)]

    @model_validator(mode="after")
    def arity(self) -> Self:
        if (self.kind == "serial") != (len(self.request_ids) == 1):
            raise ValueError("serial rounds have one request; paired and cancel rounds two")
        return self


class TokenComparison(ProfileModel):
    request_id: Label
    role: Literal["serial", "paired", "cancel-victim", "cancel-survivor"]
    generated_ids: tuple[NonNegativeInt, ...]
    expected_ids: tuple[NonNegativeInt, ...]
    exact: bool
    seconds: Annotated[float, Field(ge=0)]
    error: str | None = None

    @model_validator(mode="after")
    def honest(self) -> Self:
        if self.exact != (self.generated_ids == self.expected_ids):
            raise ValueError("exact flag contradicts the recorded tokens")
        return self


class OverlapInterval(ProfileModel):
    worker_id: Label
    request_ids: tuple[Label, Label]
    start_monotonic_ns: NonNegativeInt  # Later admission of the pair on this worker's clock.
    end_monotonic_ns: NonNegativeInt  # First retirement of the pair, or last joint snapshot.
    admission_sequence: PositiveInt | None  # None when the admissions left the bounded log.
    # How simultaneous allocated reservations were shown: the worker's own retirement
    # event listing the other member as allocated, or a polled snapshot with both rows.
    proof: Literal["retirement-event", "joint-snapshot"]
    detail: Label


class WorkerPeak(ProfileModel):
    worker_id: Label
    weights: MemoryAmounts  # Counted once per worker.
    cache: MemoryAmounts  # Largest simultaneous per-request sum.
    workspace: MemoryAmounts
    combined: MemoryAmounts
    admission_capacity: MemoryAmounts
    within_admission: bool

    @model_validator(mode="after")
    def arithmetic(self) -> Self:
        if self.combined != add_amounts(self.weights, add_amounts(self.cache, self.workspace)):
            raise ValueError("combined logical peak must be weights plus cache plus workspace")
        if self.within_admission != fits(self.combined, self.admission_capacity):
            raise ValueError("admission flag contradicts the combined peak")
        return self


class PhysicalEnvelope(ProfileModel):
    """One worker's serving envelope for one cycle against that cycle's live budget."""

    worker_id: Label
    cycle: NonNegativeInt
    policy: Policy
    requested_mlx_policy: MlxFitPolicy | None = Field(default=None, exclude_if=lambda v: v is None)
    policy_notes: tuple[str, ...] = ()
    host_peak_bytes: NonNegativeInt | None
    device_peak_bytes: NonNegativeInt | None
    host_envelope_bytes: NonNegativeInt | None
    device_envelope_bytes: NonNegativeInt | None
    host_budget: PhysicalBudget
    device_budget: PhysicalBudget | None
    status: Literal["safe", "unsafe", "unknown"]
    reasons: tuple[str, ...]


class GuardOutcome(ProfileModel):
    """What one worker's guard record contains; the outcome is derived from these fields."""

    worker_id: Label
    record_sha256: Digest | None
    process_id: PositiveInt | None  # The process the guard spawned and watched.
    started_unix_time: Annotated[float, Field(ge=0)] | None
    last_observed_unix_time: Annotated[float, Field(ge=0)] | None
    exited_unix_time: Annotated[float, Field(ge=0)] | None
    observations: NonNegativeInt
    exit_reason: str | None
    returncode: int | None
    detail: str  # Free text from the harness, for example why a record was unreadable.
    outcome: Literal["normal", "stopped", "unavailable"]  # Derived; re-derived on load.
    reasons: tuple[str, ...]  # Derived; re-derived on load.


class CleanupEvidence(ProfileModel):
    worker_id: Label
    verified: bool
    detail: str


class Results(ProfileModel):
    finished_at: AwareDatetime
    rounds: tuple[Round, ...]
    comparisons: tuple[TokenComparison, ...]
    overlap: tuple[OverlapInterval, ...]  # Derived; re-derived on load.
    logical_peaks: tuple[WorkerPeak, WorkerPeak]  # Derived; re-derived on load.
    physical: tuple[PhysicalEnvelope, ...]  # Derived per worker and cycle; re-derived on load.
    guards: tuple[GuardOutcome, GuardOutcome]
    cleanup: tuple[CleanupEvidence, CleanupEvidence]
    failure_reason: str | None
    acceptance: Status
    acceptance_reasons: tuple[str, ...]


# ----------------------------------------------------------------------------- record


class ConcurrentServingEvidence(ProfileModel):
    schema_version: Literal["1.1"]
    evidence_digest: Digest
    identity: Identity
    workload: Workload
    configuration: Configuration
    preconditions: Preconditions
    observations: Observations
    results: Results

    @model_validator(mode="after")
    def bound(self) -> Self:
        order = self.identity.worker_order
        for name, ids in (
            ("configuration", tuple(w.worker_id for w in self.configuration.workers)),
            ("preconditions", tuple(p.worker_id for p in self.preconditions.profiles)),
            ("observations", tuple(w.worker_id for w in self.observations.workers)),
            ("logical peaks", tuple(p.worker_id for p in self.results.logical_peaks)),
            ("guards", tuple(g.worker_id for g in self.results.guards)),
            ("cleanup", tuple(c.worker_id for c in self.results.cleanup)),
        ):
            if ids != order:
                raise ValueError(f"{name} must list the workers in stage order")
        if any(g.worker_id not in order for g in self.preconditions.gates):
            raise ValueError("fit gates must name the record's workers")
        c = self.configuration
        for gate in self.preconditions.gates:
            for budget, headroom in (
                (gate.host_budget, c.host_headroom_bytes),
                (gate.device_budget, c.device_headroom_bytes),
            ):
                if budget is not None and (
                    budget.headroom_bytes != headroom
                    or budget.extra_overhead_bytes != c.extra_overhead_bytes
                    or budget.safety_fraction != c.safety_fraction
                ):
                    raise ValueError(
                        f"{gate.worker_id}: cycle {gate.cycle} fit gate used allowances "
                        "other than the recorded configuration"
                    )
        content = self.model_dump(mode="json", exclude={"evidence_digest"})
        if self.evidence_digest != digest(content):
            raise ValueError("evidence digest mismatch")
        if self.results.overlap != derive_overlap(self.observations, self.results.rounds):
            raise ValueError("recorded overlap intervals are not what the observations show")
        peaks = derive_logical_peaks(self.observations, self.configuration)
        if self.results.logical_peaks != peaks:
            raise ValueError("recorded logical peaks are not what the observations show")
        physical = derive_physical(
            self.identity,
            self.configuration,
            self.preconditions,
            self.observations,
            self.results.finished_at,
        )
        if self.results.physical != physical:
            raise ValueError("recorded physical envelopes are not what the observations show")
        for guard, worker in zip(self.results.guards, self.identity.workers, strict=True):
            outcome, reasons = classify_guard(
                guard, worker.process_id, self.preconditions.started_at, self.results.finished_at
            )
            if (guard.outcome, guard.reasons) != (outcome, reasons):
                raise ValueError(f"{guard.worker_id}: recorded guard outcome is not derived")
        status, reasons = assess_evidence(self)
        if (self.results.acceptance, self.results.acceptance_reasons) != (status, reasons):
            raise ValueError(f"recorded acceptance {self.results.acceptance!r} is not derived")
        return self


# ------------------------------------------------------------------------- derivations


def _allocated(snapshot: WorkerSnapshot, identifier: str) -> RequestReservation | None:
    for r in snapshot.requests:
        if (
            r.request_id == identifier
            and not r.allocating
            and sum(getattr(r.cache, n) for n in MemoryAmounts.model_fields) > 0
        ):
            return r
    return None


def derive_overlap(
    observations: Observations, rounds: Iterable[Round]
) -> tuple[OverlapInterval, ...]:
    """Per-worker overlap from the worker's own lifecycle events, never from poll luck.

    An interval needs proof that both members held allocated KV at one instant: the
    worker's own retirement event for either member listing the other as an allocated
    concurrent reservation (reservations are erased under one lock, so the earlier
    retirement sees the later one live), otherwise a polled snapshot carrying both
    allocated rows. The interval starts at the later admission, taken from the admission
    events or, if they left the bounded log, from the admission times the joint snapshot
    rows carry. Admission events list only peers whose allocation had completed, so two
    requests reserving together list nobody; that is not evidence against overlap.
    Missing evidence yields no interval, never an inferred one.
    """
    intervals: list[OverlapInterval] = []
    for round_ in rounds:
        if round_.kind == "serial":
            continue
        pair = (round_.request_ids[0], round_.request_ids[1])
        for worker in observations.workers:
            events: dict[int, LifecycleEvent] = {}
            for snapshot in worker.snapshots:
                for event in snapshot.events:
                    events[event.sequence] = event
            admissions = {
                e.request_id: e
                for e in events.values()
                if e.kind == "admitted" and e.request_id in pair
            }
            joint = [
                s
                for s in worker.snapshots
                if s.request_observations_available
                and all(_allocated(s, identifier) is not None for identifier in pair)
            ]
            admission_sequence: int | None = None
            if len(admissions) == 2:
                later = max(admissions.values(), key=lambda e: e.sequence)
                start = later.at_monotonic_ns
                admission_sequence = later.sequence
                listed = any(
                    o in e.concurrent_request_ids for e in admissions.values() for o in pair
                )
                how = (
                    "later admission listed the earlier reservation as allocated"
                    if listed
                    else "admissions listed no allocated peer (reserved together)"
                )
            elif joint:
                rows = [_allocated(joint[0], identifier) for identifier in pair]
                start = max(r.admitted_at_monotonic_ns for r in rows if r is not None)
                how = "admission events absent from the bounded log; start from snapshot rows"
            else:
                continue
            joint = [s for s in joint if (s.observed_at_monotonic_ns or 0) >= start]
            retirements = sorted(
                (
                    e
                    for e in events.values()
                    if e.kind == "retired"
                    and e.request_id in pair
                    and e.at_monotonic_ns >= start
                    and (admission_sequence is None or e.sequence > admission_sequence)
                    and e.reason != "allocation-failed"
                ),
                key=lambda e: e.sequence,
            )
            witnessed = [
                e
                for e in retirements
                if any(o in e.concurrent_request_ids for o in pair if o != e.request_id)
            ]
            if witnessed:
                proof: Literal["retirement-event", "joint-snapshot"] = "retirement-event"
                detail = f"event {witnessed[0].sequence}: {witnessed[0].request_id} retired; {how}"
                end = witnessed[0].at_monotonic_ns
            elif joint:
                proof = "joint-snapshot"
                detail = f"snapshot {joint[0].phase}; {how}"
                end = (
                    retirements[0].at_monotonic_ns
                    if retirements
                    else max(s.observed_at_monotonic_ns or 0 for s in joint)
                )
            else:
                continue
            intervals.append(
                OverlapInterval(
                    worker_id=worker.worker_id,
                    request_ids=pair,
                    start_monotonic_ns=start,
                    end_monotonic_ns=max(end, start),
                    admission_sequence=admission_sequence,
                    proof=proof,
                    detail=detail,
                )
            )
    return tuple(intervals)


def derive_logical_peaks(
    observations: Observations, configuration: Configuration
) -> tuple[WorkerPeak, WorkerPeak]:
    peaks: list[WorkerPeak] = []
    for worker, config in zip(observations.workers, configuration.workers, strict=True):
        weights = max_amounts(s.loaded_weights for s in worker.snapshots)
        cache = max_amounts(s.reserved_cache for s in worker.snapshots)
        workspace = max_amounts(s.reserved_workspace for s in worker.snapshots)
        combined = add_amounts(weights, add_amounts(cache, workspace))
        peaks.append(
            WorkerPeak(
                worker_id=worker.worker_id,
                weights=weights,
                cache=cache,
                workspace=workspace,
                combined=combined,
                admission_capacity=config.admission_capacity,
                within_admission=fits(combined, config.admission_capacity),
            )
        )
    return (peaks[0], peaks[1])


def _envelope(
    peak: int | None,
    budget: PhysicalBudget | None,
    name: str,
    unsafe: list[str],
    unknown: list[str],
) -> int | None:
    if peak is None or budget is None or budget.available_bytes is None:
        unknown.append(f"{name}: physical peak/prelaunch availability unavailable")
        return None
    needed = peak + math.ceil(peak * budget.safety_fraction) + budget.extra_overhead_bytes
    if needed + budget.headroom_bytes > budget.available_bytes:
        unsafe.append(f"{name}: serving envelope plus headroom exceeds prelaunch availability")
    return needed


def assess_serving_envelope(
    worker_id: str,
    backend: Backend,
    snapshots: Iterable[WorkerSnapshot],
    *,
    cycle: int,
    host: PhysicalBudget,
    device: PhysicalBudget | None,
    mlx_policy: MlxFitPolicy,
    device_process_peak_bytes: int | None,
) -> PhysicalEnvelope:
    """Apply the selected policy once to one cycle's serving observations of one worker.

    MLX under `footprint-v1` takes the maximum of coherent OS footprint lifetime peaks,
    RSS and allocator envelopes, and falls back to the conservative sum with a note when
    any snapshot lacks that telemetry. CUDA additionally needs out-of-band process-device
    observations (for example `memory_watch.py --cuda`); without them the device side is
    unknown, never safe. Nothing is added per request.
    """
    rows = list(snapshots)
    unsafe: list[str] = []
    unknown: list[str] = []
    notes: list[str] = []
    if not rows:
        unknown.append("no serving snapshots recorded for this cycle")
    rss = [
        v
        for s in rows
        if s.process_memory is not None
        for v in (s.process_memory.rss_bytes, s.process_memory.rss_lifetime_peak_bytes)
        if v is not None
    ]
    if any(s.process_memory is None or s.process_memory.rss_bytes is None for s in rows):
        unknown.append("host: process RSS unavailable in some serving snapshots")
    allocator = [
        max(s.allocator.peak_bytes, s.allocator.active_bytes + s.allocator.cached_bytes)
        for s in rows
        if s.allocator is not None
    ]
    policy: Policy = "cpu-rss-v1"
    host_peak = max(rss, default=None)
    device_peak: int | None = None
    requested: MlxFitPolicy | None = None
    if backend == Backend.MLX:
        requested = mlx_policy
        policy = "mlx-rss-plus-allocator-v1"
        if any(s.allocator is None for s in rows):
            unknown.append("host/unified: allocator observations unavailable in some snapshots")
        if host_peak is not None:
            host_peak += max(allocator, default=0)
        if mlx_policy == MlxFitPolicy.FOOTPRINT:
            footprints: list[int] = []
            coherent = True
            previous_time = previous_peak = 0
            process = None
            for s in rows:
                p = s.process_memory
                if (
                    p is None
                    or not p.physical_footprint_bytes
                    or not p.physical_footprint_lifetime_peak_bytes
                    or not p.rss_lifetime_peak_bytes
                    or (process is not None and p.process_id != process)
                    or p.observed_at_unix_ns < previous_time
                    or p.physical_footprint_lifetime_peak_bytes
                    < max(previous_peak, p.physical_footprint_bytes)
                ):
                    coherent = False
                    break
                process = p.process_id
                previous_time = p.observed_at_unix_ns
                previous_peak = p.physical_footprint_lifetime_peak_bytes
                footprints.append(p.physical_footprint_lifetime_peak_bytes)
            if coherent and footprints and allocator:
                host_peak = max(*footprints, *rss, *allocator)
                policy = "mlx-footprint-max-v1"
            else:
                notes.append(
                    "footprint policy requires coherent footprint, RSS and allocator telemetry "
                    "in every serving snapshot; using conservative-v1"
                )
    elif backend == Backend.CUDA:
        policy = "cuda-rss-device-v1"
        if device_process_peak_bytes is None:
            unknown.append("device: CUDA physical process observations unavailable")
        elif not allocator:
            unknown.append("device: allocator observations unavailable")
        else:
            device_peak = max(device_process_peak_bytes, *allocator)
    host_envelope = _envelope(host_peak, host, "host/unified", unsafe, unknown)
    device_envelope = None
    if backend == Backend.CUDA:
        device_envelope = _envelope(device_peak, device, "device", unsafe, unknown)
    return PhysicalEnvelope(
        worker_id=worker_id,
        cycle=cycle,
        policy=policy,
        requested_mlx_policy=requested,
        policy_notes=tuple(notes),
        host_peak_bytes=host_peak,
        device_peak_bytes=device_peak,
        host_envelope_bytes=host_envelope,
        device_envelope_bytes=device_envelope,
        host_budget=host,
        device_budget=device if backend == Backend.CUDA else None,
        status="unsafe" if unsafe else "unknown" if unknown else "safe",
        reasons=tuple(dict.fromkeys(unsafe + unknown)),
    )


def derive_physical(
    identity: Identity,
    configuration: Configuration,
    preconditions: Preconditions,
    observations: Observations,
    finished_at: datetime,
) -> tuple[PhysicalEnvelope, ...]:
    """One envelope per worker and gated cycle, each against that cycle's live budget.

    A cycle's device observations are those sampled between its fit gate and the next
    gate (or the end of the experiment) on the sampling host's clock. Cycles with serving
    snapshots but no gate are reported by `assess_evidence`, not silently scored.
    """
    envelopes: list[PhysicalEnvelope] = []
    for worker, observed in zip(identity.workers, observations.workers, strict=True):
        gates = sorted(
            (g for g in preconditions.gates if g.worker_id == worker.worker_id),
            key=lambda g: g.cycle,
        )
        for index, gate in enumerate(gates):
            window_end = gates[index + 1].sampled_at if index + 1 < len(gates) else finished_at
            samples = [
                d.cuda_process_bytes
                for d in observed.device_samples
                if gate.sampled_at.timestamp() <= d.unix_time <= window_end.timestamp()
                and d.cuda_process_bytes is not None
            ]
            envelopes.append(
                assess_serving_envelope(
                    worker.worker_id,
                    worker.backend,
                    [s for s in observed.snapshots if s.cycle == gate.cycle],
                    cycle=gate.cycle,
                    host=gate.host_budget,
                    device=gate.device_budget,
                    mlx_policy=configuration.mlx_fit_policy,
                    device_process_peak_bytes=max(samples, default=None),
                )
            )
    return tuple(envelopes)


def classify_guard(
    guard: GuardOutcome, worker_process_id: int, started_at: datetime, finished_at: datetime
) -> tuple[Literal["normal", "stopped", "unavailable"], tuple[str, ...]]:
    """Did this guard protect this worker for the whole experiment, and did it stay quiet?

    A guard counts only if it spawned the worker process itself, started before the
    experiment, observed it until the end (within `GUARD_STALENESS_SECONDS`) and neither
    stopped the process nor saw it exit abnormally. Anything else is unavailable, never
    normal; a recorded stop reason, an early exit or a failing exit status is stopped.
    """
    stopped: list[str] = []
    unavailable: list[str] = []
    if guard.started_unix_time is None:
        return "unavailable", ("no guard start event",)
    if guard.process_id != worker_process_id:
        unavailable.append(
            f"guard watched process {guard.process_id}, not worker process {worker_process_id}"
        )
    if guard.started_unix_time > started_at.timestamp():
        unavailable.append("guard started after the experiment began")
    if guard.exit_reason:
        stopped.append(f"guard stopped the worker: {guard.exit_reason}")
    if guard.exited_unix_time is not None:
        if guard.exited_unix_time < finished_at.timestamp():
            stopped.append("guarded worker exited before the experiment finished")
        if guard.returncode not in (0, None):
            stopped.append(f"guarded worker exited with status {guard.returncode}")
    if guard.last_observed_unix_time is None:
        unavailable.append("guard recorded no observations")
    elif guard.last_observed_unix_time < finished_at.timestamp() - GUARD_STALENESS_SECONDS:
        lag = finished_at.timestamp() - guard.last_observed_unix_time
        unavailable.append(f"guard observations end {lag:.0f} s before the experiment finished")
    if stopped:
        return "stopped", tuple(stopped + unavailable)
    if unavailable:
        return "unavailable", tuple(unavailable)
    return "normal", ()


def assess_evidence(e: ConcurrentServingEvidence) -> tuple[Status, tuple[str, ...]]:
    """Derive acceptance: failed beats unknown; anything unobserved is unknown, never passed."""
    failed: list[str] = []
    unknown: list[str] = []
    r = e.results
    if r.failure_reason:
        failed.append("run failed: " + r.failure_reason)
    baseline = e.preconditions.baseline
    rehearsal = e.workload.execution_dtype == DType.F32
    if not baseline.completed or not baseline.all_exact:
        failed.append("baseline single-request report was incomplete or inexact")
    if baseline.scope != ("cpu-rehearsal" if rehearsal else "f16-reload-qualification"):
        failed.append(f"baseline scope {baseline.scope!r} does not match the workload precision")
    if baseline.plan_digest != e.identity.plan_digest:
        failed.append("baseline report was produced for a different plan")
    if baseline.reference_sha256 != e.identity.reference_sha256:
        failed.append("baseline report was produced against a different reference")
    binaries = tuple((w.worker_id, w.worker_binary_digest) for w in e.identity.workers)
    if baseline.worker_binary_digests != binaries:
        failed.append("baseline report was produced on different worker binaries")
    for p in e.preconditions.profiles:
        age = e.preconditions.started_at - p.measured_at
        if not timedelta(0) <= age <= PROFILE_FRESHNESS:
            failed.append(f"{p.worker_id}: single-request profile stale or future-dated")
    gated = {(g.worker_id, g.cycle) for g in e.preconditions.gates}
    for g in e.preconditions.gates:
        if g.status != "safe":
            failed.append(f"{g.worker_id}: fit gate before cycle {g.cycle} was {g.status}")
    for c in e.configuration.workers:
        if not c.request_observations:
            unknown.append(f"{c.worker_id}: worker did not advertise request observations")
    for w, identity in zip(e.observations.workers, e.identity.workers, strict=True):
        if not w.snapshots:
            unknown.append(f"{w.worker_id}: no snapshots recorded")
        if (w.worker_id, 0) not in gated:
            unknown.append(f"{w.worker_id}: no fit gate recorded before the first cycle")
        for cycle in sorted({s.cycle for s in w.snapshots if s.cycle is not None}):
            if (w.worker_id, cycle) not in gated:
                unknown.append(f"{w.worker_id}: cycle {cycle} has snapshots but no fit gate")
        processes = {s.process_id for s in w.snapshots if s.process_id is not None}
        processes |= {s.process_memory.process_id for s in w.snapshots if s.process_memory}
        if processes - {identity.process_id}:
            failed.append(f"{w.worker_id}: worker process differs from the initial identity")
        if any(d.process_id != identity.process_id for d in w.device_samples):
            failed.append(f"{w.worker_id}: device observations belong to another process")
        if any(not s.request_observations_available for s in w.snapshots):
            unknown.append(f"{w.worker_id}: request observations missing from some snapshots")
        clocks = [s.observed_at_monotonic_ns for s in w.snapshots if s.observed_at_monotonic_ns]
        if any(b < a for a, b in pairwise(clocks)):
            failed.append(f"{w.worker_id}: steady clock went backwards between snapshots")
    expected_roles = {
        "serial": ("serial",),
        "paired": ("paired", "paired"),
        "cancel": ("cancel-victim", "cancel-survivor"),
    }
    by_request = {c.request_id: c for c in r.comparisons}
    if len(by_request) != len(r.comparisons):
        failed.append("duplicate request IDs in token comparisons")
    overlapping = {(o.worker_id, o.request_ids) for o in r.overlap}
    for round_ in r.rounds:
        comparisons = [by_request.get(identifier) for identifier in round_.request_ids]
        if any(c is None for c in comparisons):
            failed.append(f"round {round_.cycle}/{round_.index}: missing token comparison")
            continue
        roles = tuple(c.role for c in comparisons if c is not None)
        if roles != expected_roles[round_.kind]:
            failed.append(f"round {round_.cycle}/{round_.index}: comparison roles mismatch")
        for position, c in enumerate(comparisons):
            assert c is not None
            sealed = e.workload.requests[position].expected_ids
            if c.error:
                failed.append(f"{c.request_id}: stream error: {c.error}")
            if not c.exact:
                failed.append(f"{c.request_id}: continuation differs from the reference")
            if c.role == "cancel-victim":
                if len(c.expected_ids) < REQUIRED_CANCEL_PREFIX:
                    failed.append(
                        f"{c.request_id}: cancelled before {REQUIRED_CANCEL_PREFIX} tokens"
                    )
                if (
                    len(c.expected_ids) >= len(sealed)
                    or sealed[: len(c.expected_ids)] != c.expected_ids
                ):
                    failed.append(
                        f"{c.request_id}: expected tokens are not a proper prefix of the "
                        "sealed reference"
                    )
            elif c.expected_ids != sealed:
                failed.append(f"{c.request_id}: expected tokens differ from the sealed reference")
        if round_.kind != "serial":
            pair = (round_.request_ids[0], round_.request_ids[1])
            for w in e.identity.worker_order:
                if (w, pair) not in overlapping:
                    unknown.append(f"{w}: overlap of {pair[0]} and {pair[1]} not observed")
        if round_.kind == "cancel":
            victim, survivor = round_.request_ids[0], round_.request_ids[1]
            for w in e.observations.workers:
                events = {ev.sequence: ev for s in w.snapshots for ev in s.events}
                cancelled = [
                    ev
                    for ev in events.values()
                    if ev.kind == "retired" and ev.request_id == victim and ev.reason == "cancelled"
                ]
                independent = [ev for ev in cancelled if survivor in ev.concurrent_request_ids]
                completed = [
                    ev
                    for ev in events.values()
                    if ev.kind == "retired"
                    and ev.request_id == survivor
                    and ev.reason == "completed"
                    and any(ev.sequence > c.sequence for c in cancelled)
                ]
                if not cancelled:
                    # A victim that retired for another reason (usually `completed`: the
                    # cancellation lost the race against its natural completion on this
                    # worker) is named so the operator can lengthen the reference or
                    # cancel earlier rather than guess.
                    other = sorted(
                        {
                            ev.reason
                            for ev in events.values()
                            if ev.kind == "retired" and ev.request_id == victim
                        }
                    )
                    detail = f" (it retired as {', '.join(other)})" if other else ""
                    unknown.append(
                        f"{w.worker_id}: no cancelled retirement of {victim} observed{detail}"
                    )
                elif not independent:
                    unknown.append(
                        f"{w.worker_id}: independent retirement of {victim} while {survivor} held "
                        "its reservation was not observed"
                    )
                if not completed:
                    unknown.append(
                        f"{w.worker_id}: completed retirement of {survivor} after the "
                        "cancellation was not observed"
                    )
    if not any(round_.kind == "paired" for round_ in r.rounds):
        unknown.append("no paired round recorded")
    if not any(round_.kind == "cancel" for round_ in r.rounds):
        unknown.append("no cancellation round recorded")
    for p in r.logical_peaks:
        if not p.within_admission:
            failed.append(f"{p.worker_id}: combined logical reservation exceeds the admission cap")
    for p in r.physical:
        if p.status == "unsafe":
            failed.append(f"{p.worker_id} cycle {p.cycle}: " + "; ".join(p.reasons))
        elif p.status == "unknown":
            unknown.append(f"{p.worker_id} cycle {p.cycle}: " + "; ".join(p.reasons))
    for g in r.guards:
        if g.outcome == "stopped":
            failed.append(f"{g.worker_id}: resource guard: " + "; ".join(g.reasons))
        elif g.outcome == "unavailable":
            unknown.append(f"{g.worker_id}: guard outcome unavailable: " + "; ".join(g.reasons))
    for c in r.cleanup:
        if not c.verified:
            failed.append(f"{c.worker_id}: cleanup not verified: {c.detail}")
    reasons = tuple(dict.fromkeys(failed + unknown))
    return ("failed" if failed else "unknown" if unknown else "accepted"), reasons


def make_evidence(
    *,
    identity: Identity,
    workload: Workload,
    configuration: Configuration,
    preconditions: Preconditions,
    observations: Observations,
    rounds: Iterable[Round],
    comparisons: Iterable[TokenComparison],
    guards: tuple[GuardOutcome, GuardOutcome],
    cleanup: tuple[CleanupEvidence, CleanupEvidence],
    finished_at: datetime,
    failure_reason: str | None,
) -> ConcurrentServingEvidence:
    """Assemble a record whose derived fields and acceptance come from its own contents.

    Guard outcomes are supplied as raw record fields; their classification, like the
    overlap, peaks, physical envelopes and acceptance, is derived here.
    """
    rounds = tuple(rounds)
    classified = tuple(
        g.model_copy(
            update=dict(
                zip(
                    ("outcome", "reasons"),
                    classify_guard(g, w.process_id, preconditions.started_at, finished_at),
                    strict=True,
                )
            )
        )
        for g, w in zip(guards, identity.workers, strict=True)
    )
    draft = Results(
        finished_at=finished_at,
        rounds=rounds,
        comparisons=tuple(comparisons),
        overlap=derive_overlap(observations, rounds),
        logical_peaks=derive_logical_peaks(observations, configuration),
        physical=derive_physical(identity, configuration, preconditions, observations, finished_at),
        guards=(classified[0], classified[1]),
        cleanup=cleanup,
        failure_reason=failure_reason,
        acceptance="unknown",
        acceptance_reasons=(),
    )
    provisional = ConcurrentServingEvidence.model_construct(
        schema_version=SCHEMA_VERSION,
        evidence_digest="0" * 64,
        identity=identity,
        workload=workload,
        configuration=configuration,
        preconditions=preconditions,
        observations=observations,
        results=draft,
    )
    status, reasons = assess_evidence(provisional)
    results = draft.model_copy(update={"acceptance": status, "acceptance_reasons": reasons})
    content: dict[str, object] = dict(
        schema_version=SCHEMA_VERSION,
        identity=identity.model_dump(mode="json"),
        workload=workload.model_dump(mode="json"),
        configuration=configuration.model_dump(mode="json"),
        preconditions=preconditions.model_dump(mode="json"),
        observations=observations.model_dump(mode="json"),
        results=results.model_dump(mode="json"),
    )
    return ConcurrentServingEvidence.model_validate({**content, "evidence_digest": digest(content)})


def write_evidence(path: Path, evidence: ConcurrentServingEvidence) -> None:
    """Immutable: an existing file is never replaced, whatever it contains."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(evidence.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")


def read_evidence(path: Path) -> ConcurrentServingEvidence:
    data: Mapping[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return ConcurrentServingEvidence.model_validate(data)
