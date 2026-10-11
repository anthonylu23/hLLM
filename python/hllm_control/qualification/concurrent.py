"""Concurrency-two serving evidence, separate from concurrency-one profiles.

A `ConcurrentServingEvidence` record binds one bounded two-request experiment: the
exact serving configuration, the single-request preconditions, every worker's own
request-lifecycle observations and the resulting token comparisons. Its acceptance
status is derived from those observations by `assess_evidence` and re-derived on every
load, so a record cannot claim overlap, exact output or cleanup that its own contents
do not show. `ProfileKey` and measured placement stay at concurrency one; nothing here
feeds a placement bundle.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import timedelta
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

SCHEMA_VERSION = "1.0"
PROFILE_FRESHNESS = timedelta(hours=24)
REQUIRED_CANCEL_PREFIX = 3
Status = Literal["accepted", "failed", "unknown"]
Policy = Literal[
    "cpu-rss-v1", "cuda-rss-device-v1", "mlx-rss-plus-allocator-v1", "mlx-footprint-max-v1"
]


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
    process_id: PositiveInt  # Observed before the first load; a change fails the record.


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


class ProfilePrecondition(ProfileModel):
    worker_id: Label
    artifact_digest: Digest
    measured_at: AwareDatetime
    requested_mlx_policy: MlxFitPolicy | None = Field(default=None, exclude_if=lambda v: v is None)
    prelaunch_status: Literal["safe", "unsafe", "unknown"]
    prelaunch_reasons: tuple[str, ...]
    host_budget: PhysicalBudget  # Live availability sampled before launch.
    device_budget: PhysicalBudget | None = None


class Preconditions(ProfileModel):
    baseline_report_sha256: Digest  # The passing single-request serving report.
    started_at: AwareDatetime
    profiles: tuple[ProfilePrecondition, ProfilePrecondition]
    guard: Label  # Guard settings in words; outcomes are in results.


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
    reason: str  # Retirement only: completed, cancelled, expired, allocation-failed.


class WorkerSnapshot(ProfileModel):
    phase: Label
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


class WorkerObservations(ProfileModel):
    worker_id: Label
    snapshots: Annotated[tuple[WorkerSnapshot, ...], Field(min_length=1)]


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
    start_monotonic_ns: NonNegativeInt  # Second admission on this worker's clock.
    end_monotonic_ns: NonNegativeInt  # First retirement of the pair, or last joint snapshot.
    admission_sequence: PositiveInt
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
    worker_id: Label
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
    worker_id: Label
    outcome: Literal["normal", "stopped", "unavailable"]
    record_sha256: Digest | None
    detail: str


class CleanupEvidence(ProfileModel):
    worker_id: Label
    verified: bool
    detail: str


class Results(ProfileModel):
    rounds: tuple[Round, ...]
    comparisons: tuple[TokenComparison, ...]
    overlap: tuple[OverlapInterval, ...]  # Derived; re-derived on load.
    logical_peaks: tuple[WorkerPeak, WorkerPeak]
    physical: tuple[PhysicalEnvelope, PhysicalEnvelope]
    guards: tuple[GuardOutcome, GuardOutcome]
    cleanup: tuple[CleanupEvidence, CleanupEvidence]
    failure_reason: str | None
    acceptance: Status
    acceptance_reasons: tuple[str, ...]


# ----------------------------------------------------------------------------- record


class ConcurrentServingEvidence(ProfileModel):
    schema_version: Literal["1.0"]
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
            ("physical envelopes", tuple(p.worker_id for p in self.results.physical)),
            ("guards", tuple(g.worker_id for g in self.results.guards)),
            ("cleanup", tuple(c.worker_id for c in self.results.cleanup)),
        ):
            if ids != order:
                raise ValueError(f"{name} must list the workers in stage order")
        content = self.model_dump(mode="json", exclude={"evidence_digest"})
        if self.evidence_digest != digest(content):
            raise ValueError("evidence digest mismatch")
        if self.results.overlap != derive_overlap(self.observations, self.results.rounds):
            raise ValueError("recorded overlap intervals are not what the observations show")
        peaks = derive_logical_peaks(self.observations, self.configuration)
        if self.results.logical_peaks != peaks:
            raise ValueError("recorded logical peaks are not what the observations show")
        status, reasons = assess_evidence(self)
        if (self.results.acceptance, self.results.acceptance_reasons) != (status, reasons):
            raise ValueError(f"recorded acceptance {self.results.acceptance!r} is not derived")
        return self


# ------------------------------------------------------------------------- derivations


def derive_overlap(
    observations: Observations, rounds: Iterable[Round]
) -> tuple[OverlapInterval, ...]:
    """Per-worker overlap from the worker's own lifecycle events, never from poll luck.

    An interval needs the admission event of the second member listing the first as an
    allocated concurrent reservation, plus proof that both held allocated KV at one
    instant: preferably the first retirement event of either member listing the other
    (the worker erases reservations under one lock, so the earlier retirement sees the
    later one live), otherwise a polled snapshot carrying both allocated rows. Missing
    evidence yields no interval, never an inferred one.
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
            admissions = [
                e
                for e in events.values()
                if e.kind == "admitted"
                and e.request_id in pair
                and any(o in e.concurrent_request_ids for o in pair if o != e.request_id)
            ]
            if not admissions:
                continue
            admission = min(admissions, key=lambda e: e.sequence)
            retirements = sorted(
                (
                    e
                    for e in events.values()
                    if e.kind == "retired"
                    and e.request_id in pair
                    and e.sequence > admission.sequence
                    and e.reason != "allocation-failed"
                ),
                key=lambda e: e.sequence,
            )
            witnessed = [
                e
                for e in retirements
                if any(o in e.concurrent_request_ids for o in pair if o != e.request_id)
            ]
            joint = [
                s
                for s in worker.snapshots
                if s.request_observations_available
                and all(
                    any(
                        r.request_id == identifier
                        and not r.allocating
                        and sum(getattr(r.cache, n) for n in MemoryAmounts.model_fields) > 0
                        for r in s.requests
                    )
                    for identifier in pair
                )
            ]
            if witnessed:
                proof: Literal["retirement-event", "joint-snapshot"] = "retirement-event"
                detail = f"event {witnessed[0].sequence}: {witnessed[0].request_id} retired"
                end = witnessed[0].at_monotonic_ns
            elif joint:
                proof = "joint-snapshot"
                detail = f"snapshot {joint[0].phase}"
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
                    start_monotonic_ns=admission.at_monotonic_ns,
                    end_monotonic_ns=max(end, admission.at_monotonic_ns),
                    admission_sequence=admission.sequence,
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
    host: PhysicalBudget,
    device: PhysicalBudget | None,
    mlx_policy: MlxFitPolicy,
    device_process_peak_bytes: int | None,
) -> PhysicalEnvelope:
    """Apply the selected policy once to the combined serving observations of one worker.

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


def assess_evidence(e: ConcurrentServingEvidence) -> tuple[Status, tuple[str, ...]]:
    """Derive acceptance: failed beats unknown; anything unobserved is unknown, never passed."""
    failed: list[str] = []
    unknown: list[str] = []
    r = e.results
    if r.failure_reason:
        failed.append("run failed: " + r.failure_reason)
    for p in e.preconditions.profiles:
        age = e.preconditions.started_at - p.measured_at
        if not timedelta(0) <= age <= PROFILE_FRESHNESS:
            failed.append(f"{p.worker_id}: single-request profile stale or future-dated")
        if p.prelaunch_status != "safe":
            failed.append(f"{p.worker_id}: prelaunch single-request fit was {p.prelaunch_status}")
    for c in e.configuration.workers:
        if not c.request_observations:
            unknown.append(f"{c.worker_id}: worker did not advertise request observations")
    for w in e.observations.workers:
        processes = {s.process_id for s in w.snapshots if s.process_id is not None}
        if len(processes) > 1:
            failed.append(f"{w.worker_id}: worker process changed during the experiment")
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
        for c in comparisons:
            assert c is not None
            if c.error:
                failed.append(f"{c.request_id}: stream error: {c.error}")
            if not c.exact:
                failed.append(f"{c.request_id}: continuation differs from the reference")
            if c.role == "cancel-victim" and len(c.expected_ids) < REQUIRED_CANCEL_PREFIX:
                failed.append(f"{c.request_id}: cancelled before {REQUIRED_CANCEL_PREFIX} tokens")
        if round_.kind != "serial":
            pair = (round_.request_ids[0], round_.request_ids[1])
            for w in e.identity.worker_order:
                if (w, pair) not in overlapping:
                    unknown.append(f"{w}: overlap of {pair[0]} and {pair[1]} not observed")
        if round_.kind == "cancel":
            victim, survivor = round_.request_ids[0], round_.request_ids[1]
            for w in e.observations.workers:
                independent = any(
                    ev.kind == "retired"
                    and ev.request_id == victim
                    and survivor in ev.concurrent_request_ids
                    for s in w.snapshots
                    for ev in s.events
                )
                if not independent:
                    unknown.append(
                        f"{w.worker_id}: independent retirement of {victim} while {survivor} held "
                        "its reservation was not observed"
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
            failed.append(f"{p.worker_id}: " + "; ".join(p.reasons))
        elif p.status == "unknown":
            unknown.append(f"{p.worker_id}: " + "; ".join(p.reasons))
    for g in r.guards:
        if g.outcome == "stopped":
            failed.append(f"{g.worker_id}: resource guard stopped the run: {g.detail}")
        elif g.outcome == "unavailable":
            unknown.append(f"{g.worker_id}: guard outcome unavailable: {g.detail}")
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
    physical: tuple[PhysicalEnvelope, PhysicalEnvelope],
    guards: tuple[GuardOutcome, GuardOutcome],
    cleanup: tuple[CleanupEvidence, CleanupEvidence],
    failure_reason: str | None,
) -> ConcurrentServingEvidence:
    """Assemble a record whose derived fields and acceptance come from its own contents."""
    rounds = tuple(rounds)
    draft = Results(
        rounds=rounds,
        comparisons=tuple(comparisons),
        overlap=derive_overlap(observations, rounds),
        logical_peaks=derive_logical_peaks(observations, configuration),
        physical=physical,
        guards=guards,
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
