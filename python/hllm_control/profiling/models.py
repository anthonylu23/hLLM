"""Immutable, content-addressed measurements. Separate from configured v1 profiles."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from hllm_control.models import (
    Backend,
    ConnectionType,
    DType,
    NonNegativeInt,
    PositiveInt,
    StageAssignment,
    StrictModel,
    WorkloadProfile,
)
from hllm_control.serialization import canonical_json_bytes

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Label = Annotated[str, Field(min_length=1)]
Milliseconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


def digest(value: StrictModel | dict[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class ProfileModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Environment(ProfileModel):
    backend: Backend
    device_identity: Label
    device_name: Label
    backend_version: Label
    driver_version: Label  # "not-applicable" on CPU; never silently absent
    allocator: Label
    allocator_config: Label
    source_revision: Label
    binary_digest: Digest
    compiler: Label
    os: Label


class ProfileKey(ProfileModel):
    manifest_digest: Digest
    checkpoint_digest: Digest  # Content hash of config and every checkpoint shard.
    environment: Environment
    assignment: StageAssignment
    workload: WorkloadProfile
    workload_digest: Digest
    execution_dtype: DType
    transport_mode: Literal["pageable", "pinned"]
    input_kind: Literal["synthetic-shape", "reference"]
    input_digest: Digest

    @model_validator(mode="after")
    def supported_workload(self) -> Self:
        w = self.workload
        if self.workload_digest != digest(w):
            raise ValueError("workload digest mismatch")
        if w.concurrency != 1:
            raise ValueError("measured profiles currently support concurrency 1")
        if w.total_cached_tokens < w.prompt_tokens + w.output_tokens:
            raise ValueError("cache capacity must cover prompt plus output tokens")
        supported = (
            (DType.F32,) if self.environment.backend == Backend.CPU else (DType.F16, DType.F32)
        )
        if self.execution_dtype not in supported or w.kv_dtype != self.execution_dtype:
            raise ValueError("unsupported execution/KV dtype for backend")
        if w.activation_dtype != DType.F16:
            raise ValueError("native boundary requires F16")
        if self.transport_mode == "pinned" and self.environment.backend != Backend.CUDA:
            raise ValueError("pinned transport is currently CUDA-only")
        a = self.assignment
        if a.layer_start >= a.layer_end:
            raise ValueError("stage must own a nonempty layer range")
        if a.owns_token_embedding != (a.stage_index == 0 and a.layer_start == 0):
            raise ValueError("embedding ownership does not match first stage")
        if not (a.owns_final_norm == a.owns_lm_head == a.owns_sampling):
            raise ValueError("final tensor and sampling ownership must agree")
        return self


class Conditions(ProfileModel):
    measured_at: AwareDatetime
    warmup_cycles: NonNegativeInt
    measured_cycles: PositiveInt
    process_policy: Literal[
        "fresh-process-then-reloads", "loaded-stage-paired-passes", "native-stream-exchanges"
    ]
    concurrent_load: Label
    notes: tuple[str, ...] = ()


class MemoryAmounts(ProfileModel):
    host: NonNegativeInt = 0
    device: NonNegativeInt = 0
    pinned: NonNegativeInt = 0
    unified: NonNegativeInt = 0

    @model_validator(mode="after")
    def pinned_is_host(self) -> Self:
        if self.pinned > self.host:
            raise ValueError("pinned memory is a subset of host memory")
        return self


class AllocatorSample(ProfileModel):
    active_bytes: NonNegativeInt
    cached_bytes: NonNegativeInt
    peak_bytes: NonNegativeInt
    peak_scope: Literal["phase", "unavailable"]


MemoryPhase = Literal[
    "baseline", "load", "allocate", "prefill", "decode", "request_cleanup", "unload"
]
MEMORY_PHASES = ("baseline", "load", "allocate", "prefill", "decode", "request_cleanup", "unload")


class MemorySample(ProfileModel):
    cycle: NonNegativeInt
    phase: MemoryPhase
    weights: MemoryAmounts
    cache: MemoryAmounts
    workspace: MemoryAmounts
    allocator: AllocatorSample | None
    rss_bytes: NonNegativeInt | None
    rss_lifetime_peak_bytes: NonNegativeInt | None
    device_available_bytes: NonNegativeInt | None
    completed_steps: NonNegativeInt


class PhysicalSample(ProfileModel):
    elapsed_seconds: Annotated[float, Field(ge=0)]
    rss_bytes: NonNegativeInt | None
    device_process_bytes: NonNegativeInt | None


class MemoryMeasurement(ProfileModel):
    kind: Literal["memory"] = "memory"
    admission_capacity: MemoryAmounts
    samples: tuple[MemorySample, ...]
    physical_samples: tuple[PhysicalSample, ...]
    physical_sample_interval_seconds: Annotated[float, Field(gt=0)]
    completed: bool
    error: str | None = None
    # No correctness assertion: independent shape execution does not generate a continuation.
    execution: Literal["isolated-stage-shape-exercise"] = "isolated-stage-shape-exercise"


class ComputeMeasurement(ProfileModel):
    kind: Literal["compute"] = "compute"
    phase: Literal["prefill", "decode"]
    context_tokens: NonNegativeInt
    component: Literal["stage", "layer", "embedding", "final_norm", "lm_head", "sampling"]
    layer_index: NonNegativeInt | None = None
    samples_ms: Annotated[tuple[Milliseconds, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def layer_identity(self) -> Self:
        if (self.component == "layer") != (self.layer_index is not None):
            raise ValueError("layer index is required only for a layer measurement")
        return self


class ConversionMeasurement(ProfileModel):
    kind: Literal["conversion"] = "conversion"
    direction: Literal["to-wire", "from-wire"]
    payload_bytes: PositiveInt
    samples_ms: Annotated[tuple[Milliseconds, ...], Field(min_length=1)]


class LinkMeasurement(ProfileModel):
    kind: Literal["link"] = "link"
    source_worker_id: Label
    target_worker_id: Label
    source_environment: Environment
    target_environment: Environment
    connection_type: ConnectionType
    payload_bytes: PositiveInt
    stream_policy: Literal["persistent", "new-stream"]
    attribution: Literal["round-trip-including-feedback", "stream-setup"]
    samples_ms: Annotated[tuple[Milliseconds, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def distinct_endpoints(self) -> Self:
        if self.source_worker_id == self.target_worker_id:
            raise ValueError("directional link requires distinct endpoints")
        if self.connection_type == ConnectionType.UNKNOWN:
            raise ValueError("measured link requires a qualified connection type")
        return self


class TimingRecord(ProfileModel):
    cycle: NonNegativeInt
    step: NonNegativeInt
    context_tokens: NonNegativeInt
    phase: Literal["prefill", "decode"]
    component: Literal[
        "stage",
        "layer",
        "embedding",
        "final_norm",
        "lm_head",
        "sampling",
        "from-wire",
        "to-wire",
        "validation",
        "runtime_overhead",
        "sequence_allocation",
    ]
    layer_index: NonNegativeInt | None
    timing_mode: Literal["component-synchronized", "whole-stage"]
    milliseconds: Milliseconds


class ComputeRunMeasurement(ProfileModel):
    kind: Literal["compute-run"] = "compute-run"
    admission_capacity: MemoryAmounts
    records: tuple[TimingRecord, ...]
    identical_output_cycles: tuple[NonNegativeInt, ...]
    completed: bool
    error: str | None = None


Measurement = Annotated[
    MemoryMeasurement
    | ComputeMeasurement
    | ConversionMeasurement
    | LinkMeasurement
    | ComputeRunMeasurement,
    Field(discriminator="kind"),
]


class ProfileArtifact(ProfileModel):
    schema_version: Literal["1.0", "1.1"] = "1.0"
    profiler_version: Literal["0.1.0", "0.2.0"] = "0.1.0"
    key: ProfileKey
    conditions: Conditions
    measurement: Measurement
    artifact_digest: Digest

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.artifact_digest != digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("profile artifact digest mismatch")
        m = self.measurement
        if isinstance(m, MemoryMeasurement):
            if m.completed:
                cycles = self.conditions.warmup_cycles + self.conditions.measured_cycles
                expected = [(cycle, phase) for cycle in range(cycles) for phase in MEMORY_PHASES]
                if [(s.cycle, s.phase) for s in m.samples] != expected or m.error is not None:
                    raise ValueError("completed memory profile must contain all ordered phases")
                for sample in m.samples:
                    if sample.phase in ("request_cleanup", "unload"):
                        if sample.cache != MemoryAmounts() or sample.workspace != MemoryAmounts():
                            raise ValueError("sequence reservations did not retire")
                    if sample.phase in ("baseline", "unload") and sample.weights != MemoryAmounts():
                        raise ValueError("model reservations did not retire")
                    if sample.phase in ("decode", "request_cleanup", "unload"):
                        if sample.completed_steps != self.key.workload.output_tokens:
                            raise ValueError("incomplete context exercise")
            elif not m.error:
                raise ValueError("incomplete measurement requires an error")
        elif isinstance(m, ComputeRunMeasurement):
            if self.schema_version != "1.1":
                raise ValueError("compute runs require measured profile schema 1.1")
            if m.completed:
                validate_compute_run(self.key, self.conditions, m)
            elif not m.error:
                raise ValueError("incomplete compute run requires an error")
        elif len(m.samples_ms) != self.conditions.measured_cycles:
            raise ValueError("sample count must match measured cycles; exclude warmups")
        return self


def make_artifact(
    key: ProfileKey, conditions: Conditions, measurement: Measurement
) -> ProfileArtifact:
    unsigned: dict[str, object] = {
        "schema_version": "1.1" if isinstance(measurement, ComputeRunMeasurement) else "1.0",
        "profiler_version": "0.2.0" if isinstance(measurement, ComputeRunMeasurement) else "0.1.0",
        "key": key.model_dump(mode="json"),
        "conditions": conditions.model_dump(mode="json"),
        "measurement": measurement.model_dump(mode="json"),
    }
    return ProfileArtifact.model_validate({**unsigned, "artifact_digest": digest(unsigned)})


class Compatibility(ProfileModel):
    status: Literal["compatible", "missing", "incompatible", "out-of-range", "incomplete"]
    reasons: tuple[str, ...] = ()


def check_compatibility(
    artifact: ProfileArtifact | None,
    key: ProfileKey,
    *,
    kind: Literal["memory", "compute", "conversion", "link", "compute-run"],
    scope: dict[str, object] | None = None,
    now: datetime | None = None,
    maximum_age_seconds: float | None = None,
) -> Compatibility:
    if artifact is None:
        return Compatibility(status="missing")
    differences = tuple(
        name
        for name in ProfileKey.model_fields
        if getattr(artifact.key, name) != getattr(key, name)
    )
    if differences:
        shape_only = set(differences) <= {"workload", "workload_digest"}
        return Compatibility(
            status="out-of-range" if shape_only else "incompatible", reasons=differences
        )
    if artifact.measurement.kind != kind:
        return Compatibility(status="incompatible", reasons=("measurement kind",))
    if kind not in ("memory", "compute-run"):
        actual_scope = artifact.measurement.model_dump(mode="json", exclude={"samples_ms"})
        if scope != actual_scope:
            return Compatibility(status="incompatible", reasons=("measurement scope",))
    if maximum_age_seconds is not None:
        if now is None or now.tzinfo is None or maximum_age_seconds < 0:
            raise ValueError("age checking requires an aware time and nonnegative maximum age")
        age = (now - artifact.conditions.measured_at).total_seconds()
        if not 0 <= age <= maximum_age_seconds:
            return Compatibility(status="incompatible", reasons=("measurement age",))
    if (
        isinstance(artifact.measurement, (MemoryMeasurement, ComputeRunMeasurement))
        and not artifact.measurement.completed
    ):
        return Compatibility(
            status="incomplete", reasons=(artifact.measurement.error or "incomplete",)
        )
    return Compatibility(status="compatible")


def validate_compute_run(key: ProfileKey, conditions: Conditions, m: ComputeRunMeasurement) -> None:
    cycles = conditions.warmup_cycles + conditions.measured_cycles
    if m.error is not None or m.identical_output_cycles != tuple(range(cycles)):
        raise ValueError("missing paired output parity checks")
    indexed: dict[tuple[int, str, int], list[TimingRecord]] = {}
    for r in m.records:
        if r.cycle >= cycles or r.step >= key.workload.output_tokens:
            raise ValueError("timing sample outside workload")
        position = 0 if r.step == 0 else key.workload.prompt_tokens + r.step - 1
        if r.context_tokens != position or r.phase != ("prefill" if r.step == 0 else "decode"):
            raise ValueError("timing context/phase mismatch")
        if (r.component == "layer") != (r.layer_index is not None):
            raise ValueError("layer timing identity mismatch")
        indexed.setdefault((r.cycle, r.timing_mode, r.step), []).append(r)
    for cycle in range(cycles):
        for mode in ("whole-stage", "component-synchronized"):
            for step in range(key.workload.output_tokens):
                rows = indexed.get((cycle, mode, step), [])
                actual = [(r.component, r.layer_index) for r in rows]
                expected: list[tuple[str, int | None]] = [("stage", None)]
                if step == 0:
                    expected.append(("sequence_allocation", None))
                if mode == "component-synchronized":
                    expected += [
                        ("embedding" if key.assignment.owns_token_embedding else "from-wire", None)
                    ]
                    expected += [
                        ("layer", i)
                        for i in range(key.assignment.layer_start, key.assignment.layer_end)
                    ]
                    if key.environment.backend != Backend.CPU:
                        expected.append(("validation", None))
                    expected += (
                        [(name, None) for name in ("final_norm", "lm_head", "sampling")]
                        if key.assignment.owns_sampling
                        else [("to-wire", None)]
                    )
                    expected.append(("runtime_overhead", None))
                if set(actual) != set(expected) or len(actual) != len(expected):
                    raise ValueError("missing/duplicate or unexpected timing components")
                if mode == "component-synchronized":
                    total = next(r.milliseconds for r in rows if r.component == "stage")
                    components = sum(
                        r.milliseconds
                        for r in rows
                        if r.component not in ("stage", "sequence_allocation")
                    )
                    if abs(total - components) > 0.001:
                        raise ValueError("timing components do not reconcile with stage total")
