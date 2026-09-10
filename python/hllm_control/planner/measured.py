"""Exact-scope measured placement. Missing evidence never becomes a zero cost."""

from __future__ import annotations

from datetime import datetime
from statistics import median
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from hllm_control.models import (
    ModelManifest,
    PerformanceEstimate,
    StageAssignment,
    WorkerProfile,
    WorkloadProfile,
)
from hllm_control.profiling.link import (
    LinkArtifact,
    PathObservation,
    ProbeIdentity,
    check_link_compatibility,
)
from hllm_control.profiling.memory import PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    ComputeRunMeasurement,
    Digest,
    Environment,
    MemoryAmounts,
    Milliseconds,
    ProfileArtifact,
    ProfileModel,
    digest,
)


class WorkerEvidence(ProfileModel):
    worker: WorkerProfile
    memory_environment: Environment
    compute_environment: Environment
    runtime_device_identity: str
    runtime_driver_version: str
    runtime_binary_digest: Digest
    admission_capacity: MemoryAmounts
    host: PhysicalBudget
    device: PhysicalBudget | None = None
    observed_at: AwareDatetime
    transport_mode: Literal["pageable", "pinned"]

    @model_validator(mode="after")
    def environment_match(self) -> Self:
        for field in (
            "backend",
            "device_identity",
            "device_name",
            "backend_version",
            "driver_version",
            "allocator",
            "allocator_config",
            "os",
        ):
            if getattr(self.memory_environment, field) != getattr(self.compute_environment, field):
                raise ValueError("memory/compute device environment mismatch: " + field)
        if self.worker.backend != self.compute_environment.backend:
            raise ValueError("worker/backend environment mismatch")
        return self


class RequestSetupSample(ProfileModel):
    client_ttft_ms: Milliseconds
    native_ttft_ms: Milliseconds
    native_setup_ms: Milliseconds

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.native_setup_ms > self.native_ttft_ms or self.native_ttft_ms > self.client_ttft_ms:
            raise ValueError("inconsistent production request timing")
        return self

    @property
    def residual_ms(self) -> float:
        return self.native_setup_ms + self.client_ttft_ms - self.native_ttft_ms


class DirectionEvidence(ProfileModel):
    source_worker_id: str
    target_worker_id: str
    source_probe: ProbeIdentity
    target_probe: ProbeIdentity
    path: PathObservation
    link_digest: Digest
    # Residual production request setup (client ingress, control reservation and stream
    # startup), measured separately with model execution/sequence allocation excluded.
    # The link probe's diagnostic metadata barrier is NOT this measurement.
    request_setup_samples_ms: Annotated[tuple[Milliseconds, ...], Field(min_length=5)]
    request_setup_raw: Annotated[tuple[RequestSetupSample, ...], Field(min_length=5)]
    request_setup_evidence_digest: Digest
    request_setup_method: Annotated[str, Field(min_length=1)]
    request_setup_measured_at: AwareDatetime

    @model_validator(mode="after")
    def raw_setup(self) -> Self:
        expected = tuple(s.residual_ms for s in self.request_setup_raw)
        if len(expected) != len(self.request_setup_samples_ms) or any(
            abs(a - b) > 1e-6 for a, b in zip(expected, self.request_setup_samples_ms, strict=True)
        ):
            raise ValueError("request setup samples disagree with raw production timings")
        if self.request_setup_evidence_digest != digest(
            {"samples": [s.model_dump(mode="json") for s in self.request_setup_raw]}
        ):
            raise ValueError("request setup evidence digest mismatch")
        return self


class BundleContent(ProfileModel):
    schema_version: Literal["1.0"] = "1.0"
    manifest_digest: Digest
    checkpoint_digest: Digest
    workload: WorkloadProfile
    workers: Annotated[tuple[WorkerEvidence, ...], Field(min_length=2, max_length=2)]
    profiles: tuple[ProfileArtifact, ...]
    links: tuple[LinkArtifact, ...]
    directions: tuple[DirectionEvidence, ...]
    evaluated_at: AwareDatetime
    maximum_profile_age_seconds: Annotated[float, Field(gt=0)] = 86400
    maximum_snapshot_age_seconds: Annotated[float, Field(gt=0, le=300)] = 60
    concurrent_load: str
    input_kind: Literal["synthetic-shape", "reference"]
    input_digest: Digest

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len({w.worker.worker_id for w in self.workers}) != 2:
            raise ValueError("bundle needs two distinct workers")
        for items in (self.profiles, self.links):
            if len({a.artifact_digest for a in items}) != len(items):
                raise ValueError("duplicate artifact")
        if len({(d.source_worker_id, d.target_worker_id) for d in self.directions}) != len(
            self.directions
        ):
            raise ValueError("duplicate direction")
        return self


class ProfileBundle(BundleContent):
    bundle_digest: Digest

    @model_validator(mode="after")
    def seal(self) -> Self:
        if self.bundle_digest != digest(self.model_dump(mode="json", exclude={"bundle_digest"})):
            raise ValueError("profile bundle digest mismatch")
        return self


def seal_bundle(content: BundleContent) -> ProfileBundle:
    data = content.model_dump(mode="json")
    return ProfileBundle.model_validate({**data, "bundle_digest": digest(data)})


def _fresh(at: datetime, now: datetime, limit: float) -> bool:
    return 0 <= (now - at).total_seconds() <= limit


def evaluate(
    bundle: ProfileBundle,
    manifest: ModelManifest,
    workload: WorkloadProfile,
    assignments: tuple[StageAssignment, StageAssignment],
) -> tuple[str, tuple[str, ...], PerformanceEstimate | None]:
    """Return measured, unknown, or infeasible; no interpolation or layer summation."""
    if bundle.manifest_digest != manifest.manifest_digest or bundle.workload != workload:
        return "unknown", ("BUNDLE_MANIFEST_OR_WORKLOAD_MISMATCH",), None
    unknown: list[str] = []
    unsafe: list[str] = []
    computes: list[ProfileArtifact] = []
    identities: list[str] = []
    host_envelopes: dict[str, int] = {}
    device_envelopes: dict[str, int] = {}
    for assignment in assignments:
        binding = next(w for w in bundle.workers if w.worker.worker_id == assignment.worker_id)
        if not _fresh(
            binding.observed_at, bundle.evaluated_at, bundle.maximum_snapshot_age_seconds
        ):
            unknown.append(f"STALE_PHYSICAL_SNAPSHOT:{assignment.worker_id}")
        for kind, environment in (
            ("memory", binding.memory_environment),
            ("compute-run", binding.compute_environment),
        ):
            matches = [
                p
                for p in bundle.profiles
                if (
                    p.measurement.kind == kind
                    and p.key.assignment == assignment
                    and p.key.environment == environment
                    and p.key.input_kind == bundle.input_kind
                    and p.key.input_digest == bundle.input_digest
                    and p.key.manifest_digest == manifest.manifest_digest
                    and p.key.checkpoint_digest == bundle.checkpoint_digest
                    and p.key.workload == workload
                    and p.key.execution_dtype == workload.kv_dtype
                    and p.key.transport_mode == binding.transport_mode
                    and p.conditions.concurrent_load == bundle.concurrent_load
                    and _fresh(
                        p.conditions.measured_at,
                        bundle.evaluated_at,
                        bundle.maximum_profile_age_seconds,
                    )
                )
            ]
            if len(matches) != 1:
                unknown.append(f"MISSING_OR_AMBIGUOUS_{kind.upper()}:{assignment.worker_id}")
                continue
            artifact = matches[0]
            identities.append(artifact.artifact_digest)
            if kind == "memory":
                fit = assess_fit(artifact, binding.admission_capacity, binding.host, binding.device)
                if fit.host_envelope_bytes is not None:
                    host_envelopes[assignment.worker_id] = fit.host_envelope_bytes
                if fit.device_envelope_bytes is not None:
                    device_envelopes[assignment.worker_id] = fit.device_envelope_bytes
                if fit.status != "safe":
                    (unsafe if fit.status == "unsafe" else unknown).extend(
                        f"{assignment.worker_id}:{reason}" for reason in fit.reasons
                    )
            else:
                m = artifact.measurement
                if not isinstance(m, ComputeRunMeasurement) or not m.completed:
                    unknown.append(f"INCOMPLETE_COMPUTE:{assignment.worker_id}")
                elif m.admission_capacity != binding.admission_capacity:
                    unknown.append(f"COMPUTE_ADMISSION_MISMATCH:{assignment.worker_id}")
                else:
                    computes.append(artifact)
    direction = next(
        (
            d
            for d in bundle.directions
            if (d.source_worker_id, d.target_worker_id) == tuple(a.worker_id for a in assignments)
        ),
        None,
    )
    link = next(
        (a for a in bundle.links if direction and a.artifact_digest == direction.link_digest), None
    )
    if direction is None or link is None:
        unknown.append("MISSING_DIRECTIONAL_LINK_OR_REQUEST_SETUP")
    else:
        unknown.extend(
            check_link_compatibility(
                link,
                source=direction.source_probe,
                target=direction.target_probe,
                prompt_tokens=workload.prompt_tokens,
                output_tokens=workload.output_tokens,
                hidden_size=manifest.config.hidden_size,
                path=direction.path,
                now=bundle.evaluated_at,
                maximum_age_seconds=bundle.maximum_profile_age_seconds,
            )
        )
        if not _fresh(
            direction.path.observed_at, bundle.evaluated_at, bundle.maximum_snapshot_age_seconds
        ):
            unknown.append("STALE_PATH_SNAPSHOT")
        if link.concurrent_load != bundle.concurrent_load:
            unknown.append("LINK_CONCURRENT_LOAD_MISMATCH")
        if not _fresh(
            direction.request_setup_measured_at,
            bundle.evaluated_at,
            bundle.maximum_profile_age_seconds,
        ):
            unknown.append("STALE_REQUEST_SETUP")
        identities.extend((link.artifact_digest, direction.request_setup_evidence_digest))
    if unsafe or unknown:
        return "infeasible" if unsafe else "unknown", tuple(unsafe + unknown), None
    assert direction is not None and link is not None and link.result is not None
    components: dict[str, float] = {}
    stage_steps = [0.0] * workload.output_tokens
    allocation = 0.0
    for artifact in computes:
        m = artifact.measurement
        assert isinstance(m, ComputeRunMeasurement)
        rows = [
            r
            for r in m.records
            if r.cycle >= artifact.conditions.warmup_cycles and r.timing_mode == "whole-stage"
        ]
        alloc = median(r.milliseconds for r in rows if r.component == "sequence_allocation")
        allocation += alloc
        per_step = [
            median(r.milliseconds for r in rows if r.component == "stage" and r.step == step)
            for step in range(workload.output_tokens)
        ]
        stage_steps = [a + b for a, b in zip(stage_steps, per_step, strict=True)]
        prefix = artifact.key.assignment.worker_id
        components[f"{prefix}.allocation"] = alloc
        components[f"{prefix}.prefill"] = per_step[0]
        components[f"{prefix}.decode"] = sum(per_step[1:])
    r = link.result
    boundary = [
        median(
            s.sender_encode_ms + s.round_trip_ms
            for s in r.samples
            if s.cycle >= r.warmup_cycles and s.step == step
        )
        for step in range(workload.output_tokens)
    ]
    setup = median(direction.request_setup_samples_ms)
    components.update(
        request_setup=setup, boundary_prefill=boundary[0], boundary_decode=sum(boundary[1:])
    )
    ttft = setup + allocation + stage_steps[0] + boundary[0]
    decode = tuple(a + b for a, b in zip(stage_steps[1:], boundary[1:], strict=True))
    return (
        "measured",
        (),
        PerformanceEstimate(
            boundary_prefill_ms=boundary[0],
            boundary_decode_ms=median(boundary[1:]) if boundary[1:] else 0,
            confidence="exact-assignment-whole-stage; additive prediction, independently qualify",
            ttft_ms=ttft,
            decode_ms=decode,
            generation_ms=ttft + sum(decode),
            components_ms=components,
            profile_digests=tuple(identities),
            host_envelope_bytes=host_envelopes,
            device_envelope_bytes=device_envelopes,
        ),
    )
