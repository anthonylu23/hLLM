"""Fail closed on fresh capability, binary and physical checks before any stage load."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

from hllm_control.models import DeploymentPlan, ModelManifest, PlanningMode
from hllm_control.planner.measured import DiskProfileBundle, ProfileBundle, evaluate
from hllm_control.profiling.memory import assess_fit
from hllm_control.profiling.models import MemoryAmounts, digest
from hllm_control.proto import common_pb2, control_pb2_grpc, profile_pb2

# Generated service factories are untyped.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


def validate_activation(
    manifest: ModelManifest,
    plan: DeploymentPlan,
    bundle: ProfileBundle | DiskProfileBundle,
    controls: Sequence[control_pb2_grpc.WorkerControlStub],
) -> None:
    if (
        plan.planning_mode != PlanningMode.MEASURED
        or plan.profile_bundle_digest != bundle.bundle_digest
    ):
        raise ValueError("activation requires the exact measured profile bundle")
    if plan.manifest_digest != bundle.manifest_digest or plan.workload_digest != digest(
        bundle.workload
    ):
        raise ValueError("measured workload/manifest mismatch")
    unsigned = plan.model_dump(mode="json", exclude={"plan_id", "plan_digest"})
    if plan.plan_digest != digest(unsigned) or plan.plan_id != f"plan-{plan.plan_digest[:16]}":
        raise ValueError("measured plan hash mismatch")
    if plan.stages[-1].layer_end != manifest.config.num_layers:
        raise ValueError("measured plan does not cover model")
    if len(plan.stages) != 2:
        raise ValueError("measured activation requires two stages")
    status, reasons, _ = evaluate(
        bundle,
        manifest,
        bundle.workload,
        (plan.stages[0], plan.stages[1]),
        weight_dtype=plan.weight_dtype,
    )
    if status != "measured":
        raise ValueError("selected assignment is not qualified: " + "; ".join(reasons))
    now = datetime.now(UTC)
    for profile in bundle.profiles:
        if (
            profile.key.assignment in plan.stages
            and not 0
            <= (now - profile.conditions.measured_at).total_seconds()
            <= bundle.maximum_profile_age_seconds
        ):
            raise ValueError("selected profile expired; refresh measurements and replan")
    if not 0 <= (now - bundle.evaluated_at).total_seconds() <= bundle.maximum_profile_age_seconds:
        raise ValueError("stale/future profile bundle; refresh measurements and replan")
    for assignment, control in zip(plan.stages, controls, strict=True):
        b = next(w for w in bundle.workers if w.worker.worker_id == assignment.worker_id)
        c = control.GetCapabilities(common_pb2.Empty(), timeout=5).worker
        q = control.GetQualificationState(common_pb2.Empty(), timeout=5)
        if (
            (plan.weight_dtype is not None and not c.supports_mixed_precision)
            or c.worker_id != assignment.worker_id
            or c.endpoint != b.worker.endpoint
            or c.backend != profile_pb2.Backend.Value(f"BACKEND_{b.worker.backend.value.upper()}")
            or manifest.architecture.architecture_id not in c.supported_architectures
            or common_pb2.DataType.Value(f"DATA_TYPE_{plan.execution_dtype.value}")
            not in c.supported_execution_dtypes
        ):
            raise ValueError("worker capabilities changed")
        if q.binary_digest != b.runtime_binary_digest or (
            q.device_identity,
            q.backend_version,
            q.driver_version,
            q.allocator,
        ) != (
            b.runtime_device_identity,
            b.compute_environment.backend_version,
            b.runtime_driver_version,
            b.compute_environment.allocator,
        ):
            raise ValueError("worker binary/device/backend/driver/allocator changed")
        if q.device_fingerprint != b.compute_environment.device_identity:
            raise ValueError("worker host/hardware fingerprint changed")
        if q.boundary_transport_mode != b.transport_mode:
            raise ValueError("worker boundary transport mode changed")
        if b.worker.backend.value == "cuda":
            config = json.loads(b.compute_environment.allocator_config)
            for name in ("pytorch_alloc_conf", "pytorch_cuda_alloc_conf"):
                actual = getattr(q, name) if q.HasField(name) else None
                if config.get(name.upper()) != actual:
                    raise ValueError("worker allocator configuration changed")
        capacity = MemoryAmounts(
            **{
                name: next((v.capacity_bytes for v in c.memory_budgets if v.domain == domain), 0)
                for name, domain in (
                    ("host", profile_pb2.MEMORY_DOMAIN_HOST),
                    ("unified", profile_pb2.MEMORY_DOMAIN_UNIFIED),
                    ("device", profile_pb2.MEMORY_DOMAIN_DEVICE),
                    ("pinned", profile_pb2.MEMORY_DOMAIN_HOST_PINNED),
                )
            }
        )
        if capacity != b.admission_capacity:
            raise ValueError("worker admission capacity changed; reprofile/replan")
        profiles = [
            p
            for p in bundle.profiles
            if p.measurement.kind == "memory"
            and p.key.assignment == assignment
            and p.key.environment == b.memory_environment
            and p.key.manifest_digest == manifest.manifest_digest
            and p.key.checkpoint_digest == bundle.checkpoint_digest
            and p.key.workload == bundle.workload
            and p.key.transport_mode == b.transport_mode
        ]
        if len(profiles) != 1:
            raise ValueError("missing or ambiguous memory evidence")
        host = b.host.model_copy(
            update={
                "available_bytes": q.available_host_bytes
                if q.HasField("available_host_bytes")
                else None
            }
        )
        device = (
            b.device.model_copy(
                update={
                    "available_bytes": q.available_device_bytes
                    if q.HasField("available_device_bytes")
                    else None
                }
            )
            if b.device
            else None
        )
        fit = assess_fit(profiles[0], capacity, host, device)
        if fit.status != "safe":
            raise ValueError("fresh physical memory gate failed: " + "; ".join(fit.reasons))
        memory = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
        if (
            memory.loaded_weight_bytes
            or memory.active_requests
            or memory.reserved_cache_bytes
            or memory.reserved_workspace_bytes
        ):
            raise ValueError("measured activation requires idle unloaded workers")
