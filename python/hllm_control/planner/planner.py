"""Deterministic enumeration and ranking of two-worker contiguous placements."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from hllm_control.models import (
    DTYPE_BYTES,
    PLANNER_VERSION,
    ConnectionType,
    DeploymentPlan,
    DType,
    LinkProfile,
    MemoryDomain,
    ModelManifest,
    PerformanceEstimate,
    PlanCandidate,
    PlannerSettings,
    PlanningMode,
    PlanningReport,
    StageAssignment,
    StageMemory,
    TensorRecord,
    TensorRole,
    WorkerProfile,
    WorkloadProfile,
    maximum_boundary_tokens,
)
from hllm_control.planner.measured import DiskProfileBundle, ProfileBundle, evaluate
from hllm_control.profiling.models import digest
from hllm_control.serialization import canonical_json_bytes


class PlanningError(ValueError):
    """Raised when planner inputs are internally inconsistent."""


def _resident_bytes(tensor: TensorRecord, execution_dtype: DType) -> int:
    return tensor.num_elements * DTYPE_BYTES[execution_dtype]


def _stage_weight_bytes(
    manifest: ModelManifest,
    *,
    start: int,
    end: int,
    is_first: bool,
    is_final: bool,
    execution_dtype: DType,
) -> int:
    total = 0
    tied = manifest.config.tied_embeddings
    for tensor in manifest.tensors:
        include = False
        if tensor.role == TensorRole.TRANSFORMER_LAYER:
            include = tensor.layer_index is not None and start <= tensor.layer_index < end
        elif tensor.role == TensorRole.TOKEN_EMBEDDING:
            include = is_first or (is_final and tied)
        elif tensor.role == TensorRole.FINAL_NORM:
            include = is_final
        elif tensor.role == TensorRole.LM_HEAD:
            # Workers verify redundant tied heads but retain no second resident copy.
            include = is_final and not tied
        elif tensor.role == TensorRole.ARCHITECTURE_STATE:
            include = True
        if include:
            total += _resident_bytes(tensor, execution_dtype)
    return total


def _stage_memory(
    manifest: ModelManifest,
    worker: WorkerProfile,
    workload: WorkloadProfile,
    settings: PlannerSettings,
    *,
    start: int,
    end: int,
    is_first: bool,
    is_final: bool,
) -> StageMemory:
    budget = worker.budget_for(worker.primary_memory_domain)
    if budget is None:  # guarded by WorkerProfile, retained for type narrowing and defensive use
        raise PlanningError(f"worker {worker.worker_id} has no primary memory budget")
    weight_bytes = _stage_weight_bytes(
        manifest,
        start=start,
        end=end,
        is_first=is_first,
        is_final=is_final,
        execution_dtype=settings.weight_dtype or settings.execution_dtype,
    )
    layer_count = end - start
    kv_per_layer_token = (
        2 * manifest.config.num_kv_heads * manifest.config.head_dim * DTYPE_BYTES[workload.kv_dtype]
    )
    kv_cache_bytes = math.ceil(
        layer_count
        * workload.total_cached_tokens
        * kv_per_layer_token
        * workload.kv_overhead_factor
    )
    activation_payload = (
        workload.concurrency
        * workload.prompt_tokens
        * manifest.config.hidden_size
        * DTYPE_BYTES[workload.activation_dtype]
    )
    activation_buffer_bytes = activation_payload * worker.activation_buffer_count
    workspace_bytes = worker.fixed_workspace_bytes
    if worker.primary_memory_domain in (MemoryDomain.HOST, MemoryDomain.UNIFIED):
        # Host/unified stages and transport staging share the same capacity.
        workspace_bytes += worker.host_transport_buffer_bytes
    subtotal = weight_bytes + kv_cache_bytes + workspace_bytes + activation_buffer_bytes
    allocator_allowance_bytes = math.ceil(subtotal * worker.allocator_allowance_fraction)
    required_bytes = subtotal + allocator_allowance_bytes
    usable_bytes = budget.usable_bytes
    pressure = required_bytes / usable_bytes if usable_bytes else math.inf
    return StageMemory(
        worker_id=worker.worker_id,
        domain=worker.primary_memory_domain,
        layer_start=start,
        layer_end=end,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        workspace_bytes=workspace_bytes,
        activation_buffer_bytes=activation_buffer_bytes,
        allocator_allowance_bytes=allocator_allowance_bytes,
        required_bytes=required_bytes,
        usable_bytes=usable_bytes,
        pressure=pressure,
    )


def _find_link(
    links: Sequence[LinkProfile], source_worker_id: str, target_worker_id: str
) -> LinkProfile | None:
    return next(
        (
            link
            for link in links
            if link.source_worker_id == source_worker_id
            and link.target_worker_id == target_worker_id
        ),
        None,
    )


def _boundary_estimate(
    manifest: ModelManifest,
    workload: WorkloadProfile,
    link: LinkProfile,
) -> PerformanceEstimate:
    bytes_per_value = DTYPE_BYTES[workload.activation_dtype]
    prefill_bytes = (
        workload.concurrency
        * workload.prompt_tokens
        * manifest.config.hidden_size
        * bytes_per_value
    )
    decode_bytes = workload.concurrency * manifest.config.hidden_size * bytes_per_value
    conversion = link.sender_conversion_ms + link.receiver_conversion_ms
    prefill_ms = (
        link.fixed_latency_ms
        + (prefill_bytes / link.bandwidth_bytes_per_second * 1000)
        + conversion
    )
    decode_ms = (
        link.fixed_latency_ms + (decode_bytes / link.bandwidth_bytes_per_second * 1000) + conversion
    )
    confidence = "measured-link" if link.provenance.value == "MEASURED" else "configured-link"
    return PerformanceEstimate(
        boundary_prefill_ms=prefill_ms,
        boundary_decode_ms=decode_ms,
        confidence=confidence,
    )


def _relay_penalty(connection_type: ConnectionType) -> float:
    return {
        ConnectionType.DIRECT: 0.0,
        ConnectionType.PEER_RELAYED: 100.0,
        ConnectionType.DERP_RELAYED: 1000.0,
        ConnectionType.UNKNOWN: 10.0,
    }[connection_type]


def _candidate_score(
    stages: tuple[StageMemory, StageMemory],
    performance: PerformanceEstimate | None,
    settings: PlannerSettings,
    link: LinkProfile | None,
) -> float:
    maximum_pressure = max(stage.pressure for stage in stages)
    imbalance = abs(stages[0].pressure - stages[1].pressure)
    if settings.mode == PlanningMode.FEASIBILITY:
        return maximum_pressure + imbalance * 0.01
    if performance is None:
        raise PlanningError("estimated planning requires a directional link profile")
    weights = settings.objective_weights
    if settings.mode == PlanningMode.MEASURED:
        assert performance.ttft_ms is not None
        average_itl = (
            sum(performance.decode_ms) / len(performance.decode_ms) if performance.decode_ms else 0
        )
        return (
            weights.ttft * performance.ttft_ms
            + (weights.itl + weights.pipeline_period) * average_itl
            + weights.memory_pressure * maximum_pressure**2
        )
    boundary_score = (
        weights.ttft * performance.boundary_prefill_ms
        + weights.itl * performance.boundary_decode_ms
        + weights.pipeline_period * performance.boundary_decode_ms
    )
    relay = _relay_penalty(link.connection_type) if link else 0.0
    return boundary_score + weights.memory_pressure * maximum_pressure**2 + relay


def _validate_workers(
    manifest: ModelManifest, workers: Sequence[WorkerProfile], settings: PlannerSettings
) -> None:
    if len(workers) != 2:
        raise PlanningError("Milestone 0 supports exactly two workers")
    if workers[0].worker_id == workers[1].worker_id:
        raise PlanningError("worker IDs must be unique")
    for worker in workers:
        if manifest.architecture.architecture_id not in worker.supported_architectures:
            raise PlanningError(
                f"worker {worker.worker_id} does not support "
                f"{manifest.architecture.architecture_id}"
            )
        if settings.weight_dtype and not worker.supports_mixed_precision:
            raise PlanningError(f"worker {worker.worker_id} does not support mixed precision")
        if settings.execution_dtype not in worker.supported_execution_dtypes:
            raise PlanningError(
                f"worker {worker.worker_id} does not support {settings.execution_dtype.value}"
            )


def create_plan(
    manifest: ModelManifest,
    workers: Sequence[WorkerProfile],
    links: Sequence[LinkProfile],
    workload: WorkloadProfile,
    settings: PlannerSettings | None = None,
    *,
    profile_bundle: ProfileBundle | DiskProfileBundle | None = None,
) -> PlanningReport:
    settings = settings or PlannerSettings()
    version = "0.2.0" if settings.mode == PlanningMode.MEASURED else PLANNER_VERSION
    _validate_workers(manifest, workers, settings)
    if settings.weight_dtype and workload.kv_dtype != settings.execution_dtype:
        raise PlanningError("mixed execution and KV dtypes must agree")
    if settings.mode == PlanningMode.MEASURED:
        if profile_bundle is None:
            raise PlanningError("measured planning requires a profile bundle")
        if {w.worker_id: w for w in workers} != {
            w.worker.worker_id: w.worker for w in profile_bundle.workers
        }:
            raise PlanningError("worker configuration differs from frozen bundle")
        if settings.execution_dtype != workload.kv_dtype:
            raise PlanningError("measured execution and KV dtypes must agree")
    elif profile_bundle is not None:
        raise PlanningError("profile bundle requires measured mode")
    # Every two-stage candidate ships the whole prompt across one boundary message.
    boundary_limit = maximum_boundary_tokens(manifest.config.hidden_size, workload.activation_dtype)
    raw_candidates: list[PlanCandidate] = []
    for first, final in ((workers[0], workers[1]), (workers[1], workers[0])):
        link = _find_link(links, first.worker_id, final.worker_id)
        for split in range(1, manifest.config.num_layers):
            stages = (
                _stage_memory(
                    manifest,
                    first,
                    workload,
                    settings,
                    start=0,
                    end=split,
                    is_first=True,
                    is_final=False,
                ),
                _stage_memory(
                    manifest,
                    final,
                    workload,
                    settings,
                    start=split,
                    end=manifest.config.num_layers,
                    is_first=False,
                    is_final=True,
                ),
            )
            reasons: list[str] = []
            for stage, worker in zip(stages, (first, final), strict=True):
                if stage.required_bytes > stage.usable_bytes:
                    reasons.append(f"{stage.domain.value}_MEMORY_EXCEEDED:{stage.worker_id}")
                pinned_budget = worker.budget_for(MemoryDomain.HOST_PINNED)
                host_budget = worker.budget_for(MemoryDomain.HOST)
                if pinned_budget is not None and host_budget is None:
                    reasons.append(f"MISSING_HOST_MEMORY_BUDGET:{stage.worker_id}")
                for transport_budget in (host_budget, pinned_budget):
                    if (
                        transport_budget is not None
                        and worker.host_transport_buffer_bytes > transport_budget.usable_bytes
                    ):
                        reasons.append(
                            f"{transport_budget.domain.value}_MEMORY_EXCEEDED:{stage.worker_id}"
                        )
            if workload.prompt_tokens > boundary_limit:
                reasons.append(
                    f"BOUNDARY_PAYLOAD_EXCEEDED:{workload.prompt_tokens}>{boundary_limit}"
                )
            if settings.mode == PlanningMode.ESTIMATED and link is None:
                reasons.append(f"MISSING_LINK_PROFILE:{first.worker_id}->{final.worker_id}")
            performance = _boundary_estimate(manifest, workload, link) if link else None
            measurement_status = None
            if settings.mode == PlanningMode.MEASURED:
                assert profile_bundle is not None
                measurement_status, measured_reasons, performance = evaluate(
                    profile_bundle,
                    manifest,
                    workload,
                    assignments_for(
                        first.worker_id, final.worker_id, split, manifest.config.num_layers
                    ),
                    weight_dtype=settings.weight_dtype,
                )
                if reasons:
                    measurement_status = "infeasible"
                reasons.extend(measured_reasons)
            feasible = not reasons
            score = _candidate_score(stages, performance, settings, link) if feasible else None
            candidate_id = f"{first.worker_id}--{final.worker_id}-m{split:03d}"
            raw_candidates.append(
                PlanCandidate(
                    candidate_id=candidate_id,
                    stage_zero_worker_id=first.worker_id,
                    final_stage_worker_id=final.worker_id,
                    split_layer=split,
                    stages=stages,
                    feasible=feasible,
                    measurement_status=measurement_status,
                    rejection_reasons=tuple(reasons),
                    performance=performance,
                    score=score,
                )
            )

    feasible_candidates = sorted(
        (item for item in raw_candidates if item.feasible),
        key=lambda item: (item.score if item.score is not None else math.inf, item.candidate_id),
    )
    ranks = {candidate.candidate_id: rank for rank, candidate in enumerate(feasible_candidates, 1)}
    candidates = tuple(
        candidate.model_copy(update={"rank": ranks.get(candidate.candidate_id)})
        for candidate in raw_candidates
    )
    if not feasible_candidates:
        return PlanningReport(
            planner_version=version,
            settings=settings if settings.mode == PlanningMode.MEASURED else None,
            manifest_digest=manifest.manifest_digest,
            workload_id=workload.workload_id,
            mode=settings.mode,
            selected_candidate_id=None,
            candidates=candidates,
            plan=None,
            notes=("No fully qualified feasible two-worker placement was found.",),
        )

    selected_id = feasible_candidates[0].candidate_id
    selected = next(item for item in candidates if item.candidate_id == selected_id)
    assignments = (
        StageAssignment(
            stage_index=0,
            worker_id=selected.stage_zero_worker_id,
            layer_start=0,
            layer_end=selected.split_layer,
            owns_token_embedding=True,
            owns_final_norm=False,
            owns_lm_head=False,
            owns_sampling=False,
        ),
        StageAssignment(
            stage_index=1,
            worker_id=selected.final_stage_worker_id,
            layer_start=selected.split_layer,
            layer_end=manifest.config.num_layers,
            owns_token_embedding=False,
            owns_final_norm=True,
            owns_lm_head=True,
            owns_sampling=True,
        ),
    )
    duplicated_groups = ("token_embeddings",) if manifest.config.tied_embeddings else ()
    unsigned_plan: dict[str, object] = {
        "planner_version": version,
        "deployment_version": 1,
        "manifest_digest": manifest.manifest_digest,
        "workload_id": workload.workload_id,
        "planning_mode": settings.mode.value,
        "execution_dtype": settings.execution_dtype.value,
        "activation_dtype": workload.activation_dtype.value,
        "split_layer": selected.split_layer,
        "stages": [item.model_dump(mode="json") for item in assignments],
        "duplicated_tensor_groups": duplicated_groups,
        "selected_candidate_id": selected.candidate_id,
    }
    measured_fields = {}
    if settings.mode == PlanningMode.MEASURED:
        assert profile_bundle is not None
        measured_fields = {
            "schema_version": "1.1",
            "workload_digest": digest(workload),
            "profile_bundle_digest": profile_bundle.bundle_digest,
        }
        unsigned_plan.update(measured_fields)
    if settings.weight_dtype:
        measured_fields.update(schema_version="1.2")
        unsigned_plan.update(
            schema_version="1.2",
            weight_dtype=settings.weight_dtype.value,
            workload_digest=measured_fields.get("workload_digest"),
            profile_bundle_digest=measured_fields.get("profile_bundle_digest"),
        )
    plan_digest = hashlib.sha256(canonical_json_bytes(unsigned_plan)).hexdigest()
    plan = DeploymentPlan(
        **measured_fields,
        planner_version=version,
        plan_id=f"plan-{plan_digest[:16]}",
        deployment_version=1,
        plan_digest=plan_digest,
        manifest_digest=manifest.manifest_digest,
        workload_id=workload.workload_id,
        planning_mode=settings.mode,
        execution_dtype=settings.execution_dtype,
        weight_dtype=settings.weight_dtype,
        activation_dtype=workload.activation_dtype,
        split_layer=selected.split_layer,
        stages=assignments,
        duplicated_tensor_groups=duplicated_groups,
        selected_candidate_id=selected.candidate_id,
    )
    notes = (
        "Measured mode uses exact assignments, whole-stage timings including conversion, "
        "per-context decode, allocation, request setup and directional encode plus RTT. "
        "Unknown candidates are unranked. Equal scores use candidate ID."
        if settings.mode == PlanningMode.MEASURED
        else "Feasibility mode ranks maximum memory pressure before imbalance."
        if settings.mode == PlanningMode.FEASIBILITY
        else (
            "Estimated mode currently scores boundary transfer and memory pressure; "
            "compute profiles arrive in Milestone 5."
        ),
    )
    return PlanningReport(
        planner_version=version,
        settings=settings if settings.mode == PlanningMode.MEASURED else None,
        manifest_digest=manifest.manifest_digest,
        workload_id=workload.workload_id,
        mode=settings.mode,
        selected_candidate_id=selected.candidate_id,
        candidates=candidates,
        plan=plan,
        notes=notes,
    )


def assignments_for(
    first: str, final: str, split: int, layers: int
) -> tuple[StageAssignment, StageAssignment]:
    return (
        StageAssignment(
            stage_index=0,
            worker_id=first,
            layer_start=0,
            layer_end=split,
            owns_token_embedding=True,
            owns_final_norm=False,
            owns_lm_head=False,
            owns_sampling=False,
        ),
        StageAssignment(
            stage_index=1,
            worker_id=final,
            layer_start=split,
            layer_end=layers,
            owns_token_embedding=False,
            owns_final_norm=True,
            owns_lm_head=True,
            owns_sampling=True,
        ),
    )
