from __future__ import annotations

from pathlib import Path

from hllm_control.models import Backend, DType, MemoryBudget, MemoryDomain, WorkerProfile
from hllm_control.planner.planner import create_plan
from hllm_control.prepare.manifest import prepare_model
from hllm_control.wire import (
    deployment_plan_from_proto,
    deployment_plan_to_proto,
    model_manifest_from_proto,
    model_manifest_to_proto,
)


def _worker(worker_id: str) -> WorkerProfile:
    return WorkerProfile(
        worker_id=worker_id,
        endpoint=f"{worker_id}:50051",
        backend=Backend.CPU,
        primary_memory_domain=MemoryDomain.HOST,
        supported_architectures=("llama.v1",),
        supported_execution_dtypes=(DType.F16,),
        memory_budgets=(MemoryBudget(domain=MemoryDomain.HOST, capacity_bytes=10_000_000),),
    )


def test_manifest_round_trips_through_wire_contract(tiny_model: Path) -> None:
    manifest = prepare_model(tiny_model, model_id="test/tiny", revision="abc123")

    encoded = model_manifest_to_proto(manifest).SerializeToString(deterministic=True)
    decoded = type(model_manifest_to_proto(manifest)).FromString(encoded)

    assert model_manifest_from_proto(decoded) == manifest


def test_deployment_plan_round_trips_through_wire_contract(tiny_model: Path) -> None:
    from hllm_control.models import WorkloadProfile

    manifest = prepare_model(tiny_model)
    report = create_plan(
        manifest,
        (_worker("cpu-a"), _worker("cpu-b")),
        (),
        WorkloadProfile(
            workload_id="wire-test",
            prompt_tokens=4,
            output_tokens=4,
            total_cached_tokens=32,
        ),
    )
    assert report.plan is not None

    encoded = deployment_plan_to_proto(report.plan).SerializeToString(deterministic=True)
    decoded = type(deployment_plan_to_proto(report.plan)).FromString(encoded)

    assert deployment_plan_from_proto(decoded) == report.plan


def test_single_stage_reference_plan_roundtrip() -> None:
    from hllm_control.models import DeploymentPlan, PlanningMode, StageAssignment

    plan = DeploymentPlan(
        plan_id="reference",
        plan_digest="reference-digest",
        manifest_digest="manifest",
        workload_id="reference",
        planning_mode=PlanningMode.FEASIBILITY,
        execution_dtype=DType.F32,
        activation_dtype=DType.F16,
        split_layer=0,
        stages=(
            StageAssignment(
                stage_index=0,
                worker_id="cpu-a",
                layer_start=0,
                layer_end=4,
                owns_token_embedding=True,
                owns_final_norm=True,
                owns_lm_head=True,
                owns_sampling=True,
            ),
        ),
        selected_candidate_id="reference",
    )
    assert deployment_plan_from_proto(deployment_plan_to_proto(plan)) == plan
