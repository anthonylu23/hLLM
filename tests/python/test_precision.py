"""Mixed precision is an explicit identity; legacy serialization stays unchanged."""

from pathlib import Path

import pytest
from hllm_control.models import (
    Backend,
    DeploymentPlan,
    DType,
    MemoryBudget,
    MemoryDomain,
    PlannerSettings,
    WorkerProfile,
    WorkloadProfile,
)
from hllm_control.planner.planner import PlanningError, create_plan
from hllm_control.prepare.manifest import prepare_model
from hllm_control.profiling.models import (
    ProfileArtifact,
    ProfileKey,
    check_compatibility,
    digest,
    make_artifact,
)
from hllm_control.wire import deployment_plan_from_proto, deployment_plan_to_proto

from tests.python.test_profiling import conditions, key, memory  # noqa: F401


def test_mixed_plan_roundtrip_and_residency(tiny_model: Path) -> None:
    manifest = prepare_model(tiny_model)
    workers = tuple(
        WorkerProfile(
            worker_id=name,
            endpoint=f"{name}:1",
            backend=Backend.MLX,
            primary_memory_domain=MemoryDomain.UNIFIED,
            supported_architectures=("llama.v1",),
            supported_execution_dtypes=(DType.F32,),
            supports_mixed_precision=True,
            memory_budgets=(MemoryBudget(domain=MemoryDomain.UNIFIED, capacity_bytes=100_000_000),),
        )
        for name in ("a", "b")
    )
    workload = WorkloadProfile(
        workload_id="mixed",
        prompt_tokens=4,
        output_tokens=4,
        total_cached_tokens=8,
        kv_dtype=DType.F32,
    )
    legacy = create_plan(
        manifest, workers, (), workload, PlannerSettings(execution_dtype=DType.F32)
    )
    mixed = create_plan(
        manifest,
        workers,
        (),
        workload,
        PlannerSettings(execution_dtype=DType.F32, weight_dtype=DType.F16),
    )
    assert legacy.plan is not None and mixed.plan is not None
    assert "weight_dtype" not in legacy.plan.model_dump(mode="json")
    assert not deployment_plan_to_proto(legacy.plan).HasField("weight_dtype")
    assert mixed.plan.schema_version == "1.2"
    assert mixed.plan.plan_digest != legacy.plan.plan_digest
    assert mixed.plan.plan_digest == digest(
        mixed.plan.model_dump(mode="json", exclude={"plan_id", "plan_digest"})
    )
    assert deployment_plan_from_proto(deployment_plan_to_proto(mixed.plan)) == mixed.plan
    for a, b in zip(legacy.candidates, mixed.candidates, strict=True):
        for x, y in zip(a.stages, b.stages, strict=True):
            assert x.weight_bytes == 2 * y.weight_bytes
            assert x.kv_cache_bytes == y.kv_cache_bytes
    with pytest.raises(ValueError, match=r"schema 1\.2"):
        DeploymentPlan.model_validate({**mixed.plan.model_dump(), "schema_version": "1.0"})
    with pytest.raises(PlanningError, match="mixed precision"):
        create_plan(
            manifest,
            tuple(w.model_copy(update={"supports_mixed_precision": False}) for w in workers),
            (),
            workload,
            PlannerSettings(execution_dtype=DType.F32, weight_dtype=DType.F16),
        )


def test_mixed_profiles_cannot_reuse_legacy_evidence(key: ProfileKey) -> None:  # noqa: F811
    gpu = key.model_copy(
        update={"environment": key.environment.model_copy(update={"backend": Backend.MLX})}
    )
    mixed = ProfileKey.model_validate({**gpu.model_dump(), "weight_dtype": "F16"})
    old = make_artifact(gpu, conditions(), memory(gpu))
    new = make_artifact(mixed, conditions(), memory(mixed))
    assert "weight_dtype" not in old.key.model_dump()
    assert new.schema_version == "1.2"
    assert check_compatibility(old, mixed, kind="memory").status == "incompatible"
    assert check_compatibility(new, mixed, kind="memory").status == "compatible"
    tampered = new.model_dump(mode="json", exclude={"artifact_digest"})
    tampered["schema_version"] = "1.0"
    with pytest.raises(ValueError, match=r"schema 1\.2"):
        ProfileArtifact.model_validate({**tampered, "artifact_digest": digest(tampered)})


@pytest.mark.parametrize(
    "resident,execution", [(DType.F32, DType.F32), (DType.F16, DType.F16), (DType.BF16, DType.F32)]
)
def test_unsupported_precision_pairs_fail(resident: DType, execution: DType) -> None:
    with pytest.raises(ValueError, match="F16 weights and F32"):
        PlannerSettings(weight_dtype=resident, execution_dtype=execution)
