from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hllm_control.models import Backend, ConnectionType, DType, WorkloadProfile
from hllm_control.profiling.memory import PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    MEMORY_PHASES,
    AllocatorSample,
    ComputeMeasurement,
    Conditions,
    ConversionMeasurement,
    Environment,
    LinkMeasurement,
    MemoryAmounts,
    MemoryMeasurement,
    MemorySample,
    PhysicalSample,
    ProfileArtifact,
    ProfileKey,
    check_compatibility,
    digest,
    make_artifact,
)
from hllm_control.profiling.runner import checkpoint_digest

from tests.process_helpers import plan, write_model


@pytest.fixture
def key(tmp_path: Path) -> ProfileKey:
    root = tmp_path / "model"
    manifest = write_model(root)
    workload = WorkloadProfile(
        workload_id="memory-test",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    return ProfileKey(
        manifest_digest=manifest.manifest_digest,
        checkpoint_digest=checkpoint_digest(root, manifest),
        environment=Environment(
            backend=Backend.CPU,
            device_identity="cpu-1",
            device_name="CPU",
            backend_version="hllm-native",
            driver_version="not-applicable",
            allocator="system",
            allocator_config="default",
            source_revision="test",
            binary_digest="a" * 64,
            compiler="clang",
            os="test-os",
        ),
        assignment=plan(manifest).stages[0],
        workload=workload,
        workload_digest=digest(workload),
        execution_dtype=DType.F32,
        transport_mode="pageable",
        input_kind="synthetic-shape",
        input_digest="b" * 64,
    )


def conditions() -> Conditions:
    return Conditions(
        measured_at=datetime(2026, 9, 8, tzinfo=UTC),
        warmup_cycles=0,
        measured_cycles=1,
        process_policy="fresh-process-then-reloads",
        concurrent_load="idle",
    )


def memory(key: ProfileKey, *, allocator: bool = False) -> MemoryMeasurement:
    return MemoryMeasurement(
        admission_capacity=MemoryAmounts(host=1000, device=1000),
        completed=True,
        samples=tuple(
            MemorySample(
                cycle=0,
                phase=phase,
                weights=MemoryAmounts(),
                cache=MemoryAmounts(),
                workspace=MemoryAmounts(),
                allocator=AllocatorSample(
                    active_bytes=100, cached_bytes=100, peak_bytes=300, peak_scope="phase"
                )
                if allocator
                else None,
                rss_bytes=400,
                rss_lifetime_peak_bytes=500,
                device_available_bytes=2000,
                completed_steps=key.workload.output_tokens
                if phase in ("decode", "request_cleanup", "unload")
                else 0,
            )
            for phase in MEMORY_PHASES
        ),
        physical_samples=(
            PhysicalSample(elapsed_seconds=1, rss_bytes=400, device_process_bytes=350),
        ),
        physical_sample_interval_seconds=0.25,
    )


@pytest.mark.parametrize("kind", ["memory", "compute", "conversion", "link"])
def test_artifact_round_trip_and_tamper_detection(key: ProfileKey, kind: str) -> None:
    measurement = {
        "memory": memory(key),
        "compute": ComputeMeasurement(
            phase="prefill", context_tokens=0, component="stage", samples_ms=(1.0,)
        ),
        "conversion": ConversionMeasurement(
            direction="to-wire", payload_bytes=64, samples_ms=(1.0,)
        ),
        "link": LinkMeasurement(
            source_worker_id="a",
            target_worker_id="b",
            source_environment=key.environment,
            target_environment=key.environment,
            connection_type=ConnectionType.DIRECT,
            payload_bytes=64,
            stream_policy="persistent",
            attribution="round-trip-including-feedback",
            samples_ms=(1.0,),
        ),
    }[kind]
    artifact = make_artifact(key, conditions(), measurement)
    assert ProfileArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    assert make_artifact(key, conditions(), measurement).artifact_digest == artifact.artifact_digest
    modified = artifact.model_dump(mode="json")
    modified["key"]["checkpoint_digest"] = "f" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        ProfileArtifact.model_validate(modified)
    modified["schema_version"] = "2.0"
    with pytest.raises(ValueError):
        ProfileArtifact.model_validate(modified)


@pytest.mark.parametrize(
    "field,value",
    [("checkpoint_digest", "c" * 64), ("manifest_digest", "c" * 64), ("input_digest", "c" * 64)],
)
def test_incompatible_identity(key: ProfileKey, field: str, value: str) -> None:
    artifact = make_artifact(key, conditions(), memory(key))
    changed = ProfileKey.model_validate({**key.model_dump(), field: value})
    assert check_compatibility(artifact, changed, kind="memory").status == "incompatible"


def test_device_ownership_and_shape_compatibility(key: ProfileKey) -> None:
    artifact = make_artifact(key, conditions(), memory(key))
    for field in ("device_identity", "backend_version", "driver_version", "allocator_config"):
        environment = {**key.environment.model_dump(), field: "different"}
        changed = ProfileKey.model_validate({**key.model_dump(), "environment": environment})
        assert check_compatibility(artifact, changed, kind="memory").status == "incompatible"
    assignment = {**key.assignment.model_dump(), "layer_end": key.assignment.layer_end + 1}
    changed = ProfileKey.model_validate({**key.model_dump(), "assignment": assignment})
    assert check_compatibility(artifact, changed, kind="memory").status == "incompatible"
    workload = key.workload.model_copy(update={"total_cached_tokens": 8})
    changed = ProfileKey.model_validate(
        {**key.model_dump(), "workload": workload, "workload_digest": digest(workload)}
    )
    assert check_compatibility(artifact, changed, kind="memory").status == "out-of-range"
    assert check_compatibility(None, key, kind="memory").status == "missing"
    assert check_compatibility(artifact, key, kind="memory").status == "compatible"


@pytest.mark.parametrize(
    "field,value",
    [
        ("concurrency", 2),
        ("total_cached_tokens", 2),
        ("activation_dtype", "F32"),
        ("kv_dtype", "F16"),
    ],
)
def test_unsupported_workload(key: ProfileKey, field: str, value: object) -> None:
    workload = WorkloadProfile.model_validate({**key.workload.model_dump(), field: value})
    with pytest.raises(ValueError):
        ProfileKey.model_validate(
            {**key.model_dump(), "workload": workload, "workload_digest": digest(workload)}
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_timings(value: float) -> None:
    with pytest.raises(ValueError):
        ConversionMeasurement(direction="to-wire", payload_bytes=1, samples_ms=(value,))


def test_timing_scope_requires_exact_payload_and_direction(key: ProfileKey) -> None:
    m = ConversionMeasurement(direction="to-wire", payload_bytes=64, samples_ms=(1.0,))
    a = make_artifact(key, conditions(), m)
    scope = m.model_dump(mode="json", exclude={"samples_ms"})
    assert check_compatibility(a, key, kind="conversion", scope=scope).status == "compatible"
    assert check_compatibility(a, key, kind="conversion").status == "incompatible"
    assert (
        check_compatibility(a, key, kind="conversion", scope={**scope, "payload_bytes": 128}).status
        == "incompatible"
    )


def test_memory_completeness_and_failure(key: ProfileKey) -> None:
    m = memory(key)
    with pytest.raises(ValueError, match="all ordered phases"):
        make_artifact(key, conditions(), m.model_copy(update={"samples": m.samples[:-1]}))
    failed = m.model_copy(update={"completed": False, "error": "allocation failed"})
    a = make_artifact(key, conditions(), failed)
    assert check_compatibility(a, key, kind="memory").status == "incomplete"
    sample = m.samples[-1].model_copy(update={"cache": MemoryAmounts(host=1)})
    with pytest.raises(ValueError, match="reservations did not retire"):
        make_artifact(
            key, conditions(), m.model_copy(update={"samples": (*m.samples[:-1], sample)})
        )


def test_fit_keeps_reservations_and_physical_usage_separate(key: ProfileKey) -> None:
    m = memory(key)
    a = make_artifact(key, conditions(), m)
    budget = PhysicalBudget(available_bytes=1000, headroom_bytes=100, extra_overhead_bytes=50)
    result = assess_fit(a, m.admission_capacity, budget)
    assert result.status == "safe"
    assert result.host_envelope_bytes == 600  # 500 * 1.1 + 50; no added reservation bytes.
    assert (
        assess_fit(
            a, m.admission_capacity, budget.model_copy(update={"available_bytes": 650})
        ).status
        == "unsafe"
    )
    assert assess_fit(a, MemoryAmounts(host=999, device=1000), budget).status == "unknown"
    assert (
        assess_fit(
            a, m.admission_capacity, budget.model_copy(update={"available_bytes": None})
        ).status
        == "unknown"
    )
    missing = m.model_copy(update={"physical_samples": ()})
    cuda_key = ProfileKey.model_validate(
        {**key.model_dump(), "environment": {**key.environment.model_dump(), "backend": "cuda"}}
    )
    cuda = make_artifact(cuda_key, conditions(), missing)
    result = assess_fit(cuda, m.admission_capacity, budget, budget)
    assert result.status == "unknown"
    assert "CUDA physical process observations unavailable" in result.reasons


def test_checkpoint_hash_detects_payload_change(tmp_path: Path) -> None:
    root = tmp_path / "model"
    manifest = write_model(root)
    first = checkpoint_digest(root, manifest)
    path = root / manifest.tensor_files[0].name
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    assert checkpoint_digest(root, manifest) != first


def test_milestone_workload_preserves_long_capacity_example() -> None:
    from hllm_control.planner.config import load_settings, load_workload

    w = load_workload(Path("examples/workloads/milestone-5.yaml"))
    assert (w.prompt_tokens, w.output_tokens, w.total_cached_tokens) == (512, 256, 768)
    assert load_workload(Path("examples/workloads/interactive.yaml")).total_cached_tokens == 32768
    weights = load_settings(Path("examples/profiles/milestone-5-objective.yaml")).objective_weights
    assert (weights.ttft, weights.itl, weights.pipeline_period, weights.memory_pressure) == (
        1,
        255,
        0,
        0,
    )


def test_mlx_unknown_overlap_uses_conservative_union_bound(key: ProfileKey) -> None:
    mlx = ProfileKey.model_validate(
        {**key.model_dump(), "environment": {**key.environment.model_dump(), "backend": "mlx"}}
    )
    m = memory(mlx, allocator=True)
    a = make_artifact(mlx, conditions(), m)
    b = PhysicalBudget(available_bytes=2000, headroom_bytes=100, extra_overhead_bytes=50)
    result = assess_fit(a, m.admission_capacity, b)
    assert result.status == "safe"
    assert result.host_envelope_bytes == 930  # (RSS 500 + allocator 300) * 1.1 + 50.


def test_preflight_refuses_to_start_and_saves_report(tmp_path: Path, monkeypatch) -> None:
    from hllm_control.profiling.runner import run_memory_profile

    from tests.process_helpers import plan, write_model

    root = tmp_path / "model"
    manifest = write_model(root)
    monkeypatch.setattr("hllm_control.profiling.runner.host_available_bytes", lambda: 10)
    w = WorkloadProfile(
        workload_id="preflight",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    with pytest.raises(ValueError, match="exceeds availability"):
        run_memory_profile(
            binary=Path(sys.executable),
            root=root,
            manifest=manifest,
            plan=plan(manifest),
            stage_index=0,
            workload=w,
            backend=Backend.CPU,
            capacity=MemoryAmounts(host=1000),
            output=tmp_path / "rejected.json",
            source_revision="test",
            concurrent_load="test",
        )
    report = (tmp_path / "rejected.preflight.json").read_text()
    assert '"status": "rejected"' in report
    assert not (tmp_path / "rejected.native.jsonl").exists()
