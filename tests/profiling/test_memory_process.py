"""Owned-process profiling checks, registered separately for each native backend."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hllm_control.models import Backend, DType, WorkloadProfile
from hllm_control.profiling.models import MemoryAmounts, MemoryMeasurement, ProfileArtifact
from hllm_control.profiling.runner import run_memory_profile

from tests.process_helpers import plan, write_model

BINARY = os.environ.get("HLLM_MEMORY_PROFILER")
pytestmark = pytest.mark.skipif(BINARY is None, reason="native profiling binary not configured")


@pytest.mark.parametrize("stage_index", [0, 1])
def test_native_phases_reload_and_cleanup(tmp_path: Path, stage_index: int) -> None:
    assert BINARY is not None
    backend = Backend(os.environ["HLLM_PROFILE_BACKEND"])
    manifest = write_model(tmp_path / "model", redundant_head=True)
    deployment = plan(manifest)
    dtype = DType.F32 if backend == Backend.CPU else DType.F16
    deployment = deployment.model_copy(update={"execution_dtype": dtype})
    w = WorkloadProfile(
        workload_id="native-memory",
        prompt_tokens=8,
        output_tokens=4,
        total_cached_tokens=12,
        kv_dtype=dtype,
    )
    output = tmp_path / "memory.json"
    a = run_memory_profile(
        binary=Path(BINARY),
        root=tmp_path / "model",
        manifest=manifest,
        plan=deployment,
        stage_index=stage_index,
        workload=w,
        backend=backend,
        capacity=MemoryAmounts(host=32 * 1024**2, device=32 * 1024**2, unified=32 * 1024**2),
        output=output,
        source_revision="test",
        concurrent_load="test suite",
        measured_cycles=3,
        timeout_seconds=60,
        host_headroom_bytes=0,
        device_headroom_bytes=0,
    )
    assert isinstance(a.measurement, MemoryMeasurement)
    assert a.measurement.completed
    assert ProfileArtifact.model_validate_json(output.read_text()) == a
    assert len(a.measurement.samples) == 21
    loads = [s for s in a.measurement.samples if s.phase == "load"]
    assert loads[0].weights == loads[1].weights == loads[2].weights
    if backend != Backend.CPU:
        prefill = [s for s in a.measurement.samples if s.phase == "prefill"]
        assert all(s.allocator is not None and s.allocator.peak_scope == "phase" for s in prefill)
        # A reset unload peak is lower than the preceding allocation/execution peak.
        unload = [s for s in a.measurement.samples if s.phase == "unload"]
        assert all(s.allocator is not None for s in unload)
        assert max(s.allocator.peak_bytes for s in unload if s.allocator) <= max(
            s.allocator.peak_bytes for s in prefill if s.allocator
        )
    with pytest.raises(FileExistsError):
        run_memory_profile(
            binary=Path(BINARY),
            root=tmp_path / "model",
            manifest=manifest,
            plan=deployment,
            stage_index=stage_index,
            workload=w,
            backend=backend,
            capacity=MemoryAmounts(host=32 * 1024**2),
            output=output,
            source_revision="test",
            concurrent_load="test suite",
        )


def test_native_allocation_failure_preserves_evidence(tmp_path: Path) -> None:
    assert BINARY is not None
    backend = Backend(os.environ["HLLM_PROFILE_BACKEND"])
    manifest = write_model(tmp_path / "model")
    w = WorkloadProfile(
        workload_id="failure",
        prompt_tokens=8,
        output_tokens=4,
        total_cached_tokens=12,
        kv_dtype=DType.F32,
    )
    # Keep MLX startup viable, but give load admission only one host/device byte.
    capacity = MemoryAmounts(host=1, device=1, unified=1024 * 1024)
    if backend == Backend.MLX:
        # Tiny checkpoint load needs more than the 64KiB loader scratch reserve.
        capacity = MemoryAmounts(unified=65536)
    a = run_memory_profile(
        binary=Path(BINARY),
        root=tmp_path / "model",
        manifest=manifest,
        plan=plan(manifest),
        stage_index=0,
        workload=w,
        backend=backend,
        capacity=capacity,
        output=tmp_path / "failed.json",
        source_revision="test",
        concurrent_load="test suite",
        measured_cycles=1,
        host_headroom_bytes=0,
        device_headroom_bytes=0,
    )
    assert isinstance(a.measurement, MemoryMeasurement)
    assert not a.measurement.completed
    assert a.measurement.error
    assert (tmp_path / "failed.native.jsonl").exists()
    assert '"status": "unknown"' in (tmp_path / "failed.fit.json").read_text()


def test_cuda_non_native_allocator_remains_unknown(tmp_path: Path, monkeypatch) -> None:
    if os.environ["HLLM_PROFILE_BACKEND"] != "cuda":
        pytest.skip("CUDA allocator alternative")
    assert BINARY is not None
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    manifest = write_model(tmp_path / "model")
    w = WorkloadProfile(
        workload_id="async",
        prompt_tokens=8,
        output_tokens=4,
        total_cached_tokens=12,
        kv_dtype=DType.F32,
    )
    a = run_memory_profile(
        binary=Path(BINARY),
        root=tmp_path / "model",
        manifest=manifest,
        plan=plan(manifest),
        stage_index=0,
        workload=w,
        backend=Backend.CUDA,
        capacity=MemoryAmounts(host=32 * 1024**2, device=32 * 1024**2),
        output=tmp_path / "async.json",
        source_revision="test",
        concurrent_load="test suite",
        measured_cycles=1,
        host_headroom_bytes=0,
        device_headroom_bytes=0,
    )
    assert isinstance(a.measurement, MemoryMeasurement)
    assert a.measurement.completed
    assert all(s.allocator is None for s in a.measurement.samples)
    assert '"status": "unknown"' in (tmp_path / "async.fit.json").read_text()
