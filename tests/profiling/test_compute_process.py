"""Exercise paired completed-work timing in the actual backend process."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hllm_control.models import Backend, DType, WorkloadProfile
from hllm_control.profiling.models import (
    ComputeRunMeasurement,
    MemoryAmounts,
    ProfileArtifact,
    digest,
)
from hllm_control.profiling.runner import run_memory_profile

from tests.process_helpers import plan, write_model

BINARY = os.environ.get("HLLM_COMPUTE_PROFILER")
pytestmark = pytest.mark.skipif(BINARY is None, reason="native compute profiler not configured")


@pytest.mark.parametrize("stage_index", [0, 1])
@pytest.mark.parametrize("pinned", [False, True])
def test_paired_compute_and_conversion(tmp_path: Path, stage_index: int, pinned: bool) -> None:
    assert BINARY is not None
    backend = Backend(os.environ["HLLM_PROFILE_BACKEND"])
    if pinned and backend != Backend.CUDA:
        pytest.skip("pinned transport is CUDA-only")
    manifest = write_model(tmp_path / "model", redundant_head=True)
    dtype = DType.F32 if backend == Backend.CPU else DType.F16
    deployment = plan(manifest).model_copy(update={"execution_dtype": dtype})
    workload = WorkloadProfile(
        workload_id="compute-test",
        prompt_tokens=8,
        output_tokens=4,
        total_cached_tokens=12,
        kv_dtype=dtype,
    )
    output = tmp_path / "compute.json"
    artifact = run_memory_profile(
        mode="compute",
        binary=Path(BINARY),
        root=tmp_path / "model",
        manifest=manifest,
        plan=deployment,
        stage_index=stage_index,
        workload=workload,
        backend=backend,
        capacity=MemoryAmounts(
            host=32 * 1024**2,
            device=32 * 1024**2,
            unified=32 * 1024**2,
            pinned=4 * 1024**2 if pinned else 0,
        ),
        transport_mode="pinned" if pinned else "pageable",
        output=output,
        source_revision="test",
        concurrent_load="test suite",
        warmup_cycles=1,
        measured_cycles=2,
        timeout_seconds=60,
        host_headroom_bytes=0,
        device_headroom_bytes=0,
    )
    assert isinstance(artifact.measurement, ComputeRunMeasurement)
    assert artifact.measurement.completed, artifact.measurement.error
    assert artifact.measurement.identical_output_cycles == (0, 1, 2)
    assert ProfileArtifact.model_validate_json(output.read_text()) == artifact
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert len(summary["instrumentation_comparison"]) == 4
    assert all(len(b["samples_ms"]) == 2 for b in summary["buckets"])
    assert {r.context_tokens for r in artifact.measurement.records} == {0, 8, 9, 10}
    assert not output.with_suffix(".fit.json").exists()

    # Re-seal corrupt content to exercise semantic validation rather than just hashing.
    for corruption in ("missing-layer", "wrong-context", "overlap"):
        data = artifact.model_dump(mode="json", exclude={"artifact_digest"})
        rows = data["measurement"]["records"]
        row = next(r for r in rows if r["component"] == "layer")
        if corruption == "missing-layer":
            rows.remove(row)
        elif corruption == "wrong-context":
            row["context_tokens"] = 123
        else:
            row["milliseconds"] += 100
        with pytest.raises(ValueError):
            ProfileArtifact.model_validate({**data, "artifact_digest": digest(data)})
