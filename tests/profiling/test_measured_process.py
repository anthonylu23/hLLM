"""Real native activation/hash and independent runner smoke (tiny, one output token)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from hllm_control.models import Backend, DType, PlanningMode, WorkloadProfile
from hllm_control.profiling.models import MemoryAmounts, digest
from hllm_control.profiling.runner import checkpoint_digest
from hllm_control.proto import control_pb2
from hllm_control.qualification.native import NativeExecutor, NativeWorker
from hllm_control.qualification.sweep import Reference, SweepContent, freeze, placements, run
from hllm_control.serialization import sha256_file
from hllm_control.wire import deployment_plan_to_proto, model_manifest_to_proto

from tests.process_helpers import Workers, endpoint, plan, write_model

BINARY = os.environ.get("HLLM_CPU_WORKER")
MEMORY = os.environ.get("HLLM_MEMORY_PROFILER")
pytestmark = pytest.mark.skipif(
    not BINARY or not MEMORY, reason="native sweep tools not configured"
)


def test_native_identity_measured_hash_and_sweep(tmp_path: Path):
    assert BINARY and MEMORY
    root = tmp_path / "model"
    manifest = write_model(root)
    workload = WorkloadProfile(
        workload_id="smoke",
        prompt_tokens=5,
        output_tokens=1,
        total_cached_tokens=6,
        kv_dtype=DType.F32,
    )
    selected = plan(manifest).model_copy(
        update={
            "planning_mode": PlanningMode.MEASURED,
            "schema_version": "1.1",
            "workload_digest": digest(workload),
            "profile_bundle_digest": "a" * 64,
            "workload_id": workload.workload_id,
            "selected_candidate_id": "cpu-a--cpu-b-m001",
        }
    )
    d = digest(selected.model_dump(mode="json", exclude={"plan_id", "plan_digest"}))
    selected = selected.model_copy(update={"plan_digest": d, "plan_id": "plan-" + d[:16]})
    with Workers(root, binaries=(Path(BINARY), Path(BINARY))) as workers:
        wire = deployment_plan_to_proto(selected)
        request = control_pb2.LoadStageRequest(
            plan=wire,
            manifest=model_manifest_to_proto(manifest),
            stage_index=0,
            stage_endpoints=[
                control_pb2.StageEndpoint(
                    stage_index=i, worker_id=s.worker_id, endpoint=workers.endpoints[s.worker_id]
                )
                for i, s in enumerate(selected.stages)
            ],
        )
        accepted = workers.controls[0].LoadStage(request, timeout=10)
        assert accepted.accepted, accepted.detail
        workers.controls[0].UnloadStage(
            control_pb2.UnloadStageRequest(plan_id=selected.plan_id, deployment_version=1),
            timeout=5,
        )
        request.plan.workload_digest = "b" * 64
        rejected = workers.controls[0].LoadStage(request, timeout=10)
        assert not rejected.accepted
        assert "hash mismatch" in rejected.detail
    oracle = json.loads(
        (Path(__file__).parents[1] / "fixtures/qwen3/tiny-reference.json").read_text()
    )
    logits = oracle["logits"]["values"][-manifest.config.vocabulary_size :]
    reference = Reference(
        manifest_digest=manifest.manifest_digest,
        checkpoint_digest=checkpoint_digest(root, manifest),
        producer="Transformers fixture",
        producer_artifact_digest="b" * 64,
        prompt_ids=(1, 4, 2, 8, 3),
        generated_ids=(logits.index(max(logits)),),
    )
    executor = NativeExecutor(
        workers=tuple(
            NativeWorker(
                worker_id=s.worker_id,
                backend=Backend.CPU,
                endpoint=endpoint(),
                binary=str(Path(BINARY).resolve()),
                binary_digest=sha256_file(Path(BINARY)),
                memory_binary=str(Path(MEMORY).resolve()),
                memory_binary_digest=sha256_file(Path(MEMORY)),
                model_root=str(root),
                evidence_root=str(tmp_path / "raw"),
                capacity=MemoryAmounts(host=32 * 1024**2),
                python=sys.executable,
                host_headroom_bytes=0,
                device_headroom_bytes=0,
                extra_overhead_bytes=1024**2,
            )
            for s in selected.stages
        ),
        source_revision="test",
    )
    directory = tmp_path / "sweep"
    spec = freeze(
        SweepContent(
            manifest=manifest,
            workload=workload,
            reference=reference,
            selected_plan=selected,
            planning_report_digest="a" * 64,
            profile_bundle_digest="a" * 64,
            executor_digest=executor.identity(),
            concurrent_load="test suite",
        ),
        directory,
    )
    summary = run(spec, directory, executor, maximum_jobs=1)
    assert summary["decision"] == "incomplete"  # One smoke job cannot qualify a sweep.
    result = json.loads(next(directory.glob("*/result.json")).read_text())["result"]
    assert result["status"] == "measured", result["detail"]
    assert result["sample"]["native_arrivals_ms"]
    assert result["sample"]["native_request_setup_ms"] is not None
    assert len(placements(spec)) == 2
