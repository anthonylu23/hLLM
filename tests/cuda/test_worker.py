"""Hardware-required CUDA control and single-worker generation tests."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import DeploymentPlan, DType, PlanningMode, StageAssignment
from hllm_control.prepare.manifest import prepare_model
from hllm_control.profiling.models import digest
from hllm_control.proto import (
    common_pb2,
    control_pb2,
    control_pb2_grpc,
    execution_pb2,
    profile_pb2,
)

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("HLLM_CUDA_WORKER", ROOT / "build/native/cuda/cpp/hllm-worker-cuda"))


def arguments(root: Path) -> list[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    return [
        str(BINARY),
        "--listen",
        address,
        "--worker-id",
        "cuda-a",
        "--model-root",
        str(root),
        "--memory-limit-bytes",
        "16777216",
    ]


@pytest.fixture
def cuda_worker(tmp_path: Path) -> Iterator[tuple[str, control_pb2_grpc.WorkerControlStub]]:
    command = [
        *arguments(tmp_path),
        "--device-memory-limit-bytes",
        "67108864",
        "--pinned-host-memory-limit-bytes",
        "1048576",
        "--device-id",
        "0",
    ]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            with grpc.insecure_channel(command[2]) as channel:
                grpc.channel_ready_future(channel).result(timeout=30)
                yield command[2], control_pb2_grpc.WorkerControlStub(channel)
        finally:
            process.terminate()
            try:
                _, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate(timeout=5)
            assert process.returncode in (0, -15), stderr.decode()


def test_cuda_capabilities_memory_and_unsupported_model(
    cuda_worker: tuple[str, control_pb2_grpc.WorkerControlStub],
) -> None:
    _, control = cuda_worker
    caps = control.GetCapabilities(common_pb2.Empty(), timeout=5).worker
    assert caps.supports_mixed_precision
    assert caps.backend == profile_pb2.BACKEND_CUDA
    assert caps.primary_memory_domain == profile_pb2.MEMORY_DOMAIN_DEVICE
    assert set(caps.supported_architectures) == {"llama.v1", "qwen3.v1"}
    assert set(caps.supported_execution_dtypes) == {
        common_pb2.DATA_TYPE_F32,
        common_pb2.DATA_TYPE_F16,
    }
    assert {b.domain: b.capacity_bytes for b in caps.memory_budgets} == {
        profile_pb2.MEMORY_DOMAIN_HOST: 16777216,
        profile_pb2.MEMORY_DOMAIN_DEVICE: 67108864,
        profile_pb2.MEMORY_DOMAIN_HOST_PINNED: 1048576,
    }
    health = control.Health(common_pb2.Empty(), timeout=5)
    assert health.serving
    assert "CUDA runtime ready on device 0" in health.detail
    report = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
    assert len(report.domain_usage) == 3
    assert report.loaded_weight_bytes == report.active_requests == 0
    metrics = control.GetMetrics(common_pb2.Empty(), timeout=5)
    assert metrics.HasField("allocator")
    assert metrics.allocator.domain == profile_pb2.MEMORY_DOMAIN_DEVICE
    assert metrics.allocator.peak_bytes >= metrics.allocator.active_bytes
    load = control_pb2.LoadStageRequest()
    load.plan.schema_version.major = load.manifest.schema_version.major = 1
    load.plan.plan_id = "probe"
    load.plan.plan_digest = "plan-digest"
    load.plan.deployment_version = 1
    load.plan.manifest_digest = load.manifest.manifest_digest = "manifest-digest"
    load.manifest.architecture.architecture_id = "unsupported.v1"
    load.plan.execution_dtype = common_pb2.DATA_TYPE_F32
    load.plan.activation_dtype = common_pb2.DATA_TYPE_F16
    result = control.LoadStage(load, timeout=5)
    assert not result.accepted
    assert result.error.code == common_pb2.ERROR_CODE_INCOMPATIBLE_WORKER
    assert "unsupported architecture" in result.detail


def test_cuda_omits_metrics_for_unsupported_allocator(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "backend:cudaMallocAsync")
    _, control = request.getfixturevalue("cuda_worker")
    metrics = control.GetMetrics(common_pb2.Empty(), timeout=5)
    assert metrics.worker_id == "cuda-a"
    assert not metrics.HasField("allocator")


@pytest.mark.parametrize("dtype,mixed", [(DType.F32, False), (DType.F16, False), (DType.F32, True)])
def test_cuda_generation_reservations_and_reload(
    tmp_path: Path,
    cuda_worker: tuple[str, control_pb2_grpc.WorkerControlStub],
    dtype: DType,
    mixed: bool,
) -> None:
    address, control = cuda_worker
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/create_cpu_demo.py"), str(tmp_path)], check=True
    )
    manifest = prepare_model(tmp_path)
    plan = DeploymentPlan(
        plan_id="cuda-generation",
        plan_digest="cuda-test-plan",
        manifest_digest=manifest.manifest_digest,
        workload_id="tiny",
        planning_mode=PlanningMode.FEASIBILITY,
        execution_dtype=dtype,
        activation_dtype=DType.F16,
        split_layer=0,
        selected_candidate_id="test",
        stages=(
            StageAssignment(
                stage_index=0,
                worker_id="cuda-a",
                layer_start=0,
                layer_end=manifest.config.num_layers,
                owns_token_embedding=True,
                owns_final_norm=True,
                owns_lm_head=True,
                owns_sampling=True,
            ),
        ),
    )
    if mixed:
        plan = plan.model_copy(update={"schema_version": "1.2", "weight_dtype": DType.F16})
        d = digest(plan.model_dump(mode="json", exclude={"plan_id", "plan_digest"}))
        plan = plan.model_copy(update={"plan_id": "plan-" + d[:16], "plan_digest": d})
    previous = None
    for _ in range(2):
        with DeploymentSession(manifest, plan, {"cuda-a": address}) as session:
            reserved = control.ReserveRequest(
                control_pb2.ReserveRequestMessage(
                    plan_id=plan.plan_id,
                    deployment_version=plan.deployment_version,
                    request_id="reserved",
                    maximum_total_tokens=16,
                ),
                timeout=5,
            )
            assert reserved.accepted, reserved.detail
            report = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
            usage = {item.domain: item for item in report.domain_usage}
            assert usage[profile_pb2.MEMORY_DOMAIN_HOST].loaded_weight_bytes == 0
            assert usage[profile_pb2.MEMORY_DOMAIN_DEVICE].loaded_weight_bytes > 0
            assert usage[profile_pb2.MEMORY_DOMAIN_DEVICE].reserved_cache_bytes == (
                16
                * manifest.config.num_layers
                * manifest.config.num_kv_heads
                * manifest.config.head_dim
                * 2
                * (4 if dtype == DType.F32 else 2)
            )
            assert usage[profile_pb2.MEMORY_DOMAIN_HOST_PINNED].reserved_workspace_bytes == 0
            control.CancelRequest(
                control_pb2.CancelRequestMessage(
                    plan_id=plan.plan_id,
                    deployment_version=plan.deployment_version,
                    request_id="reserved",
                ),
                timeout=5,
            )
            for _ in range(2):
                events = list(session.generate([1, 4, 2], maximum_new_tokens=16, stop_token_ids=[]))
                assert events[0].HasField("prefill_complete")
                assert events[-1].terminal.state == execution_pb2.TERMINAL_STATE_COMPLETED
                result = [event.token.token_id for event in events if event.HasField("token")]
                assert len(result) == 16
                if previous is not None:
                    assert result == previous
                previous = result
                report = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
                assert report.active_requests == report.reserved_cache_bytes == 0
                assert report.reserved_workspace_bytes == 0
        assert control.GetMemoryReport(common_pb2.Empty(), timeout=5).loaded_weight_bytes == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ([], "requires --device-memory-limit-bytes"),
        (
            ["--device-memory-limit-bytes", "67108864", "--device-id", "999999"],
            "device ID is out of range",
        ),
    ],
)
def test_cuda_rejects_invalid_startup(tmp_path: Path, extra: list[str], message: str) -> None:
    result = subprocess.run(arguments(tmp_path) + extra, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert message in result.stderr
