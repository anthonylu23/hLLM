"""Hardware-required integration check for the CUDA backend foundation."""

from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path

import grpc
import pytest
from hllm_control.proto import common_pb2, control_pb2, control_pb2_grpc, profile_pb2

BINARY = Path(os.environ.get("HLLM_CUDA_WORKER", "build/native/cuda/cpp/hllm-worker-cuda"))


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


def test_cuda_probe_capabilities_memory_and_load_rejection(tmp_path: Path) -> None:
    command = arguments(tmp_path)
    command += [
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
                control = control_pb2_grpc.WorkerControlStub(channel)
                caps = control.GetCapabilities(common_pb2.Empty(), timeout=5).worker
                assert caps.backend == profile_pb2.BACKEND_CUDA
                assert caps.primary_memory_domain == profile_pb2.MEMORY_DOMAIN_DEVICE
                assert not caps.supported_architectures
                assert not caps.supported_execution_dtypes
                assert {b.domain: b.capacity_bytes for b in caps.memory_budgets} == {
                    profile_pb2.MEMORY_DOMAIN_HOST: 16777216,
                    profile_pb2.MEMORY_DOMAIN_DEVICE: 67108864,
                    profile_pb2.MEMORY_DOMAIN_HOST_PINNED: 1048576,
                }
                health = control.Health(common_pb2.Empty(), timeout=5)
                assert not health.serving
                assert "CUDA runtime ready on device 0" in health.detail
                assert "model execution is not implemented" in health.detail
                report = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
                assert len(report.domain_usage) == 3
                assert report.loaded_weight_bytes == report.active_requests == 0
                load = control_pb2.LoadStageRequest()
                load.plan.schema_version.major = load.manifest.schema_version.major = 1
                load.plan.plan_id = "probe"
                load.plan.plan_digest = "plan-digest"
                load.plan.deployment_version = 1
                load.plan.manifest_digest = load.manifest.manifest_digest = "manifest-digest"
                load.manifest.architecture.architecture_id = "qwen3.v1"
                load.plan.execution_dtype = common_pb2.DATA_TYPE_F32
                load.plan.activation_dtype = common_pb2.DATA_TYPE_F16
                result = control.LoadStage(load, timeout=5)
                assert not result.accepted
                assert result.error.code == common_pb2.ERROR_CODE_INCOMPATIBLE_WORKER
                assert "model execution is not implemented" in result.detail
        finally:
            process.terminate()
            try:
                _, stderr = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate(timeout=5)
            assert process.returncode in (0, -15), stderr.decode()


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
