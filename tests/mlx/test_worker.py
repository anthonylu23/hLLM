"""MLX-only RPC execution, capabilities, allocator telemetry and reloads."""

from pathlib import Path

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import DType
from hllm_control.proto import common_pb2, control_pb2, profile_pb2

from tests.mlx.helpers import MLX_BINARY
from tests.process_helpers import Workers, plan, tokens, wait_clean, write_model


@pytest.mark.parametrize("dtype", [DType.F32, DType.F16])
@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_mlx_only_generation_and_metrics(tmp_path: Path, dtype: DType, family: str) -> None:
    manifest = write_model(tmp_path, family)
    with Workers(tmp_path) as reference:
        with DeploymentSession(
            manifest, plan(manifest, split=None), reference.endpoints
        ) as session:
            expected = tokens(session)
    with Workers(tmp_path, limit=128 * 1024 * 1024, binaries=(MLX_BINARY, MLX_BINARY)) as workers:
        control = workers.controls[0]
        caps = control.GetCapabilities(common_pb2.Empty(), timeout=5).worker
        assert caps.backend == profile_pb2.BACKEND_MLX
        assert caps.primary_memory_domain == profile_pb2.MEMORY_DOMAIN_UNIFIED
        assert set(caps.supported_architectures) == {"llama.v1", "qwen3.v1"}
        assert set(caps.supported_execution_dtypes) == {
            common_pb2.DATA_TYPE_F32,
            common_pb2.DATA_TYPE_F16,
        }
        assert {b.domain: b.capacity_bytes for b in caps.memory_budgets} == {
            profile_pb2.MEMORY_DOMAIN_UNIFIED: 128 * 1024 * 1024,
        }
        assert control.Health(common_pb2.Empty(), timeout=5).serving
        configured = plan(manifest, split=None).model_copy(update={"execution_dtype": dtype})
        samples = []
        for cycle in range(12):
            with DeploymentSession(manifest, configured, workers.endpoints) as session:
                reserved = control.ReserveRequest(
                    control_pb2.ReserveRequestMessage(
                        plan_id=configured.plan_id,
                        deployment_version=1,
                        request_id="reserved",
                        maximum_total_tokens=16,
                    ),
                    timeout=5,
                )
                assert reserved.accepted, reserved.detail
                report = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
                assert len(report.domain_usage) == 1
                usage = report.domain_usage[0]
                assert usage.domain == profile_pb2.MEMORY_DOMAIN_UNIFIED
                assert usage.loaded_weight_bytes == report.loaded_weight_bytes > 0
                assert (
                    usage.reserved_cache_bytes
                    == report.reserved_cache_bytes
                    == (
                        16
                        * manifest.config.num_layers
                        * manifest.config.num_kv_heads
                        * manifest.config.head_dim
                        * 2
                        * (4 if dtype == DType.F32 else 2)
                    )
                )
                assert usage.reserved_workspace_bytes == report.reserved_workspace_bytes > 0
                control.CancelRequest(
                    control_pb2.CancelRequestMessage(
                        plan_id=configured.plan_id,
                        deployment_version=1,
                        request_id="reserved",
                    ),
                    timeout=5,
                )
                assert tokens(session) == expected
                wait_clean(workers)
            wait_clean(workers, loaded=False)
            metrics = control.GetMetrics(common_pb2.Empty(), timeout=5)
            assert metrics.HasField("allocator")
            assert metrics.allocator.domain == profile_pb2.MEMORY_DOMAIN_UNIFIED
            assert metrics.allocator.peak_bytes >= metrics.allocator.active_bytes
            if cycle >= 3:
                samples.append((metrics.allocator.active_bytes, metrics.allocator.cached_bytes))
        assert max(s[0] for s in samples) == min(s[0] for s in samples)
        assert max(s[1] for s in samples) - min(s[1] for s in samples) <= 1024 * 1024
        print({"family": family, "dtype": dtype.value, "allocator_after_unload": samples[-1]})


def test_rejects_cuda_flags_and_zero_budget(tmp_path: Path) -> None:
    import subprocess

    base = [
        str(MLX_BINARY),
        "--listen",
        "127.0.0.1:0",
        "--worker-id",
        "mlx",
        "--model-root",
        str(tmp_path),
    ]
    for extra, message in (
        (["--memory-limit-bytes", "0"], "positive integer"),
        (["--memory-limit-bytes", "1048576", "--device-id", "0"], "unknown argument"),
    ):
        result = subprocess.run(base + extra, capture_output=True, text=True, timeout=10)
        assert result.returncode == 2
        assert message in result.stderr
