"""Optional OS counters retain presence through the HTTP metrics adapter."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from hllm_control.controller import DeploymentSession
from hllm_control.proto import control_pb2
from hllm_control.serving.runtime import ServingRuntime


def test_process_metrics_distinguish_unavailable_from_zero() -> None:
    runtime = ServingRuntime(MagicMock(spec=DeploymentSession), 1)
    worker = control_pb2.WorkerMetrics()
    control = MagicMock()
    control.GetMemoryReport = AsyncMock(return_value=control_pb2.MemoryReport())
    control.GetMetrics = AsyncMock(return_value=worker)
    runtime.controls = [control]
    # Old workers do not expose the new message at all.
    assert "hllm_worker_process_" not in asyncio.run(runtime.metrics())
    worker.process_memory.rss_bytes = 123
    worker.process_memory.physical_footprint_bytes = 0
    worker.allocator.active_bytes = 45
    result = asyncio.run(runtime.metrics())
    assert 'hllm_worker_process_rss_bytes{stage="0"} 123\n' in result
    assert 'hllm_worker_process_physical_footprint_bytes{stage="0"} 0\n' in result
    assert "hllm_worker_process_physical_footprint_lifetime_peak_bytes" not in result
    assert 'hllm_worker_allocator_active_bytes{stage="0"} 45\n' in result
