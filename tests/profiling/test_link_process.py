"""Two native processes, real RPC framing, failure recovery and directional identity."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import grpc
import pytest
from hllm_control.profiling.link import LinkArtifact, run_link_profile, server_config
from hllm_control.proto import (
    common_pb2,
    control_pb2,
    control_pb2_grpc,
    execution_pb2,
    execution_pb2_grpc,
)

from tests.process_helpers import endpoint

BINARY = os.environ.get("HLLM_LINK_PROFILER")
pytestmark = pytest.mark.skipif(BINARY is None, reason="native link profiler not configured")


@pytest.fixture
def probes(tmp_path):
    assert BINARY is not None
    addresses = [endpoint(), endpoint()]
    processes = []
    try:
        for i in range(2):
            config = tmp_path / f"server-{i}.json"
            server_config(
                binary=Path(BINARY),
                worker_id=f"probe-{i}",
                listen=addresses[i],
                peers={f"probe-{1 - i}": addresses[1 - i], "absent": endpoint()},
                source_revision="test",
                output=config,
            )
            processes.append(subprocess.Popen([BINARY, str(config)], stdout=subprocess.DEVNULL))
        for address in addresses:
            with grpc.insecure_channel(address) as channel:
                grpc.channel_ready_future(channel).result(timeout=10)
        yield addresses
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            process.wait(timeout=10)


@pytest.mark.parametrize("direction", [0, 1])
def test_payload_exchange_and_artifact(probes, tmp_path: Path, direction: int):
    a = run_link_profile(
        source_endpoint=probes[direction],
        target_endpoint=probes[1 - direction],
        target_worker_id=f"probe-{1 - direction}",
        prompt_tokens=512,
        output_tokens=4,
        hidden_size=64,
        warmup_cycles=1,
        measured_cycles=2,
        output=tmp_path / "link.json",
        concurrent_load="tests",
    )
    assert a.qualified, a.error
    assert a.result is not None
    assert len(a.result.samples) == 12
    assert len(a.result.streams) == 3
    assert a.result.source.worker_id == f"probe-{direction}"
    assert a.result.samples[0].payload_bytes == 65536
    assert a.result.samples[1].payload_bytes == 128
    assert LinkArtifact.model_validate_json((tmp_path / "link.json").read_text()) == a


def test_limits_deadline_and_recovery(probes):
    with grpc.insecure_channel(probes[0]) as channel:
        stub = control_pb2_grpc.WorkerControlStub(channel)
        valid: dict[str, Any] = dict(
            target_worker_id="probe-1",
            prompt_tokens=8,
            hidden_size=16,
            output_tokens=3,
            warmup_cycles=0,
            measured_cycles=1,
            timeout_ms=2000,
        )
        for update in (
            {"target_worker_id": "unlisted"},
            {"hidden_size": 8193},
            {"prompt_tokens": 4096, "hidden_size": 8192},
            {"measured_cycles": 33},
        ):
            with pytest.raises(grpc.RpcError) as failure:
                stub.QualifyLink(
                    control_pb2.LinkQualificationRequest(**cast(Any, valid | update)), timeout=3
                )
            assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        start = time.monotonic()
        with pytest.raises(grpc.RpcError):
            stub.QualifyLink(
                control_pb2.LinkQualificationRequest(
                    **cast(Any, valid | {"target_worker_id": "absent"})
                ),
                timeout=0.05,
            )
        assert time.monotonic() - start < 1
        time.sleep(0.15)  # Give the bounded native connection loop time to see cancellation.
        assert stub.QualifyLink(control_pb2.LinkQualificationRequest(**valid), timeout=3).HasField(
            "qualification"
        )


def test_stream_cancellation_and_bad_payload(probes):
    with grpc.insecure_channel(probes[1]) as channel:
        stub = execution_pb2_grpc.StageExecutionStub(channel)
        opened = execution_pb2.StageMessage(
            open_sequence=execution_pb2.SequenceOpen(
                protocol_version=1,
                deployment_id="hllm-link-probe-v1",
                deployment_version=1,
                request_id="bad",
                maximum_total_tokens=12,
                maximum_new_tokens=1,
            )
        )
        bad = execution_pb2.StageMessage(
            tensor=execution_pb2.TensorEnvelope(
                protocol_version=1,
                deployment_id="hllm-link-probe-v1",
                deployment_version=1,
                request_id="bad",
                sequence_number=0,
                phase=execution_pb2.EXECUTION_PHASE_PREFILL,
                sequence_lengths=[8],
                cache_slot_ids=[0],
                shape=[1, 8, 16],
                dtype=common_pb2.DATA_TYPE_F16,
                layout="dense_row_major_le",
                payload_length=256,
                payload=b"short",
            )
        )
        with pytest.raises(grpc.RpcError) as failure:
            list(stub.Execute(iter([opened, bad]), timeout=2))
        assert failure.value.code() == grpc.StatusCode.INVALID_ARGUMENT

        # A stalled reader is bounded by the RPC deadline and releases the busy slot.
        def stalled():
            yield opened
            time.sleep(0.3)

        with pytest.raises(grpc.RpcError):
            list(stub.Execute(stalled(), timeout=0.05))
