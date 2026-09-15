"""Two native processes, real RPC framing, failure recovery and directional identity."""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
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
    profile_pb2,
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
        # The application limit expires before the outer RPC deadline, so the
        # returned code must come from the native connection loop.
        with pytest.raises(grpc.RpcError) as failure:
            stub.QualifyLink(
                control_pb2.LinkQualificationRequest(
                    **cast(Any, valid | {"target_worker_id": "absent", "timeout_ms": 100})
                ),
                timeout=3,
            )
        assert failure.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
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


def test_probe_contention_preserves_resource_status_and_recovers(probes):
    release = Event()
    opened = execution_pb2.StageMessage(
        open_sequence=execution_pb2.SequenceOpen(
            protocol_version=1,
            deployment_id="hllm-link-probe-v1",
            deployment_version=1,
            request_id="held",
            maximum_total_tokens=12,
            maximum_new_tokens=1,
        )
    )

    def hold():
        yield opened
        release.wait(timeout=10)

    with grpc.insecure_channel(probes[1]) as target, grpc.insecure_channel(probes[0]) as source:
        execution = execution_pb2_grpc.StageExecutionStub(target)
        held = execution.Execute(hold(), timeout=5)
        try:
            held.initial_metadata()  # Target acquired Busy and accepted the opening.
            for channel, peer in ((target, "probe-0"), (source, "probe-1")):
                with pytest.raises(grpc.RpcError) as failure:
                    control_pb2_grpc.WorkerControlStub(channel).QualifyLink(
                        control_pb2.LinkQualificationRequest(
                            target_worker_id=peer,
                            prompt_tokens=8,
                            hidden_size=16,
                            output_tokens=3,
                            measured_cycles=1,
                            timeout_ms=2000,
                        ),
                        timeout=3,
                    )
                assert failure.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
                assert "probe already active" in (failure.value.details() or "")
            with pytest.raises(grpc.RpcError) as failure:
                list(execution.Execute(iter([opened]), timeout=2))
            assert failure.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        finally:
            held.cancel()
            release.set()

        until = time.monotonic() + 3
        while True:
            with pytest.raises(grpc.RpcError) as failure:
                # Clean half-close without termination is data loss, not bad input.
                list(execution.Execute(iter([opened]), timeout=2))
            if failure.value.code() == grpc.StatusCode.DATA_LOSS:
                break
            assert failure.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            assert time.monotonic() < until
            time.sleep(0.01)


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("info", grpc.StatusCode.PERMISSION_DENIED),
        ("transport", grpc.StatusCode.UNAVAILABLE),
        ("forbidden", grpc.StatusCode.PERMISSION_DENIED),
        ("truncated", grpc.StatusCode.DATA_LOSS),
        ("malformed", grpc.StatusCode.DATA_LOSS),
        ("missing-ack", grpc.StatusCode.DATA_LOSS),
        ("bad-ack", grpc.StatusCode.DATA_LOSS),
        ("timeout", grpc.StatusCode.DEADLINE_EXCEEDED),
        ("cancel", grpc.StatusCode.CANCELLED),
    ],
)
def test_peer_failures_retain_diagnostics_and_release_source(tmp_path, fault, expected):
    assert BINARY is not None
    entered = Event()
    active_fault = [fault]

    class Peer(control_pb2_grpc.WorkerControlServicer, execution_pb2_grpc.StageExecutionServicer):
        def GetLinkProbeInfo(self, request, context):
            if active_fault[0] == "info":
                context.abort(grpc.StatusCode.PERMISSION_DENIED, "probe identity denied")
            return profile_pb2.LinkProbeIdentity(worker_id="peer")

        def Execute(self, request_iterator, context):
            next(request_iterator)
            context.send_initial_metadata(())
            for message in request_iterator:
                if message.HasField("terminate"):
                    if active_fault[0] == "missing-ack":
                        return
                    if active_fault[0] == "bad-ack":
                        message.terminate.request_id = "wrong"
                    yield message
                    return
                entered.set()
                mode = active_fault[0]
                if mode in ("transport", "forbidden"):
                    context.abort(
                        grpc.StatusCode.UNAVAILABLE
                        if mode == "transport"
                        else grpc.StatusCode.PERMISSION_DENIED,
                        "injected peer failure",
                    )
                if mode == "truncated":
                    return
                if mode in ("timeout", "cancel"):
                    cancelled = Event()
                    context.add_callback(cancelled.set)
                    cancelled.wait(timeout=5)
                    return
                tensor = message.tensor
                yield execution_pb2.StageMessage(
                    sampled_token=execution_pb2.SampledToken(
                        deployment_id=tensor.deployment_id,
                        deployment_version=tensor.deployment_version,
                        request_id=tensor.request_id,
                        sequence_number=tensor.sequence_number,
                        token_position=tensor.first_position + tensor.shape[1],
                        token_id=99 if mode == "malformed" else 0,
                    )
                )

    with ThreadPoolExecutor(max_workers=2) as pool:
        server = grpc.server(pool)
        peer = Peer()
        control_pb2_grpc.add_WorkerControlServicer_to_server(peer, server)
        execution_pb2_grpc.add_StageExecutionServicer_to_server(peer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        source = endpoint()
        config = tmp_path / "source.json"
        process = None
        try:
            server_config(
                binary=Path(BINARY),
                worker_id="source",
                listen=source,
                peers={"peer": f"127.0.0.1:{port}"},
                source_revision="test",
                output=config,
            )
            process = subprocess.Popen([BINARY, str(config)], stdout=subprocess.DEVNULL)
            with grpc.insecure_channel(source) as channel:
                grpc.channel_ready_future(channel).result(timeout=10)
                stub = control_pb2_grpc.WorkerControlStub(channel)
                request = control_pb2.LinkQualificationRequest(
                    target_worker_id="peer",
                    prompt_tokens=8,
                    hidden_size=16,
                    output_tokens=3,
                    measured_cycles=1,
                    timeout_ms=500,
                )
                if fault == "cancel":
                    request.timeout_ms = 2000
                    call = stub.QualifyLink.future(request, timeout=3)
                    assert entered.wait(timeout=2)
                    assert call.cancel()
                    assert call.code() == expected
                else:
                    with pytest.raises(grpc.RpcError) as failure:
                        stub.QualifyLink(request, timeout=3)
                    assert failure.value.code() == expected, failure.value.details()
                    if fault in ("transport", "forbidden"):
                        assert "injected peer failure" in (failure.value.details() or "")
                if fault != "info":
                    assert entered.is_set(), "failure must occur during the activation exchange"
                active_fault[0] = "healthy"
                request.timeout_ms = 2000
                until = time.monotonic() + 3
                while True:
                    try:
                        assert stub.QualifyLink(request, timeout=3).HasField("qualification")
                        break
                    except grpc.RpcError as error:
                        assert (
                            fault == "cancel" and error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
                        )
                        assert time.monotonic() < until
                        time.sleep(0.01)
        finally:
            if process is not None:
                process.terminate()
                process.wait(timeout=5)
            server.stop(0).wait(timeout=5)
