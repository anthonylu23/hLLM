"""Failure contracts shared by CPU and mixed-worker process suites."""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import common_pb2, control_pb2, execution_pb2, execution_pb2_grpc

from tests.process_helpers import Workers, plan, tokens, wait_clean, write_model


class WorkerFactory(Protocol):
    def __call__(self, root: Path, limit: int | tuple[int, int] = ...) -> Workers: ...


def check_stage_protocol(tmp_path: Path, worker_factory: WorkerFactory) -> None:
    manifest = write_model(tmp_path)
    with worker_factory(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            for mutation in ("sequence", "version", "shape", "phase", "payload", "open_only"):
                opening = execution_pb2.StageMessage(
                    open_sequence=execution_pb2.SequenceOpen(
                        protocol_version=1,
                        deployment_id=session.plan.plan_id,
                        deployment_version=1,
                        request_id=mutation,
                        maximum_total_tokens=4,
                        maximum_new_tokens=1,
                        deadline_unix_ms=int((time.time() + 3) * 1000),
                    )
                )
                tensor = execution_pb2.TensorEnvelope(
                    protocol_version=1,
                    deployment_id=session.plan.plan_id,
                    deployment_version=1,
                    request_id=mutation,
                    sequence_number=0,
                    phase=execution_pb2.EXECUTION_PHASE_PREFILL,
                    first_position=0,
                    sequence_lengths=[3],
                    cache_slot_ids=[0],
                    shape=[1, 3, 6],
                    dtype=common_pb2.DATA_TYPE_F16,
                    layout="dense_row_major_le",
                    payload_length=36,
                    payload=bytes(36),
                )
                if mutation == "sequence":
                    tensor.sequence_number = 1
                elif mutation == "version":
                    tensor.deployment_version = 2
                elif mutation == "shape":
                    tensor.shape[2] = 9
                elif mutation == "phase":
                    tensor.phase = execution_pb2.EXECUTION_PHASE_DECODE
                elif mutation == "payload":
                    tensor.payload = b"bad"
                messages = (
                    [opening]
                    if mutation == "open_only"
                    else [opening, execution_pb2.StageMessage(tensor=tensor)]
                )
                call = execution_pb2_grpc.StageExecutionStub(workers.channels[1]).Execute(
                    iter(messages), timeout=4
                )
                with pytest.raises(grpc.RpcError) as error:
                    list(call)
                assert error.value.code() == (
                    grpc.StatusCode.INTERNAL
                    if mutation == "open_only"
                    else grpc.StatusCode.INVALID_ARGUMENT
                )
                wait_clean(workers)
            assert len(tokens(session, count=1)) == 1


def check_idle_cancellation(tmp_path: Path, worker_factory: WorkerFactory) -> None:
    manifest = write_model(tmp_path)
    with worker_factory(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            for cancel in (False, True):
                pending: queue.Queue[execution_pb2.StageMessage | None] = queue.Queue()
                pending.put(
                    execution_pb2.StageMessage(
                        open_sequence=execution_pb2.SequenceOpen(
                            protocol_version=1,
                            deployment_id=session.plan.plan_id,
                            deployment_version=1,
                            request_id="idle",
                            maximum_total_tokens=4,
                            maximum_new_tokens=1,
                            deadline_unix_ms=int((time.time() + (5 if cancel else 0.2)) * 1000),
                        )
                    )
                )

                def messages(
                    source: queue.Queue[execution_pb2.StageMessage | None] = pending,
                ) -> Iterator[execution_pb2.StageMessage]:
                    while (message := source.get()) is not None:
                        yield message

                call = execution_pb2_grpc.StageExecutionStub(workers.channels[1]).Execute(
                    messages(), timeout=6
                )
                if cancel:
                    until = time.monotonic() + 3
                    while (
                        workers.controls[1].GetMemoryReport(common_pb2.Empty()).active_requests == 0
                    ):
                        assert time.monotonic() < until
                        time.sleep(0.01)
                    session.cancel("idle")
                try:
                    with pytest.raises(grpc.RpcError):
                        list(call)
                finally:
                    pending.put(None)
                    call.cancel()
                wait_clean(workers)
            assert len(tokens(session, count=1)) == 1


def check_partial_load_rollback(tmp_path: Path, worker_factory: WorkerFactory) -> None:
    manifest = write_model(tmp_path)
    manifest = manifest.model_copy(
        update={
            "tensors": tuple(
                tensor
                for tensor in manifest.tensors
                if tensor.name != "model.layers.0.input_layernorm.weight"
            )
        }
    )
    with worker_factory(tmp_path) as workers:
        with pytest.raises(RuntimeError, match="missing required tensor"):
            with DeploymentSession(manifest, plan(manifest), workers.endpoints):
                pass
        wait_clean(workers, loaded=False)


def check_admission_rollback(tmp_path: Path, worker_factory: WorkerFactory) -> None:
    manifest = write_model(tmp_path)
    with worker_factory(tmp_path, limit=(16_000_000, 100_000)) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            with pytest.raises(grpc.RpcError) as error:
                tokens(session, count=256)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            wait_clean(workers)
            assert len(tokens(session, count=1)) == 1


def check_boundary_fault(
    tmp_path: Path, worker_factory: WorkerFactory, phase: int, fault: str
) -> None:
    manifest = write_model(tmp_path)
    reached, release = threading.Event(), threading.Event()
    events = []
    with worker_factory(tmp_path) as workers:
        peer = execution_pb2_grpc.StageExecutionStub(workers.channels[1])

        class Relay(execution_pb2_grpc.StageExecutionServicer):
            def Execute(self, request_iterator, context):
                def forward():
                    for message in request_iterator:
                        if message.HasField("tensor") and message.tensor.phase == phase:
                            reached.set()
                            while not release.wait(0.01):
                                if not context.is_active():
                                    return
                            if fault == "disconnect":
                                return
                        yield message

                call = peer.Execute(forward(), timeout=context.time_remaining())
                try:
                    yield from call
                except grpc.RpcError as error:
                    context.abort(error.code(), error.details())
                finally:
                    call.cancel()

        with ThreadPoolExecutor(max_workers=4) as executor:
            server = grpc.server(executor)
            execution_pb2_grpc.add_StageExecutionServicer_to_server(Relay(), server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            session = DeploymentSession(manifest, plan(manifest), workers.endpoints)
            # Control channels still point directly at native workers; only native
            # stage-to-stage execution is routed through the test relay.
            session.endpoints["cpu-b"] = f"127.0.0.1:{port}"
            try:
                with session:
                    timeout = 3 if fault == "deadline" else 15
                    call = execution_pb2_grpc.GenerationStub(workers.channels[0]).Generate(
                        execution_pb2.GenerationRequest(
                            deployment_id=session.plan.plan_id,
                            deployment_version=1,
                            request_id="barrier",
                            token_ids=[1, 4, 2],
                            maximum_new_tokens=8,
                            deadline_unix_ms=int((time.time() + timeout) * 1000),
                        ),
                        # Keep the client alive beyond the application deadline so
                        # the assertion observes the native server's status.
                        timeout=timeout + 5,
                    )

                    def consume():
                        try:
                            for event in call:
                                events.append(event)
                        except grpc.RpcError as error:
                            return error.code()
                        return grpc.StatusCode.OK

                    done = executor.submit(consume)
                    assert reached.wait(5), "request never reached the selected phase"
                    until = time.monotonic() + 3
                    while not all(r.active_requests == 1 for r in session.memory_reports()):
                        assert time.monotonic() < until
                        time.sleep(0.01)
                    # Unload must reject active work in both worker positions.
                    for control in workers.controls:
                        with pytest.raises(grpc.RpcError) as error:
                            control.UnloadStage(
                                control_pb2.UnloadStageRequest(
                                    plan_id=session.plan.plan_id, deployment_version=1
                                ),
                                timeout=2,
                            )
                        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
                    if fault == "client":
                        call.cancel()
                    elif fault == "control":
                        session.cancel("barrier")
                    elif fault == "disconnect":
                        release.set()
                    elif fault.startswith("kill"):
                        index = 0 if fault == "kill_first" else 1
                        workers.processes[index].kill()
                        workers.processes[index].wait(timeout=5)
                    status = done.result(timeout=20)
                    expected = {
                        "client": grpc.StatusCode.CANCELLED,
                        "control": grpc.StatusCode.CANCELLED,
                        "deadline": grpc.StatusCode.DEADLINE_EXCEEDED,
                        "disconnect": grpc.StatusCode.INTERNAL,
                        "kill_first": grpc.StatusCode.UNAVAILABLE,
                        "kill_final": grpc.StatusCode.UNAVAILABLE,
                    }
                    assert status == expected[fault]
                    assert not any(e.HasField("terminal") for e in events)
                    positions = [e.token.token_position for e in events if e.HasField("token")]
                    # Cancellation may discard a token already queued by the
                    # server, but cannot replay one or publish beyond the barrier.
                    assert positions == list(range(3, 3 + len(positions)))
                    assert len(positions) <= int(phase == execution_pb2.EXECUTION_PHASE_DECODE)
                    release.set()
                    if fault.startswith("kill"):
                        survivor = 1 if fault == "kill_first" else 0
                        until = time.monotonic() + 6
                        while True:
                            report = workers.controls[survivor].GetMemoryReport(
                                common_pb2.Empty(), timeout=2
                            )
                            if (
                                report.active_requests
                                == report.reserved_cache_bytes
                                == report.reserved_workspace_bytes
                                == 0
                            ):
                                break
                            assert time.monotonic() < until, report
                            time.sleep(0.01)
                    else:
                        wait_clean(workers)
                if not fault.startswith("kill"):
                    wait_clean(workers, loaded=False)
                    with DeploymentSession(manifest, plan(manifest), workers.endpoints) as fresh:
                        assert len(tokens(fresh, count=2)) == 2
                    wait_clean(workers, loaded=False)
            finally:
                release.set()
                server.stop(0).wait()
