"""Failure contracts shared by CPU and mixed-worker process suites."""

from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import common_pb2, execution_pb2, execution_pb2_grpc

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
