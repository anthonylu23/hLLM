"""Chunk acknowledgments preserve token positions, sampling state and cleanup."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import common_pb2, control_pb2, execution_pb2, execution_pb2_grpc
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.wire import deployment_plan_to_proto, model_manifest_to_proto

from tests.process_helpers import Workers, plan, wait_clean, write_model


@pytest.mark.parametrize("split", [None, 1])
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_chunked_prefill_matches_greedy_and_seeded_reference(
    tmp_path: Path,
    split: int | None,
    chunk: int,
) -> None:
    manifest = write_model(tmp_path)
    arguments = ("--max-active-requests", "4")
    with Workers(tmp_path, extra_args=(arguments, arguments)) as workers:

        async def run() -> None:
            expected = []
            prompt = [1, 4, 2, 5, 3, 1, 4]
            for chunk_size in [0, chunk]:
                runtime = ServingRuntime(
                    DeploymentSession(manifest, plan(manifest, split=split), workers.endpoints),
                    4,
                    prefill_chunk_tokens=chunk_size,
                )
                await runtime.start()
                try:

                    async def request(
                        identifier: str, sampled: bool, runtime: ServingRuntime = runtime
                    ) -> list[int]:
                        sampling = (
                            execution_pb2.SamplingOptions(
                                temperature=1,
                                top_p=0.95,
                                top_k=8,
                                seed=42,
                            )
                            if sampled
                            else None
                        )
                        return [
                            token
                            async for token in runtime.generate(
                                identifier,
                                prompt,
                                32,
                                [],
                                30,
                                sampling,
                            )
                        ]

                    actual = await asyncio.gather(request("greedy", False), request("seeded", True))
                    if not chunk_size:
                        expected = actual
                    else:
                        assert actual == expected
                    wait_clean(workers)
                finally:
                    await runtime.close()

        asyncio.run(run())
        wait_clean(workers, loaded=False)


def _generate(channel: grpc.Channel, plan_id: str, request_id: str, chunk: int) -> list[int]:
    events = list(
        execution_pb2_grpc.GenerationStub(channel).Generate(
            execution_pb2.GenerationRequest(
                deployment_id=plan_id,
                deployment_version=1,
                request_id=request_id,
                token_ids=[1, 4, 2, 8, 3],
                maximum_new_tokens=4,
                prefill_chunk_tokens=chunk,
            ),
            timeout=10,
        )
    )
    assert events[-1].terminal.state == execution_pb2.TERMINAL_STATE_COMPLETED
    return [event.token.token_id for event in events if event.HasField("token")]


def test_downstream_open_carries_the_request_chunk(tmp_path: Path) -> None:
    # A test-only relay observes the stage stream; production never relays activations.
    manifest = write_model(tmp_path)
    opens: list[execution_pb2.SequenceOpen] = []
    prefill_rows: list[int] = []
    with Workers(tmp_path) as workers:
        downstream = execution_pb2_grpc.StageExecutionStub(workers.channels[1])

        class RecordingRelay(execution_pb2_grpc.StageExecutionServicer):
            def Execute(self, request_iterator, context):
                def forward():
                    for message in request_iterator:
                        if message.HasField("open_sequence"):
                            saved = execution_pb2.SequenceOpen()
                            saved.CopyFrom(message.open_sequence)
                            opens.append(saved)
                        elif (
                            message.HasField("tensor")
                            and message.tensor.phase == execution_pb2.EXECUTION_PHASE_PREFILL
                        ):
                            prefill_rows.append(message.tensor.shape[1])
                        yield message

                yield from downstream.Execute(forward(), timeout=10)

        with ThreadPoolExecutor(max_workers=2) as executor:
            server = grpc.server(executor)
            execution_pb2_grpc.add_StageExecutionServicer_to_server(RecordingRelay(), server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                wire_plan = plan(manifest)
                endpoints = [
                    control_pb2.StageEndpoint(
                        stage_index=0, worker_id="cpu-a", endpoint=workers.endpoints["cpu-a"]
                    ),
                    control_pb2.StageEndpoint(
                        stage_index=1, worker_id="cpu-b", endpoint=f"127.0.0.1:{port}"
                    ),
                ]
                for index, control in enumerate(workers.controls):
                    response = control.LoadStage(
                        control_pb2.LoadStageRequest(
                            manifest=model_manifest_to_proto(manifest),
                            plan=deployment_plan_to_proto(wire_plan),
                            stage_index=index,
                            stage_endpoints=endpoints,
                        ),
                        timeout=5,
                    )
                    assert response.accepted, response.detail
                whole = _generate(workers.channels[0], wire_plan.plan_id, "whole", 0)
                assert [o.prefill_chunk_tokens for o in opens] == [0]
                assert prefill_rows == [5]
                opens.clear()
                prefill_rows.clear()
                chunked = _generate(workers.channels[0], wire_plan.plan_id, "chunked", 2)
                assert chunked == whole
                assert [o.prefill_chunk_tokens for o in opens] == [2]
                assert prefill_rows == [2, 2, 1]
                wait_clean(workers)
            finally:
                server.stop(0).wait()


@pytest.mark.parametrize("chunk", [2, 3])
def test_downstream_rejects_prefill_input_above_its_reserved_chunk(
    tmp_path: Path, chunk: int
) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            request_id = f"chunk-{chunk}"
            messages = [
                execution_pb2.StageMessage(
                    open_sequence=execution_pb2.SequenceOpen(
                        protocol_version=1,
                        deployment_id=session.plan.plan_id,
                        deployment_version=1,
                        request_id=request_id,
                        maximum_total_tokens=4,
                        maximum_new_tokens=1,
                        prompt_tokens=3,
                        prefill_chunk_tokens=chunk,
                        deadline_unix_ms=int((time.time() + 3) * 1000),
                    )
                ),
                execution_pb2.StageMessage(
                    tensor=execution_pb2.TensorEnvelope(
                        protocol_version=1,
                        deployment_id=session.plan.plan_id,
                        deployment_version=1,
                        request_id=request_id,
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
                ),
            ]
            call = execution_pb2_grpc.StageExecutionStub(workers.channels[1]).Execute(
                iter(messages), timeout=4
            )
            if chunk == 2:
                with pytest.raises(grpc.RpcError) as error:
                    list(call)
                assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
                assert "reserved chunk" in (error.value.details() or "")
            else:
                # Within the reservation the stage samples; the stream then ends unterminated.
                reply = next(call)
                assert reply.HasField("sampled_token")
                call.cancel()
            wait_clean(workers)
