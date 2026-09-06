"""Real process/transport tests. Build hllm-worker-cpu before running this suite."""

from __future__ import annotations

import json
import queue
import struct
import time
from collections.abc import Iterator
from pathlib import Path

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import (
    common_pb2,
    control_pb2,
    execution_pb2,
    execution_pb2_grpc,
)

from tests.process_helpers import ROOT, Workers, plan, tokens, wait_clean, write_model


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_two_process_generation_all_splits_and_both_worker_orders(
    tmp_path: Path, family: str
) -> None:
    manifest = write_model(tmp_path, family)
    with Workers(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest, None), workers.endpoints) as session:
            expected = tokens(session)
            if family == "qwen3":
                oracle = json.loads((ROOT / "tests/fixtures/qwen3/tiny-reference.json").read_text())
                logits = oracle["logits"]["values"][22:33]
                assert expected[0] == max(range(11), key=logits.__getitem__)
        for split in range(1, manifest.config.num_layers):
            for reverse in (False, True):
                with DeploymentSession(
                    manifest, plan(manifest, split, reverse), workers.endpoints
                ) as session:
                    assert tokens(session) == expected
                    # Terminal event is emitted only after both stages release request state.
                    assert all(report.active_requests == 0 for report in session.memory_reports())
                    assert tokens(session) == expected
                wait_clean(workers, loaded=False)


@pytest.mark.parametrize("dtype", ["F16", "BF16"])
def test_low_precision_storage_decodes_to_float32(tmp_path: Path, dtype: str) -> None:
    manifest = write_model(tmp_path, dtype=dtype)
    with Workers(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest, None), workers.endpoints) as session:
            expected = tokens(session)
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            assert tokens(session) == expected
            assert (
                sum(report.loaded_weight_bytes for report in session.memory_reports())
                > manifest.total_storage_bytes
            )


def test_256_tokens_stop_conditions_and_repeated_cleanup(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            output = tokens(session, count=256)
            assert len(output) == 256
            assert tokens(session, stop_token_ids=[output[0]]) == output[:1]
            for _ in range(10):
                assert tokens(session, count=4) == output[:4]
                assert all(report.active_requests == 0 for report in session.memory_reports())


def test_load_rejection_rolls_back_and_memory_admission(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path, limit=100_000) as workers:
        bad = manifest.model_copy(
            update={
                "architecture": manifest.architecture.model_copy(
                    update={"architecture_revision": "99"}
                )
            }
        )
        with pytest.raises(RuntimeError, match="architecture"):
            with DeploymentSession(bad, plan(bad), workers.endpoints):
                pass
        wait_clean(workers, loaded=False)
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            with pytest.raises(grpc.RpcError) as error:
                tokens(session, count=256)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            wait_clean(workers)
            assert len(tokens(session, count=1)) == 1


def test_client_cancellation_and_worker_loss_cleanup(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            stream = session.generate([1, 4, 2], maximum_new_tokens=400)
            assert next(stream).HasField("prefill_complete")
            stream.close()
            wait_clean(workers)
            assert len(tokens(session, count=2)) == 2
            workers.processes[1].kill()
            workers.processes[1].wait(timeout=5)
            with pytest.raises(grpc.RpcError):
                tokens(session, timeout=2)
            report = workers.controls[0].GetMemoryReport(common_pb2.Empty(), timeout=2)
            assert report.active_requests == 0


def test_stage_protocol_rejects_wrong_order_and_disconnects(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as workers:
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
                with pytest.raises(grpc.RpcError):
                    list(call)
                wait_clean(workers)
            assert len(tokens(session, count=1)) == 1


def test_idle_stage_deadline_and_control_cancellation(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as workers:
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


def test_partial_load_failure_unloads_downstream(tmp_path: Path) -> None:
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
    with Workers(tmp_path) as workers:
        with pytest.raises(RuntimeError, match="missing required tensor"):
            with DeploymentSession(manifest, plan(manifest), workers.endpoints):
                pass
        wait_clean(workers, loaded=False)


def test_controller_cli_prepare_plan_generate(tmp_path: Path) -> None:
    from hllm_control.cli import app
    from typer.testing import CliRunner

    write_model(tmp_path)
    with Workers(tmp_path) as workers:
        worker_config = (ROOT / "examples/profiles/workers-cpu-local.yaml").read_text()
        for name, default in (("cpu-a", "127.0.0.1:50051"), ("cpu-b", "127.0.0.1:50052")):
            worker_config = worker_config.replace(default, workers.endpoints[name])
        config_path = tmp_path / "workers.yaml"
        config_path.write_text(worker_config)
        runner = CliRunner()
        manifest_path = tmp_path / "manifest.json"
        plan_path = tmp_path / "plan.json"
        result = runner.invoke(app, ["prepare", str(tmp_path), "--output", str(manifest_path)])
        assert result.exit_code == 0, result.output
        result = runner.invoke(
            app,
            [
                "plan",
                "--manifest",
                str(manifest_path),
                "--workers",
                str(config_path),
                "--links",
                str(ROOT / "examples/profiles/links-cpu-local.yaml"),
                "--workload",
                str(ROOT / "examples/workloads/cpu-demo.yaml"),
                "--settings",
                str(ROOT / "examples/profiles/planner-cpu.yaml"),
                "--output",
                str(plan_path),
                "--report",
                str(tmp_path / "report.json"),
            ],
        )
        assert result.exit_code == 0, result.output
        result = runner.invoke(
            app,
            [
                "generate",
                "--manifest",
                str(manifest_path),
                "--plan",
                str(plan_path),
                "--workers",
                str(config_path),
                "--token-ids",
                "1,4,2",
                "--max-new-tokens",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len([int(item) for item in result.output.split()]) == 4
        wait_clean(workers, loaded=False)


def test_transmitted_prefill_matches_float16_transformers_oracle(tmp_path: Path) -> None:
    # A test-only relay observes bytes; the production controller never relays activations.
    from concurrent.futures import ThreadPoolExecutor

    manifest = write_model(tmp_path)
    captured: list[execution_pb2.TensorEnvelope] = []
    with Workers(tmp_path) as workers:
        downstream = execution_pb2_grpc.StageExecutionStub(workers.channels[1])

        class RecordingRelay(execution_pb2_grpc.StageExecutionServicer):
            def Execute(self, request_iterator, context):
                def forward():
                    for message in request_iterator:
                        if message.HasField("tensor"):
                            saved = execution_pb2.TensorEnvelope()
                            saved.CopyFrom(message.tensor)
                            captured.append(saved)
                        yield message

                yield from downstream.Execute(forward(), timeout=5)

        with ThreadPoolExecutor(max_workers=2) as executor:
            server = grpc.server(executor)
            execution_pb2_grpc.add_StageExecutionServicer_to_server(RecordingRelay(), server)
            port = server.add_insecure_port("127.0.0.1:0")
            server.start()
            try:
                wire_plan = plan(manifest)
                from hllm_control.wire import deployment_plan_to_proto, model_manifest_to_proto

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
                result = list(
                    execution_pb2_grpc.GenerationStub(workers.channels[0]).Generate(
                        execution_pb2.GenerationRequest(
                            deployment_id=wire_plan.plan_id,
                            deployment_version=1,
                            request_id="capture",
                            token_ids=[1, 4, 2, 8, 3],
                            maximum_new_tokens=1,
                        ),
                        timeout=5,
                    )
                )
                assert result[-1].terminal.state == execution_pb2.TERMINAL_STATE_COMPLETED
                assert len(captured) == 1
                values = struct.unpack("<30e", captured[0].payload)
                oracle = json.loads((ROOT / "tests/fixtures/qwen3/tiny-reference.json").read_text())
                expected = oracle["layer_outputs"][0]["values"]
                assert values == pytest.approx(expected, abs=1e-3)
                logits = oracle["logits"]["values"][-11:]
                assert result[1].token.token_id == max(range(11), key=logits.__getitem__)
                wait_clean(workers)
            finally:
                server.stop(0).wait()


def test_downstream_memory_rejection_preserves_status_and_releases_driver(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path, limit=(16_000_000, 100_000)) as workers:
        with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
            with pytest.raises(grpc.RpcError) as error:
                tokens(session, count=256)
            assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
            wait_clean(workers)
            assert len(tokens(session, count=1)) == 1
