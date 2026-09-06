"""Mixed-backend lifecycle faults at controlled transport boundaries."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import common_pb2, control_pb2, execution_pb2, execution_pb2_grpc

from tests.cuda.mixed_helpers import mixed_workers
from tests.process_contracts import (
    check_admission_rollback,
    check_idle_cancellation,
    check_partial_load_rollback,
    check_stage_protocol,
)
from tests.process_helpers import Workers, plan, tokens, wait_clean, write_model


@pytest.mark.parametrize("mode", ["pageable", "pinned"])
@pytest.mark.parametrize("cuda_first", [False, True])
@pytest.mark.parametrize(
    "phase", [execution_pb2.EXECUTION_PHASE_PREFILL, execution_pb2.EXECUTION_PHASE_DECODE]
)
@pytest.mark.parametrize(
    "fault", ["client", "control", "deadline", "disconnect", "kill_first", "kill_final"]
)
def test_fault_at_boundary(
    tmp_path: Path, mode: str, cuda_first: bool, phase: int, fault: str
) -> None:
    manifest = write_model(tmp_path)
    reached, release = threading.Event(), threading.Event()
    events = []
    with mixed_workers(tmp_path, mode=mode, cuda_first=cuda_first) as workers:
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


@pytest.mark.parametrize("mode", ["pageable", "pinned"])
@pytest.mark.parametrize("cuda_first", [False, True])
@pytest.mark.parametrize("scenario", ["protocol", "idle", "load", "admission"])
def test_existing_failure_contracts_on_mixed_workers(
    tmp_path: Path, mode: str, cuda_first: bool, scenario: str
) -> None:
    def launch(root: Path, limit: int | tuple[int, int] = 64 * 1024 * 1024) -> Workers:
        return mixed_workers(
            root,
            mode=mode,
            cuda_first=cuda_first,
            host_limit=limit,
            device_limit=(10 if scenario == "admission" and not cuda_first else 128) * 1024 * 1024,
        )

    cases = {
        "protocol": check_stage_protocol,
        "idle": check_idle_cancellation,
        "load": check_partial_load_rollback,
        "admission": check_admission_rollback,
    }
    cases[scenario](tmp_path, launch)


@pytest.mark.parametrize("mode", ["pageable", "pinned"])
@pytest.mark.parametrize("cuda_first", [False, True])
def test_worker_rss_plateau_across_reload(tmp_path: Path, mode: str, cuda_first: bool) -> None:
    import json
    import os

    manifest = write_model(tmp_path)
    samples: list[list[int]] = []
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    with mixed_workers(tmp_path, mode=mode, cuda_first=cuda_first) as workers:
        for cycle in range(12):
            with DeploymentSession(manifest, plan(manifest), workers.endpoints) as session:
                assert len(tokens(session, count=8)) == 8
                wait_clean(workers)
            wait_clean(workers, loaded=False)
            if cycle >= 3:
                samples.append(
                    [
                        int(Path(f"/proc/{process.pid}/statm").read_text().split()[1]) * page_bytes
                        for process in workers.processes
                    ]
                )
        ranges = [max(row[i] for row in samples) - min(row[i] for row in samples) for i in (0, 1)]
        assert all(delta <= 32 * 1024 * 1024 for delta in ranges), samples
        print(
            json.dumps(
                {
                    "mode": mode,
                    "cuda_first": cuda_first,
                    "rss_final": samples[-1],
                    "rss_range_after_warmup": ranges,
                    "cycles": 12,
                }
            )
        )
