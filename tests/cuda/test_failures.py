"""Mixed-backend lifecycle faults at controlled transport boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import execution_pb2

from tests.cuda.mixed_helpers import mixed_workers
from tests.process_contracts import (
    check_admission_rollback,
    check_boundary_fault,
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
    def launch(root: Path, limit: int | tuple[int, int] = 64 * 1024 * 1024) -> Workers:
        return mixed_workers(root, mode=mode, cuda_first=cuda_first, host_limit=limit)

    check_boundary_fault(tmp_path, launch, phase, fault)


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
