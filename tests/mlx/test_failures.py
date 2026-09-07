"""CPU/MLX lifecycle qualification through real native worker processes."""

from pathlib import Path

import pytest
from hllm_control.proto import execution_pb2

from tests.mlx.helpers import mixed_workers
from tests.process_contracts import (
    check_admission_rollback,
    check_boundary_fault,
    check_idle_cancellation,
    check_partial_load_rollback,
    check_stage_protocol,
)
from tests.process_helpers import Workers


@pytest.mark.parametrize("mlx_first", [False, True])
@pytest.mark.parametrize(
    "phase", [execution_pb2.EXECUTION_PHASE_PREFILL, execution_pb2.EXECUTION_PHASE_DECODE]
)
@pytest.mark.parametrize(
    "fault", ["client", "control", "deadline", "disconnect", "kill_first", "kill_final"]
)
def test_fault_at_boundary(tmp_path: Path, mlx_first: bool, phase: int, fault: str) -> None:
    def launch(root: Path, limit: int | tuple[int, int] = 128 * 1024 * 1024) -> Workers:
        return mixed_workers(root, limit, mlx_first=mlx_first)

    check_boundary_fault(tmp_path, launch, phase, fault)


@pytest.mark.parametrize("mlx_first", [False, True])
@pytest.mark.parametrize("scenario", ["protocol", "idle", "load", "admission"])
def test_failure_contracts(tmp_path: Path, mlx_first: bool, scenario: str) -> None:
    def launch(root: Path, limit: int | tuple[int, int] = 128 * 1024 * 1024) -> Workers:
        if scenario == "admission" and not mlx_first:
            # MLX has an 8 MiB per-request workspace floor. Preserve the CPU
            # budget; 10 MiB permits recovery with one token but rejects 256.
            limit = (limit[0] if isinstance(limit, tuple) else limit, 10 * 1024 * 1024)
        return mixed_workers(root, limit, mlx_first=mlx_first)

    cases = {
        "protocol": check_stage_protocol,
        "idle": check_idle_cancellation,
        "load": check_partial_load_rollback,
        "admission": check_admission_rollback,
    }
    cases[scenario](tmp_path, launch)
