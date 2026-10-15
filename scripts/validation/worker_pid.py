"""Print the process ID a running worker reports through its control API.

Launchers bind out-of-band samplers (`memory_watch.py --cuda`) to this PID rather than
matching command lines, which can also match the worker's guard or wrapper. The
worker must answer at the endpoint and, with `--expect-worker-id`, report that ID.
"""

from __future__ import annotations

import argparse
import sys
import time

import grpc
from hllm_control.proto import common_pb2, control_pb2_grpc


def worker_pid(endpoint: str, *, expect_worker_id: str | None = None, timeout: float = 5) -> int:
    with grpc.insecure_channel(endpoint) as channel:
        control = control_pb2_grpc.WorkerControlStub(channel)
        state = control.GetQualificationState(common_pb2.Empty(), timeout=timeout)
        if expect_worker_id is not None:
            worker = control.GetCapabilities(common_pb2.Empty(), timeout=timeout).worker
            if worker.worker_id != expect_worker_id:
                raise ValueError(
                    f"{endpoint} is worker {worker.worker_id!r}, not {expect_worker_id!r}"
                )
    if state.process_id <= 0:
        raise ValueError(f"{endpoint} did not report a process ID")
    return int(state.process_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoint", help="host:port of the worker control service")
    parser.add_argument("--expect-worker-id")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0,
        help="keep retrying an unreachable worker for this long",
    )
    args = parser.parse_args(argv)
    deadline = time.monotonic() + max(0.0, args.wait_seconds)
    while True:
        try:
            print(worker_pid(args.endpoint, expect_worker_id=args.expect_worker_id))
            return 0
        except grpc.RpcError as error:
            if time.monotonic() >= deadline:
                print(f"{args.endpoint}: {error.code().name}", file=sys.stderr)
                return 1
            time.sleep(0.5)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
