"""Owned worker/probe process helper, also usable over SSH with stdin as its lease."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hllm_control.models import Backend, DeploymentPlan, ModelManifest, WorkloadProfile
from hllm_control.profiling.models import MemoryAmounts
from hllm_control.profiling.runner import run_memory_profile
from hllm_control.qualification.identity import package_digest
from hllm_control.serialization import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("serve", "probe"))
    args = parser.parse_args()
    data = json.loads(sys.stdin.readline())
    if data.get("package_digest") != package_digest():
        raise ValueError("host/controller package digest mismatch")
    w = data["worker"]
    binary = Path(w["binary"] if args.mode == "serve" else w["memory_binary"])
    expected = w["binary_digest"] if args.mode == "serve" else w["memory_binary_digest"]
    if sha256_file(binary) != expected:
        raise ValueError("host executable digest changed")
    capacity = MemoryAmounts.model_validate(w["capacity"])
    if args.mode == "probe":
        output = Path(w["evidence_root"]) / data["run_id"] / (w["worker_id"] + ".json")
        artifact = run_memory_profile(
            binary=binary,
            root=Path(w["model_root"]),
            manifest=ModelManifest.model_validate(data["manifest"]),
            plan=DeploymentPlan.model_validate(data["plan"]),
            stage_index=data["stage_index"],
            workload=WorkloadProfile.model_validate(data["workload"]),
            backend=Backend(w["backend"]),
            capacity=capacity,
            transport_mode=w["transport_mode"],
            output=output,
            source_revision=data["source_revision"],
            concurrent_load=data["concurrent_load"],
            warmup_cycles=0,
            measured_cycles=1,
            timeout_seconds=w["timeout_seconds"],
            host_headroom_bytes=w["host_headroom_bytes"],
            device_headroom_bytes=w["device_headroom_bytes"],
            extra_overhead_bytes=w["extra_overhead_bytes"],
        )
        files = {
            p.name: p.read_text() for p in output.parent.glob(output.stem + ".*") if p.is_file()
        }
        print(
            json.dumps({"artifact": artifact.model_dump(mode="json"), "files": files}), flush=True
        )
        return
    command = [
        str(binary),
        "--listen",
        w["endpoint"],
        "--worker-id",
        w["worker_id"],
        "--model-root",
        w["model_root"],
        "--memory-limit-bytes",
        str(capacity.unified if w["backend"] == "mlx" else capacity.host),
    ]
    if w["backend"] == "cuda":
        command += [
            "--device-memory-limit-bytes",
            str(capacity.device),
            "--boundary-transfer-mode",
            w["transport_mode"],
        ]
        if capacity.pinned:
            command += ["--pinned-host-memory-limit-bytes", str(capacity.pinned)]
    child = subprocess.Popen(command)
    print(json.dumps({"owned_pid": child.pid}), flush=True)
    try:
        # The controller closes this pipe on success, error or interruption. SSH EOF
        # also retires the remote native child; no process-name or port-wide kill.
        sys.stdin.read()
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
