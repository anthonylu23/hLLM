"""Real process/transport tests. Build hllm-worker-cpu before running this suite."""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import time
from pathlib import Path

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import DType, ModelManifest, PlanningMode, StageAssignment
from hllm_control.prepare.manifest import prepare_model
from hllm_control.proto import (
    common_pb2,
    control_pb2_grpc,
    execution_pb2,
)

ROOT = Path(__file__).resolve().parents[1]
BINARY = Path(os.environ.get("HLLM_CPU_WORKER", ROOT / "build/native/dev/cpp/hllm-worker-cpu"))


def write_model(
    path: Path,
    family: str = "qwen3",
    dtype: str = "F32",
    *,
    tied: bool | None = None,
    redundant_head: bool = False,
) -> ModelManifest:
    path.mkdir(exist_ok=True)
    oracle = json.loads((ROOT / "tests/fixtures/qwen3/tiny-reference.json").read_text())
    config = oracle["config"]
    config.update(
        model_type=family,
        architectures=["Qwen3ForCausalLM" if family == "qwen3" else "LlamaForCausalLM"],
        eos_token_id=None,
    )
    weights = {name: value for name, value in oracle["weights"].items() if name != "lm_head.weight"}
    if family == "llama":
        config["num_hidden_layers"] = 4
        config["tie_word_embeddings"] = False
        weights = {
            name: value
            for name, value in weights.items()
            if "q_norm" not in name and "k_norm" not in name
        }
        for layer in (2, 3):
            for name, value in list(weights.items()):
                if name.startswith(f"model.layers.{layer - 2}."):
                    weights[name.replace(f"layers.{layer - 2}.", f"layers.{layer}.")] = value
        weights["lm_head.weight"] = oracle["weights"]["lm_head.weight"]
    if tied is not None:
        config["tie_word_embeddings"] = tied
        if tied:
            weights.pop("lm_head.weight", None)
        else:
            weights["lm_head.weight"] = oracle["weights"]["lm_head.weight"]
    if redundant_head:
        # Some tied checkpoints still ship lm_head.weight; workers must ignore it.
        assert config["tie_word_embeddings"]
        weights["lm_head.weight"] = oracle["weights"]["lm_head.weight"]
    config["max_position_embeddings"] = 512
    (path / "config.json").write_text(json.dumps(config))
    payload = bytearray()
    header = {}
    for name, tensor in sorted(weights.items()):
        start = len(payload)
        for value in tensor["values"]:
            if dtype == "BF16":
                bits = struct.unpack("<I", struct.pack("<f", value))[0]
                payload.extend(struct.pack("<H", (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16))
            else:
                payload.extend(struct.pack("<f" if dtype == "F32" else "<e", value))
        header[name] = {
            "shape": tensor["shape"],
            "dtype": dtype,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return prepare_model(path)


def endpoint() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return f"127.0.0.1:{sock.getsockname()[1]}"


class Workers:
    def __init__(
        self,
        root: Path,
        limit: int | tuple[int, int] = 16_000_000,
        *,
        binaries: tuple[Path, Path] = (BINARY, BINARY),
        extra_args: tuple[tuple[str, ...], tuple[str, ...]] = ((), ()),
        worker_ids: tuple[str, str] = ("cpu-a", "cpu-b"),
    ) -> None:
        for binary in binaries:
            if not binary.is_file():
                pytest.fail(f"Build the requested native worker first: {binary}")
        self.endpoints = {name: endpoint() for name in worker_ids}
        self.processes = []
        self.channels = []
        self.controls = []
        try:
            for index, (name, address) in enumerate(self.endpoints.items()):
                self.processes.append(
                    subprocess.Popen(
                        [
                            str(binaries[index]),
                            "--listen",
                            address,
                            "--worker-id",
                            name,
                            "--model-root",
                            str(root),
                            "--memory-limit-bytes",
                            str(limit if isinstance(limit, int) else limit[index]),
                            *extra_args[index],
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                )
                channel = grpc.insecure_channel(address)
                self.channels.append(channel)
                grpc.channel_ready_future(channel).result(timeout=30)
                self.controls.append(control_pb2_grpc.WorkerControlStub(channel))
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for channel in self.channels:
            channel.close()
        for process in self.processes:
            process.terminate()
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stderr:
                process.stderr.close()

    def __enter__(self) -> Workers:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def plan(manifest: ModelManifest, split: int | None = 1, reverse: bool = False):
    # The planner emits two stages; a one-stage reference plan is constructed here.
    from hllm_control.models import DeploymentPlan

    names = ["cpu-b", "cpu-a"] if reverse else ["cpu-a", "cpu-b"]
    ranges = (
        [(0, split), (split, manifest.config.num_layers)]
        if split
        else [(0, manifest.config.num_layers)]
    )
    return DeploymentPlan(
        plan_id=f"test-{split}-{reverse}",
        plan_digest=f"digest-{split}-{reverse}",
        manifest_digest=manifest.manifest_digest,
        workload_id="tiny",
        planning_mode=PlanningMode.FEASIBILITY,
        execution_dtype=DType.F32,
        activation_dtype=DType.F16,
        split_layer=split or 0,
        stages=tuple(
            StageAssignment(
                stage_index=i,
                worker_id=names[i],
                layer_start=start,
                layer_end=end,
                owns_token_embedding=i == 0,
                owns_final_norm=i == len(ranges) - 1,
                owns_lm_head=i == len(ranges) - 1,
                owns_sampling=i == len(ranges) - 1,
            )
            for i, (start, end) in enumerate(ranges)
        ),
        selected_candidate_id="test",
        duplicated_tensor_groups=("token_embeddings",)
        if split and manifest.config.tied_embeddings
        else (),
    )


def tokens(
    session: DeploymentSession,
    count: int = 8,
    *,
    stop_token_ids: list[int] | None = None,
    timeout: float = 60,
) -> list[int]:
    events = list(
        session.generate(
            [1, 4, 2],
            maximum_new_tokens=count,
            stop_token_ids=stop_token_ids,
            timeout=timeout,
        )
    )
    assert events[0].HasField("prefill_complete")
    assert events[-1].terminal.state == execution_pb2.TERMINAL_STATE_COMPLETED
    sampled = [event.token.token_id for event in events if event.HasField("token")]
    positions = [event.token.token_position for event in events if event.HasField("token")]
    assert positions == list(range(3, 3 + len(sampled)))
    assert events[-2].usage.generated_tokens == len(sampled)
    return sampled


def wait_clean(workers: Workers, *, loaded: bool = True) -> None:
    until = time.monotonic() + 5
    while True:
        reports = [stub.GetMemoryReport(common_pb2.Empty(), timeout=2) for stub in workers.controls]
        if all(
            r.active_requests == 0
            and r.reserved_cache_bytes == 0
            and r.reserved_workspace_bytes == 0
            for r in reports
        ):
            if not loaded:
                assert all(r.loaded_weight_bytes == 0 for r in reports)
            return
        assert time.monotonic() < until, reports
        time.sleep(0.01)
