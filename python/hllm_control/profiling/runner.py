"""Run one assignment in an owned native process and seal its memory observations."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from google.protobuf.json_format import MessageToDict

from hllm_control.models import Backend, DeploymentPlan, ModelManifest, WorkloadProfile
from hllm_control.profiling.memory import PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    ComputeRunMeasurement,
    Conditions,
    Environment,
    MemoryAmounts,
    MemoryMeasurement,
    MemorySample,
    PhysicalSample,
    ProfileArtifact,
    ProfileKey,
    TimingRecord,
    digest,
    make_artifact,
)
from hllm_control.proto import control_pb2
from hllm_control.serialization import sha256_file
from hllm_control.wire import deployment_plan_to_proto, model_manifest_to_proto


def checkpoint_digest(root: Path, manifest: ModelManifest) -> str:
    root = root.resolve(strict=True)
    config = (root / "config.json").resolve(strict=True)
    if not config.is_relative_to(root):
        raise ValueError("checkpoint config escapes model root")
    if sha256_file(config) != manifest.source.config_sha256:
        raise ValueError("checkpoint config differs from manifest")
    hashes: dict[str, object] = {"config.json": sha256_file(config)}
    for shard in manifest.tensor_files:
        path = (root / shard.name).resolve(strict=True)
        if not path.is_relative_to(root) or path.stat().st_size != shard.size_bytes:
            raise ValueError("checkpoint shard path/size differs from manifest")
        value = sha256_file(path)
        if shard.sha256 is not None and shard.sha256 != value:
            raise ValueError("checkpoint payload differs from manifest")
        hashes[shard.name] = value
    return digest(hashes)


def host_available_bytes() -> int | None:
    """Linux MemAvailable, or conservative free+inactive pages on macOS."""
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        elif platform.system() == "Darwin":
            lines = subprocess.check_output(["vm_stat"], text=True, timeout=5).splitlines()
            page_size = int(lines[0].split("page size of ")[1].split()[0])
            pages = sum(
                int(line.split(":")[1].strip().rstrip("."))
                for line in lines[1:]
                if line.startswith(("Pages free:", "Pages inactive:"))
            )
            return pages * page_size
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def device_available_bytes(device_id: int) -> int | None:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(device_id),
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return int(value.strip()) * 1024**2
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def observe_process(pid: int, *, cuda: bool, elapsed: float) -> PhysicalSample:
    rss: int | None = None
    gpu: int | None = None
    try:
        value = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        rss = int(value.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    if cuda:
        try:
            value = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            )
            values = [
                int(size.strip()) * 1024**2
                for process, size in (line.split(",") for line in value.splitlines())
                if int(process) == pid
            ]
            if values:
                gpu = sum(values)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return PhysicalSample(elapsed_seconds=elapsed, rss_bytes=rss, device_process_bytes=gpu)


def write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run_memory_profile(
    *,
    binary: Path,
    root: Path,
    manifest: ModelManifest,
    plan: DeploymentPlan,
    stage_index: int,
    workload: WorkloadProfile,
    backend: Backend,
    capacity: MemoryAmounts,
    output: Path,
    source_revision: str,
    concurrent_load: str,
    transport_mode: Literal["pageable", "pinned"] = "pageable",
    device_id: int = 0,
    warmup_cycles: int = 0,
    measured_cycles: int = 2,
    timeout_seconds: float = 600,
    sample_interval_seconds: float = 0.25,
    host_headroom_bytes: int = 1024**3,
    device_headroom_bytes: int = 512 * 1024**2,
    extra_overhead_bytes: int = 256 * 1024**2,
    mode: Literal["memory", "compute"] = "memory",
) -> ProfileArtifact:
    if not 0 <= stage_index < len(plan.stages):
        raise ValueError("stage index outside plan")
    if plan.manifest_digest != manifest.manifest_digest:
        raise ValueError("plan manifest mismatch")
    if plan.stages[-1].layer_end != manifest.config.num_layers:
        raise ValueError("plan does not cover the manifest layers")
    if plan.activation_dtype != workload.activation_dtype:
        raise ValueError("plan/workload boundary dtype mismatch")
    if workload.total_cached_tokens > manifest.config.maximum_sequence_length:
        raise ValueError("workload exceeds model context")
    if (
        warmup_cycles < 0
        or measured_cycles < 1
        or warmup_cycles + measured_cycles > 32
        or not 0 < timeout_seconds <= 86400
        or not 0.05 <= sample_interval_seconds <= 5
        or device_id < 0
    ):
        raise ValueError("invalid profiling run limits")
    binary = binary.resolve(strict=True)
    root = root.resolve(strict=True)
    output = output.resolve()
    raw = output.with_suffix(".native.jsonl")
    spec_path = output.with_suffix(".spec.json")
    fit_path = output.with_suffix(".fit.json")
    log_path = output.with_suffix(".log")
    preflight_path = output.with_suffix(".preflight.json")
    for path in (
        output,
        raw,
        spec_path,
        fit_path,
        log_path,
        preflight_path,
        output.with_suffix(".summary.json"),
    ):
        if path.exists():
            raise FileExistsError(f"profile output already exists: {path}")
    binary_hash = sha256_file(binary)
    environment = Environment(
        backend=backend,
        device_identity="pending",
        device_name="pending",
        backend_version="pending",
        driver_version="pending",
        allocator="pending",
        allocator_config=json.dumps(
            {
                "capacity": capacity.model_dump(),
                "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            },
            sort_keys=True,
        ),
        source_revision=source_revision,
        binary_digest=binary_hash,
        compiler="pending",
        os=platform.platform() + ";kernel=" + platform.version(),
    )
    # Validate all workload/ownership/dtype constraints before starting native work.
    key = ProfileKey(
        manifest_digest=manifest.manifest_digest,
        checkpoint_digest=checkpoint_digest(root, manifest),
        environment=environment,
        assignment=plan.stages[stage_index],
        workload=workload,
        workload_digest=digest(workload),
        execution_dtype=plan.execution_dtype,
        weight_dtype=plan.weight_dtype,
        transport_mode=transport_mode,
        input_kind="synthetic-shape",
        input_digest=digest({"generator": "zero-token-or-zero-f16-boundary-v1"}),
    )
    host_budget = PhysicalBudget(
        available_bytes=host_available_bytes(),
        headroom_bytes=host_headroom_bytes,
        extra_overhead_bytes=extra_overhead_bytes,
    )
    device_budget = (
        PhysicalBudget(
            available_bytes=device_available_bytes(device_id),
            headroom_bytes=device_headroom_bytes,
            extra_overhead_bytes=extra_overhead_bytes,
        )
        if backend == Backend.CUDA
        else None
    )
    host_cap = capacity.unified if backend == Backend.MLX else capacity.host
    # Conservative cold-load preflight. A configured cap is not physical free memory.
    preflight_reasons: list[str] = []
    for name, cap, budget in (
        ("host/unified", host_cap, host_budget),
        ("device", capacity.device, device_budget),
    ):
        if budget is None:
            continue
        if budget.available_bytes is None:
            preflight_reasons.append(f"{name}: current availability unavailable")
            continue
        if (
            cap * (1 + budget.safety_fraction) + budget.headroom_bytes + budget.extra_overhead_bytes
            > budget.available_bytes
        ):
            preflight_reasons.append(
                f"{name}: configured probe cap plus physical allowances exceeds availability"
            )
    write_exclusive(
        preflight_path,
        {
            "schema_version": "1.0",
            "observed_at": datetime.now(UTC).isoformat(),
            "key": key.model_dump(mode="json"),
            "admission_capacity": capacity.model_dump(),
            "host_budget": host_budget.model_dump(),
            "device_budget": device_budget.model_dump() if device_budget else None,
            "status": "rejected" if preflight_reasons else "accepted",
            "reasons": preflight_reasons,
        },
    )
    if preflight_reasons:
        raise ValueError("; ".join(preflight_reasons) + f"; see {preflight_path}")
    load = control_pb2.LoadStageRequest(
        plan=deployment_plan_to_proto(plan),
        manifest=model_manifest_to_proto(manifest),
        stage_index=stage_index,
    )
    write_exclusive(
        spec_path,
        {
            "mode": mode,
            "load_request": MessageToDict(load),
            "budget": capacity.model_dump(),
            "capacity_tokens": workload.total_cached_tokens,
            "prompt_tokens": workload.prompt_tokens,
            "output_tokens": workload.output_tokens,
            "cycles": warmup_cycles + measured_cycles,
            "transport_mode": transport_mode,
            "device_id": device_id,
        },
    )
    conditions = Conditions(
        measured_at=datetime.now(UTC),
        warmup_cycles=warmup_cycles,
        measured_cycles=measured_cycles,
        process_policy=(
            "fresh-process-then-reloads" if mode == "memory" else "loaded-stage-paired-passes"
        ),
        concurrent_load=concurrent_load,
        notes=(
            (
                "Paired identical synthetic inputs; completed native work timed.",
                "External physical telemetry disabled during timing.",
            )
            if mode == "compute"
            else (
                "Synthetic shape exercise; not a correctness or timing benchmark.",
                "External physical sampling is a lower bound; RSS high water is lifetime.",
            )
        ),
    )
    physical: list[PhysicalSample] = []
    timed_out = False
    with log_path.open("x") as log:
        process = subprocess.Popen(
            [str(binary), str(root), str(spec_path), str(raw)], stdout=log, stderr=log
        )
        start = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - start > timeout_seconds:
                    timed_out = True
                    process.kill()
                    break
                if mode == "memory":
                    physical.append(
                        observe_process(
                            process.pid,
                            cuda=backend == Backend.CUDA,
                            elapsed=time.monotonic() - start,
                        )
                    )
                time.sleep(sample_interval_seconds)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    records: list[dict[str, object]] = []
    for line in raw.read_text().splitlines() if raw.exists() else []:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break  # Preserve an interrupted raw record; completed remains false.
    env = next((r for r in records if r.get("event") == "environment"), None)
    if env is None:
        raise RuntimeError(f"native profiler failed before initialization; see {log_path}")
    if str(env["backend"]) != f"BACKEND_{backend.value.upper()}":
        raise ValueError("native executable backend does not match requested profile")
    # cudaDriverGetVersion reports CUDA API compatibility, not the NVIDIA release.
    # Keep both so a driver update cannot accidentally reuse an old profile.
    if backend == Backend.CUDA:
        release = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(device_id),
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
        ).strip()
        if not release:
            raise RuntimeError("NVIDIA driver release unavailable; raw measurements preserved")
        env["driver_version"] = f"NVIDIA {release}; CUDA API {env['driver_version']}"
    elif platform.system() == "Darwin":
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=5
        ).strip()
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True, timeout=5).strip()
        env["device_name"] = f"{env['device_name']}; {brand}; {model}"
    environment = Environment.model_validate(
        {
            **environment.model_dump(),
            **{
                k: env[k]
                for k in (
                    "device_name",
                    "backend_version",
                    "driver_version",
                    "allocator",
                    "compiler",
                )
            },
            "device_identity": digest(
                {
                    "host": platform.node(),
                    "device": env["device_identity"],
                    "hardware": env["device_name"],
                }
            ),
        }
    )
    key = ProfileKey.model_validate({**key.model_dump(), "environment": environment})
    samples = tuple(
        MemorySample.model_validate({k: v for k, v in r.items() if k != "event"})
        for r in records
        if r.get("event") == "sample"
    )
    completed = process.returncode == 0 and any(r.get("event") == "complete" for r in records)
    error = (
        None
        if completed
        else "timeout"
        if timed_out
        else next(
            (str(r["error"]) for r in records if r.get("event") == "error"),
            f"native process exited {process.returncode}",
        )
    )
    if mode == "compute":
        from hllm_control.profiling.compute import summarize_compute

        artifact = make_artifact(
            key,
            conditions,
            ComputeRunMeasurement(
                admission_capacity=capacity,
                records=tuple(
                    TimingRecord.model_validate({k: v for k, v in r.items() if k != "event"})
                    for r in records
                    if r.get("event") == "timing"
                ),
                identical_output_cycles=tuple(
                    int(str(r["cycle"]))
                    for r in records
                    if r.get("event") == "pair_complete" and r.get("outputs_identical") is True
                ),
                completed=completed,
                error=error,
            ),
        )
        write_exclusive(output, artifact.model_dump(mode="json"))
        write_exclusive(output.with_suffix(".summary.json"), summarize_compute(artifact))
        return artifact
    artifact = make_artifact(
        key,
        conditions,
        MemoryMeasurement(
            admission_capacity=capacity,
            samples=samples,
            physical_samples=tuple(physical),
            physical_sample_interval_seconds=sample_interval_seconds,
            completed=completed,
            error=error,
        ),
    )
    write_exclusive(output, artifact.model_dump(mode="json"))
    fit = assess_fit(artifact, capacity, host_budget, device_budget)
    write_exclusive(
        fit_path,
        {
            "schema_version": "1.0",
            "profile_digest": artifact.artifact_digest,
            "observed_at": conditions.measured_at.isoformat(),
            "host_budget": host_budget.model_dump(),
            "device_budget": device_budget.model_dump() if device_budget else None,
            "assessment": fit.model_dump(),
        },
    )
    return artifact
