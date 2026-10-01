"""Bounded, exact-reference reload qualification on explicitly selected, guarded workers.

Workers and SSH tunnels are launched separately under resource_guard.py. This
tool refuses busy workers, checks fresh profile fit before every deployment, and
streams snapshots to JSONL. It does not claim throughput or general capacity.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import grpc
from google.protobuf.json_format import MessageToDict
from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DeploymentPlan, ModelManifest
from hllm_control.profiling.memory import MlxFitPolicy, PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    Digest,
    MemoryAmounts,
    MemoryMeasurement,
    ProfileArtifact,
    ProfileModel,
    digest,
)
from hllm_control.proto import common_pb2, control_pb2_grpc, profile_pb2
from hllm_control.serialization import sha256_file
from pydantic import Field


class Worker(ProfileModel):
    endpoint: str
    memory_profile: Path
    binary_digest: Digest
    runtime_driver_version: str
    mlx_fit_policy: MlxFitPolicy = MlxFitPolicy.CONSERVATIVE
    host_headroom_bytes: int = Field(default=1024**3, ge=0)
    device_headroom_bytes: int = Field(default=512 * 1024**2, ge=0)
    extra_overhead_bytes: int = Field(default=256 * 1024**2, ge=0)


def reference_tokens(
    reference: dict, manifest: ModelManifest, *, cpu_rehearsal: bool = False
) -> tuple[list[int], list[int]]:
    files = reference["checkpoint_files"]
    if files.get("config.json") != manifest.source.config_sha256 or any(
        f.sha256 is None or files.get(f.name) != f.sha256 for f in manifest.tensor_files
    ):
        raise ValueError("reference checkpoint does not match the fully hashed manifest")
    if reference["checkpoint_digest"] != digest(files):
        raise ValueError("reference checkpoint digest mismatch")
    generation = reference["long_generation"]
    ids, expected = generation["token_ids"], generation["generated_ids"]
    if (
        not ids
        or len(expected) < 4
        or len(ids) + len(expected) > manifest.config.maximum_sequence_length
    ):
        raise ValueError("invalid reference token capacity")
    if any(
        type(t) is not int or not 0 <= t < manifest.config.vocabulary_size
        for t in [*ids, *expected]
    ):
        raise ValueError("reference contains invalid token IDs")
    producer = reference["producer"]
    if (
        producer.get("dtype") != ("f32" if cpu_rehearsal else "f16")
        or producer.get("attention") != "eager"
        or producer.get("tf32") is not False
        or (cpu_rehearsal and producer.get("purpose") != "cpu-rehearsal")
    ):
        raise ValueError(
            "reference precision/purpose mismatch: qualification requires F16; "
            "CPU rehearsal requires explicitly labeled F32; both require eager without TF32"
        )
    return ids, expected


def validate_profile(
    a: ProfileArtifact,
    manifest: ModelManifest,
    plan: DeploymentPlan,
    worker_id: str,
    checkpoint_digest: str,
    prompt_tokens: int,
    output_tokens: int,
) -> None:
    k = a.key
    if (
        k.manifest_digest != manifest.manifest_digest
        or k.checkpoint_digest != checkpoint_digest
        or k.assignment != next(s for s in plan.stages if s.worker_id == worker_id)
        or k.execution_dtype != plan.execution_dtype
        or k.weight_dtype != plan.weight_dtype
        or k.workload.kv_dtype != plan.execution_dtype
        or k.workload.activation_dtype != plan.activation_dtype
        or k.workload.prompt_tokens != prompt_tokens
        or k.workload.output_tokens != output_tokens
        or k.workload.total_cached_tokens != prompt_tokens + output_tokens
        or k.workload.concurrency != 1
        or not isinstance(a.measurement, MemoryMeasurement)
        or not a.measurement.completed
    ):
        raise ValueError("memory profile scope does not match this exact reload workload")


def fresh_fit(a: ProfileArtifact, worker: Worker, control) -> dict:
    age = (datetime.now(UTC) - a.conditions.measured_at).total_seconds()
    if not 0 <= age <= 86400:
        raise ValueError("memory profile is stale or future-dated; reprofile")
    q = control.GetQualificationState(common_pb2.Empty(), timeout=5)
    c = control.GetCapabilities(common_pb2.Empty(), timeout=5).worker
    e = a.key.environment
    if (
        q.binary_digest != worker.binary_digest
        or q.driver_version != worker.runtime_driver_version
        or q.device_fingerprint != e.device_identity
        or q.backend_version != e.backend_version
        or q.allocator != e.allocator
        or q.boundary_transport_mode != a.key.transport_mode
        or c.worker_id != a.key.assignment.worker_id
        or c.backend != profile_pb2.Backend.Value(f"BACKEND_{e.backend.value.upper()}")
    ):
        raise ValueError("worker binary/environment/identity changed since qualification setup")
    if e.backend == Backend.CUDA:
        allocator_config = json.loads(e.allocator_config)
        for name in ("pytorch_alloc_conf", "pytorch_cuda_alloc_conf"):
            if allocator_config.get(name.upper()) != (
                getattr(q, name) if q.HasField(name) else None
            ):
                raise ValueError("CUDA allocator configuration changed")
    capacity = MemoryAmounts(
        **{
            name: next((b.capacity_bytes for b in c.memory_budgets if b.domain == domain), 0)
            for name, domain in (
                ("host", profile_pb2.MEMORY_DOMAIN_HOST),
                ("unified", profile_pb2.MEMORY_DOMAIN_UNIFIED),
                ("device", profile_pb2.MEMORY_DOMAIN_DEVICE),
                ("pinned", profile_pb2.MEMORY_DOMAIN_HOST_PINNED),
            )
        }
    )
    assert isinstance(a.measurement, MemoryMeasurement)
    if capacity != a.measurement.admission_capacity:
        raise ValueError("worker admission cap differs from profiled cap")
    host = PhysicalBudget(
        available_bytes=q.available_host_bytes if q.HasField("available_host_bytes") else None,
        headroom_bytes=worker.host_headroom_bytes,
        extra_overhead_bytes=worker.extra_overhead_bytes,
    )
    device = (
        PhysicalBudget(
            available_bytes=q.available_device_bytes
            if q.HasField("available_device_bytes")
            else None,
            headroom_bytes=worker.device_headroom_bytes,
            extra_overhead_bytes=worker.extra_overhead_bytes,
        )
        if e.backend == Backend.CUDA
        else None
    )
    fit = assess_fit(a, capacity, host, device, mlx_policy=worker.mlx_fit_policy)
    result = dict(
        qualification=MessageToDict(q),
        assessment=fit.model_dump(mode="json"),
        host_budget=host.model_dump(),
        device_budget=device.model_dump() if device else None,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "plan", "reference", "workers", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--idle-seconds", type=float, default=10)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--cpu-rehearsal",
        action="store_true",
        help="CPU/F32 procedure check only; never constitutes F16 or accelerator qualification",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.cycles <= 16
        or not 1 <= args.requests <= 16
        or not 0 <= args.idle_seconds <= 60
        or not 0 < args.timeout <= 600
    ):
        parser.error("bounded cycles/requests (1..16), idle (0..60s), timeout (0..600s) required")
    manifest = ModelManifest.model_validate_json(args.manifest.read_text())
    plan = DeploymentPlan.model_validate_json(args.plan.read_text())
    if (
        plan.manifest_digest != manifest.manifest_digest
        or plan.execution_dtype.value != ("F32" if args.cpu_rehearsal else "F16")
        or plan.weight_dtype is not None
    ):
        raise ValueError("deployment must match manifest and selected uniform precision")
    reference = json.loads(args.reference.read_text())
    ids, expected = reference_tokens(reference, manifest, cpu_rehearsal=args.cpu_rehearsal)
    workers = {k: Worker.model_validate(v) for k, v in json.loads(args.workers.read_text()).items()}
    if set(workers) != {s.worker_id for s in plan.stages}:
        raise ValueError("worker names must exactly match plan stages")
    profiles = {
        k: ProfileArtifact.model_validate_json(w.memory_profile.read_text())
        for k, w in workers.items()
    }
    for k, a in profiles.items():
        if args.cpu_rehearsal and a.key.environment.backend != Backend.CPU:
            raise ValueError("CPU rehearsal requires CPU profiles for every worker")
        validate_profile(
            a, manifest, plan, k, reference["checkpoint_digest"], len(ids), len(expected)
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict = dict(
        completed=False,
        scope="cpu-rehearsal" if args.cpu_rehearsal else "f16-reload-qualification",
        started_unix_time=time.time(),
        cycles=[],
        purpose="bounded reload evidence; inspect serving peaks before capacity acceptance",
        plan=plan.model_dump(mode="json"),
        workers={k: w.model_dump(mode="json") for k, w in workers.items()},
        reference_sha256=sha256_file(args.reference),
        profile_digests={k: a.artifact_digest for k, a in profiles.items()},
    )
    args.output.touch(exist_ok=False)

    def save():
        with tempfile.NamedTemporaryFile(
            mode="w", dir=args.output.parent, prefix=args.output.name + ".", delete=False
        ) as f:
            temporary = Path(f.name)
            try:
                f.write(json.dumps(report, indent=2) + "\n")
                f.close()
                temporary.replace(args.output)
            finally:
                temporary.unlink(missing_ok=True)

    channels = {k: grpc.insecure_channel(w.endpoint) for k, w in workers.items()}
    controls = {k: control_pb2_grpc.WorkerControlStub(c) for k, c in channels.items()}
    try:
        with args.output.with_suffix(".observations.jsonl").open("x", buffering=1) as output:

            def snapshot(phase) -> dict:
                row: dict = dict(unix_time=time.time(), phase=phase, workers=[])
                for k, control in controls.items():
                    row["workers"].append(
                        dict(
                            worker=k,
                            memory=MessageToDict(
                                control.GetMemoryReport(common_pb2.Empty(), timeout=5)
                            ),
                            metrics=MessageToDict(
                                control.GetMetrics(common_pb2.Empty(), timeout=5)
                            ),
                        )
                    )
                output.write(json.dumps(row) + "\n")
                return row

            def clean(phase, unloaded=False):
                until = time.monotonic() + 15
                fields = ["activeRequests", "reservedCacheBytes", "reservedWorkspaceBytes"]
                if unloaded:
                    fields.append("loadedWeightBytes")
                while True:
                    row = snapshot(phase)
                    if all(
                        not any(int(w["memory"].get(f, 0)) for f in fields) for w in row["workers"]
                    ):
                        return row
                    if time.monotonic() >= until:
                        raise RuntimeError("workers are busy or reservations did not retire")
                    time.sleep(0.1)

            report["baseline"] = clean("baseline", True)
            for cycle in range(args.cycles):
                record: dict = dict(cycle=cycle, requests=[], preflight={})
                report["cycles"].append(record)
                for k, w in workers.items():
                    record["preflight"][k] = fresh_fit(profiles[k], w, controls[k])
                    save()
                    if record["preflight"][k]["assessment"]["status"] != "safe":
                        raise RuntimeError(f"fresh physical-fit gate rejected {k}; see report")
                with DeploymentSession(
                    manifest, plan, {k: w.endpoint for k, w in workers.items()}
                ) as session:
                    record["after_load"] = snapshot("after-load")
                    for request in range(args.requests):
                        start = time.monotonic()
                        tokens = []
                        row: dict = dict(
                            request=request, generated_ids=tokens, exact_reference_match=False
                        )
                        record["requests"].append(row)
                        stream = session.generate(
                            ids,
                            maximum_new_tokens=len(expected),
                            stop_token_ids=[],
                            timeout=args.timeout,
                        )
                        try:
                            for event in stream:
                                if event.HasField("token"):
                                    tokens.append(event.token.token_id)
                        except BaseException as error:
                            row["error"] = f"{type(error).__name__}: {error}"
                            raise
                        finally:
                            stream.close()
                            row["seconds"] = time.monotonic() - start
                            save()
                        row["exact_reference_match"] = tokens == expected
                        row["after"] = clean("request-cleanup")
                        save()
                        print(
                            json.dumps(
                                dict(
                                    cycle=cycle,
                                    request=request,
                                    exact_reference_match=row["exact_reference_match"],
                                )
                            ),
                            flush=True,
                        )
                        if tokens != expected:
                            raise RuntimeError("independent reference mismatch")
                    # Like request rows, the probe aliases its live token list so a
                    # mismatch or stream failure leaves the observed prefix in the report.
                    tokens = []
                    probe: dict = dict(
                        expected_prefix=expected[:3], generated_ids=tokens, exact_prefix_match=False
                    )
                    record["cancellation"] = probe
                    stream = session.generate(
                        ids,
                        maximum_new_tokens=len(expected),
                        stop_token_ids=[],
                        timeout=args.timeout,
                    )
                    try:
                        for event in stream:
                            if event.HasField("token"):
                                tokens.append(event.token.token_id)
                            if len(tokens) == 3:
                                break
                    except BaseException as error:
                        probe["error"] = f"{type(error).__name__}: {error}"
                        raise
                    finally:
                        stream.close()
                        save()
                    probe["exact_prefix_match"] = tokens == expected[:3]
                    if not probe["exact_prefix_match"]:
                        raise RuntimeError("cancellation prefix mismatch")
                    record["after_cancel"] = clean("cancel-cleanup")
                record["after_unload"] = clean("unloaded", True)
                time.sleep(args.idle_seconds)
                record["after_unload_idle"] = clean("unloaded-idle", True)
                save()
            report["completed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        # Session teardown has run. Preserve what each surviving worker reports,
        # without claiming cleanup for an unreachable worker or unloading outsiders.
        report["failure_cleanup"] = {}
        for k, control in controls.items():
            try:
                report["failure_cleanup"][k] = dict(
                    memory=MessageToDict(control.GetMemoryReport(common_pb2.Empty(), timeout=2))
                )
            except grpc.RpcError as cleanup_error:
                report["failure_cleanup"][k] = dict(error=str(cleanup_error))
        raise
    finally:
        for c in channels.values():
            c.close()
        report["finished_unix_time"] = time.time()
        save()


if __name__ == "__main__":
    main()
