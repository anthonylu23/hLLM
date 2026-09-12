"""Execute independent sweeps with owned local or SSH worker processes."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import grpc
from google.protobuf.json_format import MessageToDict
from pydantic import Field

from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DeploymentPlan
from hllm_control.profiling.memory import PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    Digest,
    MemoryAmounts,
    ProfileArtifact,
    ProfileModel,
    digest,
)
from hllm_control.profiling.runner import write_exclusive
from hllm_control.proto import common_pb2, control_pb2_grpc
from hllm_control.qualification.identity import package_digest
from hllm_control.qualification.sweep import JobResult, MemoryExclusion, Sample, SweepSpec

# Generated gRPC service factories are untyped.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


class NativeWorker(ProfileModel):
    worker_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    backend: Backend
    endpoint: str
    binary: str
    binary_digest: Digest
    memory_binary: str
    memory_binary_digest: Digest
    model_root: str
    evidence_root: str
    capacity: MemoryAmounts
    transport_mode: Literal["pageable", "pinned"] = "pageable"
    ssh_host: str | None = None
    # Explicit interpreter on the target, with hllm_control installed/synced.
    python: str = sys.executable
    host_headroom_bytes: int = Field(ge=0, default=1024**3)
    device_headroom_bytes: int = Field(ge=0, default=512 * 1024**2)
    extra_overhead_bytes: int = Field(ge=0, default=256 * 1024**2)
    timeout_seconds: float = Field(gt=0, le=3600, default=180)

    def command(self, mode: str) -> list[str]:
        command = [self.python, "-m", "hllm_control.qualification.host", mode]
        if self.ssh_host:
            return [
                "ssh",
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                self.ssh_host,
                shlex.join(command),
            ]
        return command


class NativeExecutor(ProfileModel):
    workers: Annotated[tuple[NativeWorker, ...], Field(min_length=2, max_length=2)]
    source_revision: str

    def identity(self) -> str:
        return digest({"config": self.model_dump(mode="json"), "package_digest": package_digest()})

    def __call__(
        self, spec: SweepSpec, job: str, plan: DeploymentPlan, directory: Path
    ) -> JobResult:
        if spec.executor_digest != self.identity():
            raise ValueError("executor changed since selection was frozen")
        raw: dict[str, object] = {}
        status: Literal["measured", "memory-excluded", "unknown", "oom", "correctness-failed"] = (
            "unknown"
        )
        detail = ""
        cleanup = False
        health = None
        exclusion = None
        host_envelopes: dict[str, int] = {}
        device_envelopes: dict[str, int] = {}
        samples: list[Sample] = []
        memory_evidence: dict[str, MemoryExclusion] = {}
        run_id = spec.sweep_digest[:12] + "-" + job + "-" + uuid4().hex[:8]
        names = {w.worker_id: w for w in self.workers}
        try:
            # Fresh native process per stage/assignment; no planner fit decisions are read.
            for assignment in plan.stages:
                w = names[assignment.worker_id]
                payload = dict(
                    package_digest=package_digest(),
                    worker=w.model_dump(mode="json"),
                    manifest=spec.manifest.model_dump(mode="json"),
                    plan=plan.model_dump(mode="json"),
                    workload=spec.workload.model_dump(mode="json"),
                    stage_index=assignment.stage_index,
                    run_id=run_id,
                    source_revision=self.source_revision,
                    concurrent_load=spec.concurrent_load,
                )
                probe = subprocess.run(
                    w.command("probe"),
                    input=json.dumps(payload) + "\n",
                    text=True,
                    capture_output=True,
                    timeout=w.timeout_seconds + 30,
                )
                raw[w.worker_id + ".probe_stderr"] = probe.stderr
                raw[w.worker_id + ".probe_stdout"] = probe.stdout
                if probe.returncode:
                    raise RuntimeError(
                        f"fresh memory qualification failed for {w.worker_id}: "
                        f"{probe.stderr[-2000:]}"
                    )
                result = json.loads(probe.stdout)
                artifact = ProfileArtifact.model_validate(result["artifact"])
                raw[w.worker_id + ".memory"] = result
                if (
                    artifact.key.weight_dtype != plan.weight_dtype
                    or artifact.key.execution_dtype != plan.execution_dtype
                    or artifact.key.manifest_digest != spec.manifest.manifest_digest
                    or artifact.key.assignment != assignment
                    or artifact.key.workload != spec.workload
                    or artifact.key.checkpoint_digest != spec.reference.checkpoint_digest
                ):
                    raise ValueError("independent memory scope/checkpoint mismatch")
                fit = next(
                    json.loads(v) for k, v in result["files"].items() if k.endswith(".fit.json")
                )
                independent = MemoryExclusion(
                    profile=artifact,
                    capacity=w.capacity,
                    host=PhysicalBudget.model_validate(fit["host_budget"]),
                    device=PhysicalBudget.model_validate(fit["device_budget"])
                    if fit["device_budget"]
                    else None,
                )
                memory_evidence[w.worker_id] = independent
                assessed = assess_fit(artifact, w.capacity, independent.host, independent.device)
                if assessed.host_envelope_bytes is not None:
                    host_envelopes[w.worker_id] = assessed.host_envelope_bytes
                if assessed.device_envelope_bytes is not None:
                    device_envelopes[w.worker_id] = assessed.device_envelope_bytes
                if assessed.status == "unsafe":
                    exclusion = independent
                    status = "memory-excluded"
                    cleanup = True
                    raise RuntimeError("independent physical/admission envelope excludes candidate")
                if assessed.status != "safe":
                    raise RuntimeError("independent physical/admission envelope unknown")
                if not getattr(artifact.measurement, "completed", False):
                    raise RuntimeError("fresh memory exercise did not complete")
            with ExitStack() as stack:
                controls = []
                processes = []
                for w in self.workers:
                    log = stack.enter_context((directory / f"{w.worker_id}.worker.log").open("wb"))
                    process = subprocess.Popen(
                        w.command("serve"), stdin=subprocess.PIPE, stdout=log, stderr=log
                    )
                    processes.append(process)
                    stack.callback(_stop, process)
                    assert process.stdin is not None
                    process.stdin.write(
                        (
                            json.dumps(
                                {
                                    "worker": w.model_dump(mode="json"),
                                    "package_digest": package_digest(),
                                }
                            )
                            + "\n"
                        ).encode()
                    )
                    process.stdin.flush()
                    channel = stack.enter_context(grpc.insecure_channel(w.endpoint))
                    grpc.channel_ready_future(channel).result(timeout=20)
                    control = control_pb2_grpc.WorkerControlStub(channel)
                    identity = control.GetQualificationState(common_pb2.Empty(), timeout=5)
                    owned_pids = []
                    for line in (directory / f"{w.worker_id}.worker.log").read_text().splitlines():
                        try:
                            owned_pids.append(json.loads(line)["owned_pid"])
                        except (ValueError, KeyError, TypeError):
                            pass
                    if (
                        identity.binary_digest != w.binary_digest
                        or process.poll() is not None
                        or owned_pids != [identity.process_id]
                    ):
                        raise ValueError("connected worker differs from owned executable")
                    if identity.boundary_transport_mode != w.transport_mode:
                        raise ValueError("owned worker transport mode differs from sweep")
                    independent = memory_evidence[w.worker_id]
                    if (
                        identity.device_fingerprint
                        != independent.profile.key.environment.device_identity
                    ):
                        raise ValueError("fresh probe and serving worker hardware differ")
                    host = independent.host.model_copy(
                        update={
                            "available_bytes": identity.available_host_bytes
                            if identity.HasField("available_host_bytes")
                            else None
                        }
                    )
                    device = (
                        independent.device.model_copy(
                            update={
                                "available_bytes": identity.available_device_bytes
                                if identity.HasField("available_device_bytes")
                                else None
                            }
                        )
                        if independent.device
                        else None
                    )
                    fresh = assess_fit(independent.profile, w.capacity, host, device)
                    if fresh.status == "unsafe":
                        exclusion = independent.model_copy(update={"host": host, "device": device})
                        status = "memory-excluded"
                        cleanup = True
                        raise RuntimeError("fresh pre-load physical memory gate failed")
                    if fresh.status != "safe":
                        raise RuntimeError("fresh pre-load physical memory gate unknown")
                    controls.append(control)
                    raw[w.worker_id + ".identity"] = MessageToDict(identity)
                with DeploymentSession(
                    spec.manifest, plan, {w.worker_id: w.endpoint for w in self.workers}
                ) as session:
                    for _ in range(spec.warmups + 1):
                        ids: list[int] = []
                        arrivals: list[float] = []
                        native: list[float] = []
                        native_setup = None
                        started = time.perf_counter()
                        stream = session.generate(
                            spec.reference.prompt_ids,
                            maximum_new_tokens=spec.workload.output_tokens,
                            stop_token_ids=[],
                            timeout=max(w.timeout_seconds for w in self.workers),
                            capture_timing=True,
                        )
                        try:
                            for event in stream:
                                if event.HasField("token"):
                                    arrivals.append((time.perf_counter() - started) * 1000)
                                    ids.append(event.token.token_id)
                                    if event.token.HasField("native_request_setup_ms"):
                                        native_setup = event.token.native_request_setup_ms
                                    if event.token.HasField("native_elapsed_ms"):
                                        native.append(event.token.native_elapsed_ms)
                        finally:
                            stream.close()
                        sample = Sample(
                            token_ids=tuple(ids),
                            client_arrivals_ms=tuple(arrivals),
                            native_arrivals_ms=tuple(native),
                            native_request_setup_ms=native_setup,
                            terminal_ms=(time.perf_counter() - started) * 1000,
                        )
                        samples.append(sample)
                        _clean(controls, weights=False)
                        if sample.token_ids != spec.reference.generated_ids:
                            status = "correctness-failed"
                            raise RuntimeError("independent reference token divergence")
                    if job.endswith("reference-after"):
                        raw["selected_health"] = _health(session, controls, spec, samples[-1])
                        health = True
                _clean(controls, weights=True)
                cleanup = True
                status = "measured"
                detail = (
                    "fresh processes, independent memory exercise, exact tokens and unload verified"
                )
        except (Exception, KeyboardInterrupt) as error:
            detail = f"{type(error).__name__}: {error}"
            if isinstance(error, KeyboardInterrupt):
                detail = "interrupted; immutable attempt retained"
        raw.update(
            samples=[s.model_dump(mode="json") for s in samples],
            status=status,
            detail=detail,
            cleanup_passed=cleanup,
        )
        evidence = digest(raw)
        write_exclusive(directory / "evidence.json", raw)
        return JobResult(
            sweep_digest=spec.sweep_digest,
            job_id=job,
            candidate_id=plan.selected_candidate_id,
            status=status,
            warmups=tuple(samples[: spec.warmups]),
            sample=samples[spec.warmups] if len(samples) > spec.warmups else None,
            evidence_digest=evidence,
            memory_exclusion=exclusion,
            host_envelope_bytes=host_envelopes,
            device_envelope_bytes=device_envelopes,
            selected_health_passed=health,
            cleanup_passed=cleanup,
            detail=detail,
        )


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.stdin:
        process.stdin.close()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _clean(controls: list[control_pb2_grpc.WorkerControlStub], *, weights: bool) -> None:
    until = time.monotonic() + 10
    while True:
        reports = [c.GetMemoryReport(common_pb2.Empty(), timeout=5) for c in controls]
        if all(
            not (
                r.active_requests
                or r.reserved_cache_bytes
                or r.reserved_workspace_bytes
                or (weights and r.loaded_weight_bytes)
            )
            for r in reports
        ):
            return
        if time.monotonic() >= until:
            raise RuntimeError("native cleanup did not retire reservations")
        time.sleep(0.05)


def _health(
    session: DeploymentSession,
    controls: list[control_pb2_grpc.WorkerControlStub],
    spec: SweepSpec,
    sample: Sample,
) -> dict[str, object]:
    results: dict[str, object] = {}
    for fault in ("cancel", "deadline"):
        # Place the deadline during decode using this candidate's preceding sample.
        timeout = (
            (
                sample.client_arrivals_ms[0]
                + (sample.client_arrivals_ms[-1] - sample.client_arrivals_ms[0]) / 2
            )
            / 1000
            if fault == "deadline"
            else 180
        )
        received = 0
        rejected = False
        stream = session.generate(
            spec.reference.prompt_ids,
            maximum_new_tokens=spec.workload.output_tokens,
            stop_token_ids=[],
            timeout=max(0.001, timeout),
        )
        try:
            for event in stream:
                if event.HasField("token"):
                    received += 1
                    if fault == "cancel":
                        break
        except grpc.RpcError as error:
            if fault != "deadline" or error.code() != grpc.StatusCode.DEADLINE_EXCEEDED:
                raise
            rejected = True
        except RuntimeError as error:
            if fault != "deadline" or "DEADLINE_EXCEEDED" not in str(error):
                raise
            rejected = True
        finally:
            stream.close()
        if (fault == "cancel" and received != 1) or (
            fault == "deadline" and (not rejected or received == 0)
        ):
            raise RuntimeError("selected fault did not exercise the expected decode failure")
        _clean(controls, weights=False)
        recovery = tuple(
            e.token.token_id
            for e in session.generate(
                spec.reference.prompt_ids,
                maximum_new_tokens=min(4, spec.workload.output_tokens),
                stop_token_ids=[],
                timeout=180,
            )
            if e.HasField("token")
        )
        if recovery != spec.reference.generated_ids[: len(recovery)] or not recovery:
            raise RuntimeError("selected recovery output diverged")
        _clean(controls, weights=False)
        results[fault] = dict(received=received, recovered_ids=recovery, cleaned=True)
    return results
