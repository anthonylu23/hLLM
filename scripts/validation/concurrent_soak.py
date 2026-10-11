"""Bounded concurrency-two serving experiment producing `ConcurrentServingEvidence`.

Two requests are released from a barrier against explicitly selected, already running
workers started with `--request-observations on` and `--max-active-requests 2`. Each
worker's own request-lifecycle observations, not poll timing or HTTP submission, are
what establish overlap. The run refuses busy workers, requires a passing single-request
baseline report and fresh single-request profiles, re-checks physical fit before every
load, and writes one immutable evidence record whose acceptance is derived from its
contents. It claims no throughput and qualifies no concurrency beyond two.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import grpc
from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DeploymentPlan, ModelManifest
from hllm_control.profiling.memory import FitResult, PhysicalBudget
from hllm_control.profiling.models import (
    AllocatorSample,
    MemoryAmounts,
    MemoryMeasurement,
    ProcessMemoryObservation,
    ProfileArtifact,
    digest,
)
from hllm_control.proto import common_pb2, control_pb2, control_pb2_grpc, profile_pb2
from hllm_control.qualification.concurrent import (
    REQUIRED_CANCEL_PREFIX,
    CleanupEvidence,
    ConcurrentServingEvidence,
    Configuration,
    GuardOutcome,
    Identity,
    LifecycleEvent,
    Observations,
    PhysicalEnvelope,
    Preconditions,
    ProfilePrecondition,
    RequestReservation,
    RequestWorkload,
    Round,
    TokenComparison,
    WorkerConfiguration,
    WorkerIdentity,
    WorkerObservations,
    WorkerSnapshot,
    Workload,
    assess_serving_envelope,
    make_evidence,
    write_evidence,
)
from hllm_control.serialization import sha256_file

from scripts.validation.reload_soak import Worker, fresh_fit, reference_tokens, validate_profile

DOMAINS = {
    profile_pb2.MEMORY_DOMAIN_HOST: "host",
    profile_pb2.MEMORY_DOMAIN_DEVICE: "device",
    profile_pb2.MEMORY_DOMAIN_HOST_PINNED: "pinned",
    profile_pb2.MEMORY_DOMAIN_UNIFIED: "unified",
}
EVENT_KINDS: dict[int, Literal["admitted", "retired"]] = {
    control_pb2.RequestLifecycleEvent.REQUEST_LIFECYCLE_KIND_ADMITTED: "admitted",
    control_pb2.RequestLifecycleEvent.REQUEST_LIFECYCLE_KIND_RETIRED: "retired",
}
CLOCK_ALIGNMENT = (
    "Each worker's steady clock orders only its own admissions, retirements and snapshots; "
    "overlap is claimed per worker from those. Controller wall-clock times mark round "
    "boundaries only and never establish cross-host order."
)


class WorkerSpec(Worker):
    guard_record: Path | None = None  # resource_guard.py JSONL for this worker, if any.
    device_samples: Path | None = None  # memory_watch.py --cuda JSONL for CUDA workers.


def amounts(rows, cache: bool) -> MemoryAmounts:
    values: dict[str, int] = {}
    for row in rows:
        name = DOMAINS.get(row.domain)
        if name is not None:
            values[name] = int(row.cache_bytes if cache else row.workspace_bytes)
    return MemoryAmounts(**values)


def snapshot_from_proto(
    phase: str, memory: control_pb2.MemoryReport, metrics: control_pb2.WorkerMetrics
) -> WorkerSnapshot:
    usage = {DOMAINS[u.domain]: u for u in memory.domain_usage if u.domain in DOMAINS}
    available = memory.HasField("request_observations")
    observations = memory.request_observations
    allocator = None
    if metrics.HasField("allocator"):
        allocator = AllocatorSample(
            active_bytes=int(metrics.allocator.active_bytes),
            cached_bytes=int(metrics.allocator.cached_bytes),
            peak_bytes=int(metrics.allocator.peak_bytes),
            peak_scope="phase",
        )
    process_memory = None
    if metrics.HasField("process_memory") and metrics.process_memory.process_id:
        p = metrics.process_memory
        field = lambda name: int(getattr(p, name)) if p.HasField(name) else None  # noqa: E731
        process_memory = ProcessMemoryObservation(
            process_id=int(p.process_id),
            observed_at_unix_ns=int(p.observed_at_unix_ns),
            rss_bytes=field("rss_bytes"),
            rss_lifetime_peak_bytes=field("rss_lifetime_peak_bytes"),
            physical_footprint_bytes=field("physical_footprint_bytes"),
            physical_footprint_lifetime_peak_bytes=field("physical_footprint_lifetime_peak_bytes"),
        )
    return WorkerSnapshot(
        phase=phase,
        unix_time=time.time(),
        process_id=int(observations.process_id) if available else None,
        observed_at_monotonic_ns=int(observations.observed_at_monotonic_ns) if available else None,
        observed_at_unix_ns=int(observations.observed_at_unix_ns) if available else None,
        loaded_weights=MemoryAmounts(**{k: int(u.loaded_weight_bytes) for k, u in usage.items()}),
        reserved_cache=MemoryAmounts(**{k: int(u.reserved_cache_bytes) for k, u in usage.items()}),
        reserved_workspace=MemoryAmounts(
            **{k: int(u.reserved_workspace_bytes) for k, u in usage.items()}
        ),
        active_request_ids=tuple(memory.active_request_ids),
        request_observations_available=available,
        requests=tuple(
            RequestReservation(
                request_id=r.request_id,
                maximum_total_tokens=int(r.maximum_total_tokens),
                cache=amounts(r.memory, True),
                workspace=amounts(r.memory, False),
                admitted_at_monotonic_ns=int(r.admitted_at_monotonic_ns),
                allocating=r.allocating,
                running=r.running,
                cancelled=r.cancelled,
            )
            for r in observations.requests
        )
        if available
        else (),
        events=tuple(
            LifecycleEvent(
                kind=EVENT_KINDS[e.kind],
                request_id=e.request_id,
                sequence=int(e.sequence),
                at_monotonic_ns=int(e.at_monotonic_ns),
                concurrent_request_ids=tuple(e.concurrent_request_ids),
                reason=e.reason,
            )
            for e in observations.events
        )
        if available
        else (),
        events_total=int(observations.events_total) if available else 0,
        peak_concurrent_requests=int(observations.peak_concurrent_requests) if available else 0,
        allocator=allocator,
        process_memory=process_memory,
    )


def guard_outcome(worker_id: str, record: Path | None) -> GuardOutcome:
    if record is None or not record.is_file():
        return GuardOutcome(
            worker_id=worker_id, outcome="unavailable", record_sha256=None, detail="no guard record"
        )
    rows = []
    try:
        lines = record.read_text().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                # A guard that is still running may have been mirrored mid-write; only
                # the final line may be incomplete, and it is simply not yet evidence.
                if index != len(lines) - 1:
                    raise
    except (OSError, ValueError) as error:
        return GuardOutcome(
            worker_id=worker_id,
            outcome="unavailable",
            record_sha256=None,
            detail=f"unreadable guard record: {error}",
        )
    checksum = sha256_file(record)
    if not any(r.get("event") == "start" for r in rows):
        return GuardOutcome(
            worker_id=worker_id,
            outcome="unavailable",
            record_sha256=checksum,
            detail="guard record has no start event",
        )
    exits = [r for r in rows if r.get("event") == "exit"]
    if exits and exits[-1].get("reason"):
        return GuardOutcome(
            worker_id=worker_id,
            outcome="stopped",
            record_sha256=checksum,
            detail=str(exits[-1]["reason"]),
        )
    observations = sum(1 for r in rows if r.get("event") not in ("preflight", "start", "exit"))
    detail = (
        f"guard exited normally after {observations} observations"
        if exits
        else f"guard still running; {observations} observations without a stop condition"
    )
    return GuardOutcome(
        worker_id=worker_id, outcome="normal", record_sha256=checksum, detail=detail
    )


def device_peak(samples: Path | None, start: float, end: float) -> int | None:
    if samples is None or not samples.is_file():
        return None
    peak = None
    for line in samples.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("cuda_process_bytes")
        if value is None or not start <= float(row.get("unix_time", 0)) <= end:
            continue
        peak = int(value) if peak is None else max(peak, int(value))
    return peak


class Run:
    """Owns channels, the JSONL observation stream and the growing evidence inputs."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.manifest = ModelManifest.model_validate_json(args.manifest.read_text())
        self.plan = DeploymentPlan.model_validate_json(args.plan.read_text())
        if (
            self.plan.manifest_digest != self.manifest.manifest_digest
            or self.plan.execution_dtype.value != ("F32" if args.cpu_rehearsal else "F16")
            or self.plan.weight_dtype is not None
            or len(self.plan.stages) != 2
        ):
            raise ValueError(
                "deployment must be a two-stage plan in the selected uniform precision"
            )
        reference = json.loads(args.reference.read_text())
        self.ids, self.expected = reference_tokens(
            reference, self.manifest, cpu_rehearsal=args.cpu_rehearsal
        )
        self.checkpoint_digest: str = reference["checkpoint_digest"]
        if args.cancel_after >= len(self.expected):
            raise ValueError("cancel-after must leave the reference continuation incomplete")
        self.order = tuple(s.worker_id for s in self.plan.stages)
        specs = {
            k: WorkerSpec.model_validate(v) for k, v in json.loads(args.workers.read_text()).items()
        }
        if set(specs) != set(self.order):
            raise ValueError("worker names must exactly match plan stages")
        self.workers = {k: specs[k] for k in self.order}
        self.profiles = {
            k: ProfileArtifact.model_validate_json(w.memory_profile.read_text())
            for k, w in self.workers.items()
        }
        for k, a in self.profiles.items():
            if args.cpu_rehearsal and a.key.environment.backend != Backend.CPU:
                raise ValueError("CPU rehearsal requires CPU profiles for every worker")
            validate_profile(
                a,
                self.manifest,
                self.plan,
                k,
                self.checkpoint_digest,
                len(self.ids),
                len(self.expected),
            )
        self.channels = {k: grpc.insecure_channel(w.endpoint) for k, w in self.workers.items()}
        self.controls = {k: control_pb2_grpc.WorkerControlStub(c) for k, c in self.channels.items()}
        self.snapshots: dict[str, list[WorkerSnapshot]] = {k: [] for k in self.order}
        self.rounds: list[Round] = []
        self.comparisons: list[TokenComparison] = []
        self.preconditions: dict[str, ProfilePrecondition] = {}
        self.failure: str | None = None
        self.lock = threading.Lock()
        self.stream = args.output.with_suffix(".observations.jsonl").open("x", buffering=1)
        self.started = time.time()

    # ------------------------------------------------------------------ observation

    def snapshot(self, phase: str, workers: tuple[str, ...] | None = None) -> list[WorkerSnapshot]:
        rows = []
        for k in workers or self.order:
            control = self.controls[k]
            memory = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
            metrics = control.GetMetrics(common_pb2.Empty(), timeout=5)
            row = snapshot_from_proto(phase, memory, metrics)
            with self.lock:
                self.snapshots[k].append(row)
                self.stream.write(json.dumps({"worker": k, **row.model_dump(mode="json")}) + "\n")
            rows.append(row)
        return rows

    def clean(self, phase: str, unloaded: bool = False) -> None:
        until = time.monotonic() + 15
        while True:
            rows = self.snapshot(phase)
            idle = all(
                not r.active_request_ids
                and not sum(getattr(r.reserved_cache, n) for n in MemoryAmounts.model_fields)
                and not sum(getattr(r.reserved_workspace, n) for n in MemoryAmounts.model_fields)
                and (
                    not unloaded
                    or not sum(getattr(r.loaded_weights, n) for n in MemoryAmounts.model_fields)
                )
                for r in rows
            )
            if idle:
                return
            if time.monotonic() >= until:
                raise RuntimeError("workers are busy or reservations did not retire")
            time.sleep(0.1)

    def sampled(self, phase: str, body: Callable[[], None]) -> None:
        """Run `body` while polling each worker independently at a bounded rate."""
        done = threading.Event()
        errors: list[BaseException] = []

        def poll(worker: str) -> None:
            try:
                while not done.is_set():
                    self.snapshot(phase, workers=(worker,))
                    time.sleep(self.args.sample_interval)
            except BaseException as error:
                errors.append(error)

        samplers = [
            threading.Thread(target=poll, args=(k,), name=f"sampler-{k}", daemon=True)
            for k in self.order
        ]
        for sampler in samplers:
            sampler.start()
        try:
            body()
        finally:
            done.set()
            for sampler in samplers:
                sampler.join(timeout=10)
        if errors:
            raise RuntimeError(f"sampler failed: {errors[0]!r}")

    # --------------------------------------------------------------------- requests

    def generate(
        self, session: DeploymentSession, identifier: str, role: str, stop_after: int | None
    ) -> TokenComparison:
        expected = self.expected if stop_after is None else self.expected[:stop_after]
        start = time.monotonic()
        tokens: list[int] = []
        error = None
        stream = session.generate(
            self.ids,
            maximum_new_tokens=len(self.expected),
            stop_token_ids=[],
            timeout=self.args.timeout,
            request_id=identifier,
        )
        try:
            for event in stream:
                if event.HasField("token"):
                    tokens.append(event.token.token_id)
                if stop_after is not None and len(tokens) == stop_after:
                    break
        except BaseException as caught:
            error = f"{type(caught).__name__}: {caught}"
        finally:
            stream.close()
        return TokenComparison(
            request_id=identifier,
            role=role,  # type: ignore[arg-type]
            generated_ids=tuple(tokens),
            expected_ids=tuple(expected),
            exact=tokens == expected,
            seconds=time.monotonic() - start,
            error=error,
        )

    def pair(
        self, session: DeploymentSession, cycle: int, index: int, kind: str, ids: tuple[str, str]
    ) -> None:
        results: dict[str, TokenComparison] = {}
        barrier = threading.Barrier(2)
        roles = ("paired", "paired") if kind == "paired" else ("cancel-victim", "cancel-survivor")
        stops = (None, None) if kind == "paired" else (self.args.cancel_after, None)

        def one(identifier: str, role: str, stop_after: int | None) -> None:
            barrier.wait(timeout=30)
            results[identifier] = self.generate(session, identifier, role, stop_after)

        threads = [
            threading.Thread(target=one, args=(i, r, s), name=i)
            for i, r, s in zip(ids, roles, stops, strict=True)
        ]

        def body() -> None:
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.sampled(f"{kind}-{cycle}-{index}", body)
        self.rounds.append(Round(cycle=cycle, index=index, kind=kind, request_ids=ids))  # type: ignore[arg-type]
        for identifier in ids:
            comparison = results.get(identifier)
            if comparison is None:
                raise RuntimeError(f"{identifier}: request thread produced no result")
            self.comparisons.append(comparison)
            print(
                json.dumps(dict(request=identifier, role=comparison.role, exact=comparison.exact)),
                flush=True,
            )
            if comparison.error:
                raise RuntimeError(f"{identifier}: {comparison.error}")
            if not comparison.exact:
                raise RuntimeError(f"{identifier}: continuation differs from the reference")
        self.clean(f"{kind}-{cycle}-{index}-cleanup")

    # ------------------------------------------------------------------------ cycles

    def gate(self, cycle: int) -> None:
        for k, w in self.workers.items():
            result = fresh_fit(self.profiles[k], w, self.controls[k])
            fit = FitResult.model_validate(result["assessment"])
            if cycle == 0:
                self.preconditions[k] = ProfilePrecondition(
                    worker_id=k,
                    artifact_digest=self.profiles[k].artifact_digest,
                    measured_at=self.profiles[k].conditions.measured_at,
                    requested_mlx_policy=fit.requested_mlx_policy,
                    prelaunch_status=fit.status,
                    prelaunch_reasons=fit.reasons,
                    host_budget=PhysicalBudget.model_validate(result["host_budget"]),
                    device_budget=PhysicalBudget.model_validate(result["device_budget"])
                    if result["device_budget"]
                    else None,
                )
            self.stream.write(json.dumps({"worker": k, "phase": f"gate-{cycle}", **result}) + "\n")
            if fit.status != "safe":
                raise RuntimeError(f"fresh physical-fit gate rejected {k} before cycle {cycle}")

    def cycle(self, cycle: int) -> None:
        self.gate(cycle)
        with DeploymentSession(
            self.manifest, self.plan, {k: w.endpoint for k, w in self.workers.items()}
        ) as session:
            self.snapshot(f"after-load-{cycle}")
            serial = self.generate(session, f"c{cycle}-serial", "serial", None)
            self.rounds.append(
                Round(cycle=cycle, index=0, kind="serial", request_ids=(serial.request_id,))
            )
            self.comparisons.append(serial)
            print(
                json.dumps(dict(request=serial.request_id, role="serial", exact=serial.exact)),
                flush=True,
            )
            if serial.error or not serial.exact:
                raise RuntimeError(serial.error or "serial continuation differs from the reference")
            self.clean(f"serial-{cycle}-cleanup")
            for index in range(self.args.pairs):
                self.pair(
                    session,
                    cycle,
                    index,
                    "paired",
                    (f"c{cycle}-p{index}-a", f"c{cycle}-p{index}-b"),
                )
            self.pair(
                session,
                cycle,
                self.args.pairs,
                "cancel",
                (f"c{cycle}-victim", f"c{cycle}-survivor"),
            )
        self.clean(f"unloaded-{cycle}", unloaded=True)
        time.sleep(self.args.idle_seconds)
        self.clean(f"unloaded-idle-{cycle}", unloaded=True)

    # ---------------------------------------------------------------------- assembly

    def identity(self) -> tuple[Identity, Configuration]:
        workers = []
        configurations = []
        for stage, k in zip(self.plan.stages, self.order, strict=True):
            c = self.controls[k].GetCapabilities(common_pb2.Empty(), timeout=5).worker
            q = self.controls[k].GetQualificationState(common_pb2.Empty(), timeout=5)
            a = self.profiles[k]
            if c.maximum_active_requests < 2:
                raise ValueError(f"{k}: worker admits fewer than two active requests")
            if c.maximum_decode_batch != 1:
                raise ValueError(f"{k}: decode batching is a separate configuration; use batch 1")
            reservation = len(self.ids) + len(self.expected)
            if c.maximum_cached_tokens and c.maximum_cached_tokens < 2 * reservation:
                raise ValueError(f"{k}: cached-token limit cannot admit two reservations")
            if not c.supports_request_observations:
                raise ValueError(f"{k}: start the worker with --request-observations on")
            assert isinstance(a.measurement, MemoryMeasurement)
            workers.append(
                WorkerIdentity(
                    worker_id=k,
                    stage_index=stage.stage_index,
                    backend=a.key.environment.backend,
                    environment=a.key.environment,
                    worker_binary_digest=self.workers[k].binary_digest,
                    endpoint=self.workers[k].endpoint,
                    transport_mode=a.key.transport_mode,
                    process_id=int(q.process_id),
                )
            )
            configurations.append(
                WorkerConfiguration(
                    worker_id=k,
                    admission_capacity=a.measurement.admission_capacity,
                    maximum_active_requests=int(c.maximum_active_requests),
                    maximum_cached_tokens=int(c.maximum_cached_tokens),
                    maximum_decode_batch=int(c.maximum_decode_batch),
                    prefill_chunk_tokens=0,
                    request_observations=bool(c.supports_request_observations),
                )
            )
        first = self.workers[self.order[0]]
        policies = {w.mlx_fit_policy for w in self.workers.values()}
        identity = Identity(
            source_digest=self.args.source_digest,
            manifest_digest=self.manifest.manifest_digest,
            checkpoint_digest=self.checkpoint_digest,
            plan_id=self.plan.plan_id,
            plan_digest=self.plan.plan_digest,
            reference_sha256=sha256_file(self.args.reference),
            transport=self.args.transport,
            driver="native-controller",
            worker_order=(self.order[0], self.order[1]),
            workers=(workers[0], workers[1]),
        )
        configuration = Configuration(
            workers=(configurations[0], configurations[1]),
            queue_capacity=0,
            controller_limit=2,
            mlx_fit_policy=next(iter(policies)) if len(policies) == 1 else first.mlx_fit_policy,
            safety_fraction=0.10,
            extra_overhead_bytes=first.extra_overhead_bytes,
            host_headroom_bytes=first.host_headroom_bytes,
            device_headroom_bytes=first.device_headroom_bytes,
        )
        return identity, configuration

    def cleanup_evidence(self) -> list[CleanupEvidence]:
        rows = []
        for k, control in self.controls.items():
            try:
                memory = control.GetMemoryReport(common_pb2.Empty(), timeout=5)
            except grpc.RpcError as error:
                rows.append(
                    CleanupEvidence(worker_id=k, verified=False, detail=f"unreachable: {error}")
                )
                continue
            residual = (
                memory.active_requests
                or memory.reserved_cache_bytes
                or memory.reserved_workspace_bytes
                or memory.loaded_weight_bytes
            )
            rows.append(
                CleanupEvidence(
                    worker_id=k,
                    verified=not residual,
                    detail="no active requests, reservations or loaded weights"
                    if not residual
                    else "worker still holds reservations or weights",
                )
            )
        return rows

    def assemble(
        self, identity: Identity, configuration: Configuration
    ) -> ConcurrentServingEvidence:
        finished = time.time()
        observations = Observations(
            workers=tuple(
                WorkerObservations(worker_id=k, snapshots=tuple(self.snapshots[k]))
                for k in self.order
            ),  # type: ignore[arg-type]
            clock_alignment=CLOCK_ALIGNMENT,
        )
        physical: list[PhysicalEnvelope] = []
        for k, w in self.workers.items():
            precondition = self.preconditions[k]
            physical.append(
                assess_serving_envelope(
                    k,
                    self.profiles[k].key.environment.backend,
                    self.snapshots[k],
                    host=precondition.host_budget,
                    device=precondition.device_budget,
                    mlx_policy=w.mlx_fit_policy,
                    device_process_peak_bytes=device_peak(w.device_samples, self.started, finished),
                )
            )
        guards = [guard_outcome(k, w.guard_record) for k, w in self.workers.items()]
        cleanup = self.cleanup_evidence()
        request = RequestWorkload(
            prompt_ids=tuple(self.ids),
            expected_ids=tuple(self.expected),
            prompt_digest=digest({"ids": list(self.ids)}),
            expected_digest=digest({"ids": list(self.expected)}),
            reservation_tokens=len(self.ids) + len(self.expected),
        )
        return make_evidence(
            identity=identity,
            workload=Workload(
                requests=(request, request),
                weight_dtype=self.plan.weight_dtype,
                execution_dtype=self.plan.execution_dtype,
                kv_dtype=self.plan.execution_dtype,
                activation_dtype=self.plan.activation_dtype,
                sampling="greedy",
                timeout_seconds=self.args.timeout,
            ),
            configuration=configuration,
            preconditions=Preconditions(
                baseline_report_sha256=sha256_file(self.args.baseline_report),
                started_at=datetime.fromtimestamp(self.started, UTC),
                profiles=(self.preconditions[self.order[0]], self.preconditions[self.order[1]]),
                guard=self.args.guard_description,
            ),
            observations=observations,
            rounds=self.rounds,
            comparisons=self.comparisons,
            physical=(physical[0], physical[1]),
            guards=(guards[0], guards[1]),
            cleanup=(cleanup[0], cleanup[1]),
            failure_reason=self.failure,
        )

    def close(self) -> None:
        self.stream.close()
        for c in self.channels.values():
            c.close()


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "plan", "reference", "workers", "baseline-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--source-digest", required=True, help="snapshot digest of the running source"
    )
    parser.add_argument("--transport", required=True, help="how the controller reached the workers")
    parser.add_argument("--guard-description", required=True, help="guard settings in words")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--sample-interval", type=float, default=0.02)
    parser.add_argument(
        "--cancel-after",
        type=int,
        default=REQUIRED_CANCEL_PREFIX,
        help="tokens the cancelled request receives before cancellation (at least three)",
    )
    parser.add_argument(
        "--cpu-rehearsal",
        action="store_true",
        help="CPU/F32 procedure check only; never constitutes F16 or accelerator qualification",
    )
    args = parser.parse_args(argv)
    if (
        not 1 <= args.cycles <= 8
        or not 1 <= args.pairs <= 8
        or not 0 <= args.idle_seconds <= 60
        or not 0 < args.timeout <= 600
        or not 0.005 <= args.sample_interval <= 1
        or not REQUIRED_CANCEL_PREFIX <= args.cancel_after <= 64
    ):
        parser.error(
            "bounded cycles/pairs (1..8), idle (0..60s), timeout (0..600s), sampling (5ms..1s), "
            f"cancel-after ({REQUIRED_CANCEL_PREFIX}..64)"
        )
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.touch(exist_ok=False)
    args.output.unlink()  # Reserve the name early; write_evidence recreates it exclusively.
    run = Run(args)
    identity = configuration = None
    try:
        try:
            identity, configuration = run.identity()
            run.clean("baseline", unloaded=True)
            for cycle in range(args.cycles):
                run.cycle(cycle)
        except BaseException as error:
            run.failure = f"{type(error).__name__}: {error}"
            raise
        finally:
            if identity is not None and configuration is not None:
                evidence = run.assemble(identity, configuration)
                write_evidence(args.output, evidence)
                print(
                    json.dumps(
                        dict(
                            acceptance=evidence.results.acceptance,
                            reasons=list(evidence.results.acceptance_reasons),
                            evidence_digest=evidence.evidence_digest,
                        )
                    ),
                    flush=True,
                )
    finally:
        run.close()


if __name__ == "__main__":
    main()
