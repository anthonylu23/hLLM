"""Bounded concurrency-two serving experiment producing `ConcurrentServingEvidence`.

Two requests are released from a barrier against explicitly selected, already running
workers started with `--request-observations on` and `--max-active-requests 2`. Each
worker's own request-lifecycle observations, not poll timing or HTTP submission, are
what establish overlap. The run refuses busy workers, validates the passing
single-request baseline report against this plan, reference and these binaries before
any load, requires fresh single-request profiles, re-checks physical fit before every
load and keeps each cycle's live budget, and writes one immutable evidence record whose
acceptance is derived from its contents. It claims no throughput and qualifies no
concurrency beyond two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
    BaselineSummary,
    CleanupEvidence,
    ConcurrentServingEvidence,
    Configuration,
    CycleGate,
    DeviceSample,
    GuardOutcome,
    Identity,
    LifecycleEvent,
    Observations,
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
    make_evidence,
    write_evidence,
)
from hllm_control.serialization import sha256_file

if __package__ in (None, ""):  # `python scripts/validation/concurrent_soak.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.validation.reload_soak import (
    Worker,
    fresh_fit,
    reference_tokens,
    validate_profile,
)

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
    phase: str,
    cycle: int | None,
    memory: control_pb2.MemoryReport,
    metrics: control_pb2.WorkerMetrics,
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
        cycle=cycle,
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


def guard_outcome(
    worker_id: str, record: Path | None, snapshot: Path | None = None
) -> GuardOutcome:
    """Extract what a `resource_guard.py` record shows; the outcome is derived later.

    The start event's PID, the observation span and the exit event are what let the
    record validator bind the guard to this worker and this experiment window.

    A guard is still running when the evidence is assembled (it must outlive the
    experiment), so its file keeps growing. The record is read exactly once; the hash
    and every parsed field come from those bytes, and `snapshot`, when given, receives
    the same bytes so the recorded hash can be reproduced later.
    """

    def unavailable(detail: str, checksum: str | None = None) -> GuardOutcome:
        return GuardOutcome(
            worker_id=worker_id,
            record_sha256=checksum,
            process_id=None,
            started_unix_time=None,
            last_observed_unix_time=None,
            exited_unix_time=None,
            observations=0,
            exit_reason=None,
            returncode=None,
            detail=detail,
            outcome="unavailable",
            reasons=(),
        )

    if record is None or not record.is_file():
        return unavailable("no guard record")
    try:
        data = record.read_bytes()
    except OSError as error:
        return unavailable(f"unreadable guard record: {error}")
    checksum = hashlib.sha256(data).hexdigest()
    if snapshot is not None:
        with snapshot.open("xb") as copy:
            copy.write(data)
    rows = []
    try:
        lines = data.decode().splitlines()
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
    except ValueError as error:
        return unavailable(f"unreadable guard record: {error}", checksum)
    starts = [r for r in rows if r.get("event") == "start"]
    if len(starts) != 1 or not isinstance(starts[0].get("pid"), int):
        return unavailable("guard record has no single start event with a PID", checksum)
    observations = [
        r for r in rows if r.get("event") not in ("preflight", "start", "exit") and "unix_time" in r
    ]
    exits = [r for r in rows if r.get("event") == "exit"]
    exit_ = exits[-1] if exits else None
    try:
        return GuardOutcome(
            worker_id=worker_id,
            record_sha256=checksum,
            process_id=int(starts[0]["pid"]),
            started_unix_time=float(starts[0]["unix_time"]),
            last_observed_unix_time=max(
                (float(r["unix_time"]) for r in observations), default=None
            ),
            exited_unix_time=float(exit_["unix_time"]) if exit_ else None,
            observations=len(observations),
            exit_reason=str(exit_["reason"]) if exit_ and exit_.get("reason") else None,
            returncode=int(exit_["returncode"])
            if exit_ and exit_.get("returncode") is not None
            else None,
            detail=f"guard exited after {len(observations)} observations"
            if exit_
            else f"guard still running after {len(observations)} observations",
            outcome="unavailable",
            reasons=(),
        )
    except (KeyError, TypeError, ValueError) as error:
        return unavailable(f"malformed guard record: {error}", checksum)


def device_samples(samples: Path | None, start: float, end: float) -> tuple[DeviceSample, ...]:
    """Every `memory_watch.py --cuda` row inside the experiment window, PID included.

    Rows are recorded as sampled; the validator checks each PID against the worker
    identity and scores each cycle's window separately. The sampler is still running, so
    an incomplete final line is not yet evidence and is skipped; any other malformed
    line makes the whole file no samples (unknown), never a peak.
    """
    if samples is None or not samples.is_file():
        return ()
    rows: list[DeviceSample] = []
    try:
        lines = samples.read_text().splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                if index == len(lines) - 1:
                    break
                raise
            if not start <= float(row["unix_time"]) <= end:
                continue
            value = row.get("cuda_process_bytes")
            rows.append(
                DeviceSample(
                    unix_time=float(row["unix_time"]),
                    process_id=int(row["pid"]),
                    cuda_process_bytes=int(value) if value is not None else None,
                )
            )
    except (OSError, KeyError, TypeError, ValueError):
        return ()
    return tuple(rows)


def guard_snapshot_path(output: Path, worker_id: str) -> Path:
    """Where the guard bytes behind `GuardOutcome.record_sha256` are preserved."""
    return output.with_suffix(f".guard-{worker_id}.jsonl")


def validate_baseline(
    path: Path, *, plan: DeploymentPlan, reference: Path, workers: dict, cpu_rehearsal: bool
) -> BaselineSummary:
    """Parse the single-request report and bind it to this run before anything loads."""
    try:
        report = json.loads(path.read_text())
        summary = BaselineSummary(
            scope=report["scope"],
            completed=bool(report["completed"]),
            plan_digest=report["plan"]["plan_digest"],
            reference_sha256=report["reference_sha256"],
            worker_binary_digests=tuple(
                (k, report["workers"][k]["binary_digest"]) for k in workers
            ),
            cycles=len(report["cycles"]),
            requests=sum(len(c["requests"]) for c in report["cycles"]),
            all_exact=all(
                r.get("exact_reference_match") is True
                for c in report["cycles"]
                for r in c["requests"]
            )
            and all(
                c.get("cancellation", {}).get("exact_prefix_match") is True
                for c in report["cycles"]
            ),
            finished_unix_time=float(report["finished_unix_time"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"baseline report is not a usable single-request report: {error}"
        ) from error
    problems = []
    if not summary.completed or not summary.all_exact:
        problems.append("did not complete with exact outputs")
    if summary.scope != ("cpu-rehearsal" if cpu_rehearsal else "f16-reload-qualification"):
        problems.append(f"has scope {summary.scope!r}")
    if summary.plan_digest != plan.plan_digest:
        problems.append("was produced for a different plan")
    if summary.reference_sha256 != sha256_file(reference):
        problems.append("was produced against a different reference")
    if summary.worker_binary_digests != tuple((k, w.binary_digest) for k, w in workers.items()):
        problems.append("was produced on different worker binaries")
    if problems:
        raise ValueError("baseline report " + "; ".join(problems))
    return summary


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
        self.baseline = validate_baseline(
            args.baseline_report,
            plan=self.plan,
            reference=args.reference,
            workers=self.workers,
            cpu_rehearsal=args.cpu_rehearsal,
        )
        self.channels = {k: grpc.insecure_channel(w.endpoint) for k, w in self.workers.items()}
        self.controls = {k: control_pb2_grpc.WorkerControlStub(c) for k, c in self.channels.items()}
        self.snapshots: dict[str, list[WorkerSnapshot]] = {k: [] for k in self.order}
        self.rounds: list[Round] = []
        self.comparisons: list[TokenComparison] = []
        self.gates: list[CycleGate] = []
        self.current_cycle: int | None = None
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
            row = snapshot_from_proto(phase, self.current_cycle, memory, metrics)
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
        self.current_cycle = cycle
        for k, w in self.workers.items():
            result = fresh_fit(self.profiles[k], w, self.controls[k])
            fit = FitResult.model_validate(result["assessment"])
            self.gates.append(
                CycleGate(
                    worker_id=k,
                    cycle=cycle,
                    sampled_at=datetime.now(UTC),
                    status=fit.status,
                    reasons=fit.reasons,
                    host_budget=PhysicalBudget.model_validate(result["host_budget"]),
                    device_budget=PhysicalBudget.model_validate(result["device_budget"])
                    if result["device_budget"]
                    else None,
                )
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
        # Every fit gate and envelope is scored with each worker's own allowances; the
        # record states them once, so they must be the same for both workers.
        allowances = {
            (w.extra_overhead_bytes, w.host_headroom_bytes, w.device_headroom_bytes)
            for w in self.workers.values()
        }
        if len(allowances) != 1:
            raise ValueError(
                "workers must share extra overhead, host headroom and device headroom; "
                "the record states one configuration for both"
            )
        first = self.workers[self.order[0]]
        policies = {
            w.mlx_fit_policy
            for k, w in self.workers.items()
            if self.profiles[k].key.environment.backend == Backend.MLX
        }
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
            safety_fraction=PhysicalBudget.model_fields["safety_fraction"].default,
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
                WorkerObservations(
                    worker_id=k,
                    snapshots=tuple(self.snapshots[k]),
                    device_samples=device_samples(w.device_samples, self.started, finished),
                )
                for k, w in self.workers.items()
            ),  # type: ignore[arg-type]
            clock_alignment=CLOCK_ALIGNMENT,
        )
        guards = [
            guard_outcome(k, w.guard_record, guard_snapshot_path(self.args.output, k))
            for k, w in self.workers.items()
        ]
        cleanup = self.cleanup_evidence()
        request = RequestWorkload(
            prompt_ids=tuple(self.ids),
            expected_ids=tuple(self.expected),
            prompt_digest=digest({"ids": list(self.ids)}),
            expected_digest=digest({"ids": list(self.expected)}),
            reservation_tokens=len(self.ids) + len(self.expected),
        )
        profiles = tuple(
            ProfilePrecondition(
                worker_id=k,
                artifact_digest=a.artifact_digest,
                measured_at=a.conditions.measured_at,
                requested_mlx_policy=self.workers[k].mlx_fit_policy
                if a.key.environment.backend == Backend.MLX
                else None,
            )
            for k, a in self.profiles.items()
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
                baseline=self.baseline,
                started_at=datetime.fromtimestamp(self.started, UTC),
                profiles=(profiles[0], profiles[1]),
                gates=tuple(self.gates),
                guard=self.args.guard_description,
            ),
            observations=observations,
            rounds=self.rounds,
            comparisons=self.comparisons,
            guards=(guards[0], guards[1]),
            cleanup=(cleanup[0], cleanup[1]),
            finished_at=datetime.fromtimestamp(finished, UTC),
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
    """Refuse bad inputs silently; after contacting workers, always leave a record.

    Input refusals (profiles, baseline, reference) happen before any worker is
    contacted and write nothing. A failure while establishing worker identity (an
    unreachable peer, a worker without request observations) writes
    `<output>.failure.json`. Every later failure writes a partial evidence record
    whose `failure_reason` names it.
    """
    args = parse(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():  # Fail before any work; write_evidence creates it exclusively.
        raise FileExistsError(args.output)
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
                finalize(run, args.output, identity, configuration)
            elif run.failure is not None:
                note = dict(run_failure=run.failure, stage="worker identity", rounds=0)
                try:
                    with args.output.with_suffix(".failure.json").open("x") as f:
                        f.write(json.dumps(note, indent=2) + "\n")
                except OSError as error:  # Never mask the run's own failure.
                    print(f"could not write failure note: {error}", file=sys.stderr)
    finally:
        run.close()


def finalize(run: Run, output: Path, identity: Identity, configuration: Configuration) -> None:
    """Write the evidence, or a failure note beside it; never raise over the run's error."""
    try:
        evidence = run.assemble(identity, configuration)
        write_evidence(output, evidence)
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
    except Exception as error:  # The original failure must survive assembly errors.
        note = dict(
            run_failure=run.failure,
            assembly_error=f"{type(error).__name__}: {error}",
            rounds=len(run.rounds),
            snapshots={k: len(v) for k, v in run.snapshots.items()},
        )
        output.with_suffix(".failure.json").write_text(json.dumps(note, indent=2) + "\n")
        print(json.dumps(dict(acceptance="failed", **note)), flush=True)
        if run.failure is None:
            raise


if __name__ == "__main__":
    main()
