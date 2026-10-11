"""Concurrency-two evidence: acceptance is derived from observations and cannot be claimed."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hllm_control.models import Backend, DType
from hllm_control.profiling.memory import MlxFitPolicy, PhysicalBudget
from hllm_control.profiling.models import (
    AllocatorSample,
    Environment,
    MemoryAmounts,
    ProcessMemoryObservation,
    digest,
)
from hllm_control.qualification.concurrent import (
    CleanupEvidence,
    ConcurrentServingEvidence,
    Configuration,
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
    assess_serving_envelope,
    make_evidence,
    read_evidence,
    write_evidence,
)

NOW = datetime(2026, 10, 11, 3, 0, tzinfo=UTC)
ORDER = ("mlx", "cuda")
PROMPT = (1, 4, 2, 8, 3)
EXPECTED = (7, 7, 3, 1)
WEIGHTS = {"mlx": MemoryAmounts(unified=1000), "cuda": MemoryAmounts(host=100, device=900)}
CACHE = {"mlx": MemoryAmounts(unified=40), "cuda": MemoryAmounts(host=4, device=36)}
WORK = {"mlx": MemoryAmounts(unified=10), "cuda": MemoryAmounts(host=1, device=9)}
CAPACITY = {"mlx": MemoryAmounts(unified=1200), "cuda": MemoryAmounts(host=200, device=1100)}


def environment(backend: Backend) -> Environment:
    return Environment(
        backend=backend,
        device_identity="dev",
        device_name="dev",
        backend_version="1",
        driver_version="1",
        allocator="alloc",
        allocator_config="{}",
        source_revision="src",
        binary_digest="b" * 64,
        compiler="cc",
        os="test",
    )


def reservation(worker: str, identifier: str, at: int, **flags: bool) -> RequestReservation:
    return RequestReservation(
        request_id=identifier,
        maximum_total_tokens=9,
        cache=CACHE[worker],
        workspace=WORK[worker],
        admitted_at_monotonic_ns=at,
        allocating=flags.get("allocating", False),
        running=flags.get("running", True),
        cancelled=flags.get("cancelled", False),
    )


def snapshot(
    worker: str,
    phase: str,
    at: int,
    live: tuple[str, ...],
    events: tuple[LifecycleEvent, ...],
    *,
    process_id: int = 42,
    available: bool = True,
    loaded: bool = True,
    allocating: tuple[str, ...] = (),
) -> WorkerSnapshot:
    rows = tuple(
        reservation(worker, identifier, at - 1, allocating=identifier in allocating)
        for identifier in live
    )
    total = MemoryAmounts()
    work = MemoryAmounts()
    for _ in rows:
        total = MemoryAmounts(
            **{n: getattr(total, n) + getattr(CACHE[worker], n) for n in MemoryAmounts.model_fields}
        )
        work = MemoryAmounts(
            **{n: getattr(work, n) + getattr(WORK[worker], n) for n in MemoryAmounts.model_fields}
        )
    footprint = 1100 if worker == "mlx" else None
    return WorkerSnapshot(
        phase=phase,
        unix_time=1.0 + at / 1e9,
        process_id=process_id if available else None,
        observed_at_monotonic_ns=at if available else None,
        observed_at_unix_ns=1_000 + at if available else None,
        loaded_weights=WEIGHTS[worker] if loaded else MemoryAmounts(),
        reserved_cache=total,
        reserved_workspace=work,
        active_request_ids=live,
        request_observations_available=available,
        requests=rows if available else (),
        events=events if available else (),
        events_total=max((e.sequence for e in events), default=0) if available else 0,
        peak_concurrent_requests=2 if available else 0,
        allocator=AllocatorSample(
            active_bytes=1000, cached_bytes=50, peak_bytes=1050, peak_scope="phase"
        ),
        process_memory=ProcessMemoryObservation(
            process_id=process_id,
            observed_at_unix_ns=1_000 + at,
            rss_bytes=500,
            rss_lifetime_peak_bytes=600,
            physical_footprint_bytes=footprint,
            physical_footprint_lifetime_peak_bytes=footprint,
        ),
    )


def event(kind: str, identifier: str, sequence: int, at: int, concurrent=(), reason=""):
    return LifecycleEvent(
        kind=kind,  # type: ignore[arg-type]
        request_id=identifier,
        sequence=sequence,
        at_monotonic_ns=at,
        concurrent_request_ids=tuple(concurrent),
        reason=reason,
    )


def timeline(worker: str, **options) -> WorkerObservations:
    """A serial round, one paired round and one cancel round, as the worker saw them."""
    pair_events = (
        event("admitted", "serial", 1, 100),
        event("retired", "serial", 2, 200, reason="completed"),
        event("admitted", "a", 3, 300),
        event("admitted", "b", 4, 310, concurrent=("a",)),
    )
    joint_live = ("a", "b")
    witnessed = ("b",)
    if options.get("one_worker_only") and worker == "cuda":
        pair_events = (*pair_events[:3], event("admitted", "b", 4, 400))
        witnessed = ()
    if options.get("queued_only"):
        # Only one request was ever admitted natively; the other waited in a queue.
        pair_events = pair_events[:3]
        joint_live = ("a",)
        witnessed = ()
    if options.get("allocating"):
        # The second admission happened, but its allocation never completed while the
        # first was live, so neither retirement lists the other.
        pair_events = (*pair_events[:3], event("admitted", "b", 4, 310))
        witnessed = ()
    after_pair = (
        *pair_events,
        event("retired", "a", 5, 500, concurrent=witnessed, reason="completed"),
        event("retired", "b", 6, 510, reason="completed"),
        event("admitted", "victim", 7, 600),
        event("admitted", "survivor", 8, 610, concurrent=("victim",)),
    )
    cancel_events = (
        *after_pair,
        event(
            "retired",
            "victim",
            9,
            650,
            concurrent=() if options.get("no_independent_release") else ("survivor",),
            reason="cancelled",
        ),
        event("retired", "survivor", 10, 700, reason="completed"),
    )
    snapshots = [
        snapshot(worker, "baseline", 50, (), (), loaded=False),
        snapshot(worker, "serial-0", 150, ("serial",), pair_events[:1]),
        snapshot(worker, "serial-0-cleanup", 250, (), pair_events[:2]),
        snapshot(
            worker,
            "paired-0-0",
            320,
            joint_live,
            pair_events,
            allocating=("b",) if options.get("allocating") else (),
        ),
        snapshot(worker, "paired-0-0-cleanup", 520, (), after_pair[:6]),
        snapshot(worker, "cancel-0-1", 620, ("victim", "survivor"), after_pair),
        snapshot(worker, "cancel-0-1", 660, ("survivor",), cancel_events[:9]),
        snapshot(worker, "cancel-0-1-cleanup", 720, (), cancel_events),
        snapshot(
            worker,
            "unloaded-0",
            800,
            (),
            cancel_events,
            loaded=False,
            process_id=43 if options.get("process_change") else 42,
            available=not options.get("dropped_telemetry"),
        ),
    ]
    return WorkerObservations(worker_id=worker, snapshots=tuple(snapshots))


def comparisons(**options) -> list[TokenComparison]:
    def row(identifier, role, generated, expected):
        return TokenComparison(
            request_id=identifier,
            role=role,
            generated_ids=generated,
            expected_ids=expected,
            exact=generated == expected,
            seconds=0.5,
        )

    victim = EXPECTED[:3]
    survivor = EXPECTED[:3] if options.get("incomplete_survivor") else EXPECTED
    return [
        row("serial", "serial", EXPECTED, EXPECTED),
        row("a", "paired", (9, 9, 9, 9) if options.get("mismatch") else EXPECTED, EXPECTED),
        row("b", "paired", EXPECTED, EXPECTED),
        row(
            "victim",
            "cancel-victim",
            EXPECTED[:2] if options.get("premature") else victim,
            EXPECTED[:2] if options.get("premature") else victim,
        ),
        row("survivor", "cancel-survivor", survivor, EXPECTED),
    ]


def build(**options) -> ConcurrentServingEvidence:
    request = RequestWorkload(
        prompt_ids=PROMPT,
        expected_ids=EXPECTED,
        prompt_digest=digest({"ids": list(PROMPT)}),
        expected_digest=digest({"ids": list(EXPECTED)}),
        reservation_tokens=9,
    )
    identity = Identity(
        source_digest="snapshot-1",
        manifest_digest="1" * 64,
        checkpoint_digest="c" * 64,
        plan_id="plan-1",
        plan_digest="d" * 64,
        reference_sha256="2" * 64,
        transport="loopback",
        driver="native-controller",
        worker_order=ORDER,
        workers=tuple(
            WorkerIdentity(
                worker_id=w,
                stage_index=i,
                backend=Backend.MLX if w == "mlx" else Backend.CUDA,
                environment=environment(Backend.MLX if w == "mlx" else Backend.CUDA),
                worker_binary_digest="3" * 64,
                endpoint=f"127.0.0.1:{50291 + i}",
                transport_mode="pageable",
                process_id=42,
            )
            for i, w in enumerate(ORDER)
        ),  # type: ignore[arg-type]
    )
    configuration = Configuration(
        workers=tuple(
            WorkerConfiguration(
                worker_id=w,
                admission_capacity=CAPACITY[w],
                maximum_active_requests=2,
                maximum_cached_tokens=18,
                maximum_decode_batch=1,
                prefill_chunk_tokens=0,
                request_observations=not options.get("no_capability"),
            )
            for w in ORDER
        ),  # type: ignore[arg-type]
        queue_capacity=0,
        controller_limit=2,
        mlx_fit_policy=MlxFitPolicy.FOOTPRINT,
        safety_fraction=0.1,
        extra_overhead_bytes=10,
        host_headroom_bytes=10,
        device_headroom_bytes=10,
    )
    budget = PhysicalBudget(available_bytes=5000, headroom_bytes=10, extra_overhead_bytes=10)
    measured_at = NOW - timedelta(hours=30 if options.get("stale") else 1)
    preconditions = Preconditions(
        baseline_report_sha256="4" * 64,
        started_at=NOW,
        profiles=tuple(
            ProfilePrecondition(
                worker_id=w,
                artifact_digest="a" * 64,
                measured_at=measured_at,
                requested_mlx_policy=MlxFitPolicy.FOOTPRINT if w == "mlx" else None,
                prelaunch_status="safe",
                prelaunch_reasons=(),
                host_budget=budget,
                device_budget=budget if w == "cuda" else None,
            )
            for w in ORDER
        ),  # type: ignore[arg-type]
        guard="1 GiB floor, 256 MiB/60 s swap-out, 30 minutes",
    )
    observations = Observations(
        workers=(timeline("mlx", **options), timeline("cuda", **options)),
        clock_alignment="per-worker steady clocks only",
    )
    physical = tuple(
        assess_serving_envelope(
            w,
            Backend.MLX if w == "mlx" else Backend.CUDA,
            observations.workers[i].snapshots,
            host=budget,
            device=budget if w == "cuda" else None,
            mlx_policy=MlxFitPolicy.FOOTPRINT,
            device_process_peak_bytes=None if options.get("no_device_samples") else 950,
        )
        for i, w in enumerate(ORDER)
    )
    guards = tuple(
        GuardOutcome(
            worker_id=w,
            outcome="stopped" if options.get("guard_stopped") else "normal",
            record_sha256="5" * 64,
            detail="swap-out" if options.get("guard_stopped") else "exited normally",
        )
        for w in ORDER
    )
    cleanup = tuple(
        CleanupEvidence(
            worker_id=w,
            verified=not options.get("unclean"),
            detail="no reservations" if not options.get("unclean") else "weights remain",
        )
        for w in ORDER
    )
    return make_evidence(
        identity=identity,
        workload=Workload(
            requests=(request, request),
            execution_dtype=DType.F16,
            kv_dtype=DType.F16,
            activation_dtype=DType.F16,
            sampling="greedy",
            timeout_seconds=60,
        ),
        configuration=configuration,
        preconditions=preconditions,
        observations=observations,
        rounds=[
            Round(cycle=0, index=0, kind="serial", request_ids=("serial",)),
            Round(cycle=0, index=0, kind="paired", request_ids=("a", "b")),
            Round(cycle=0, index=1, kind="cancel", request_ids=("victim", "survivor")),
        ],
        comparisons=comparisons(**options),
        physical=physical,  # type: ignore[arg-type]
        guards=guards,  # type: ignore[arg-type]
        cleanup=cleanup,  # type: ignore[arg-type]
        failure_reason=options.get("failure"),
    )


def test_complete_evidence_is_accepted_and_round_trips(tmp_path: Path) -> None:
    evidence = build()
    assert evidence.results.acceptance == "accepted", evidence.results.acceptance_reasons
    assert {(o.worker_id, o.request_ids) for o in evidence.results.overlap} == {
        ("mlx", ("a", "b")),
        ("cuda", ("a", "b")),
        ("mlx", ("victim", "survivor")),
        ("cuda", ("victim", "survivor")),
    }
    for interval in evidence.results.overlap:
        assert interval.start_monotonic_ns < interval.end_monotonic_ns
        assert interval.proof == "retirement-event"
    # Shared weights are counted once; two reservations are summed.
    mlx, cuda = evidence.results.logical_peaks
    assert mlx.combined == MemoryAmounts(unified=1000 + 2 * 40 + 2 * 10)
    assert cuda.combined == MemoryAmounts(host=100 + 2 * 5, device=900 + 2 * 45)
    assert all(p.status == "safe" for p in evidence.results.physical)
    assert evidence.results.physical[0].policy == "mlx-footprint-max-v1"
    assert evidence.results.physical[1].policy == "cuda-rss-device-v1"
    path = tmp_path / "evidence.json"
    write_evidence(path, evidence)
    assert read_evidence(path) == evidence
    with pytest.raises(FileExistsError):
        write_evidence(path, evidence)


@pytest.mark.parametrize(
    ("options", "status", "fragment"),
    [
        ({"queued_only": True}, "unknown", "overlap of a and b not observed"),
        ({"allocating": True}, "unknown", "overlap of a and b not observed"),
        ({"one_worker_only": True}, "unknown", "cuda: overlap of a and b not observed"),
        ({"premature": True}, "failed", "cancelled before 3 tokens"),
        ({"incomplete_survivor": True}, "failed", "survivor: continuation differs"),
        ({"mismatch": True}, "failed", "a: continuation differs"),
        ({"dropped_telemetry": True}, "unknown", "request observations missing"),
        ({"process_change": True}, "failed", "worker process changed"),
        ({"stale": True}, "failed", "profile stale or future-dated"),
        ({"no_independent_release": True}, "unknown", "independent retirement of victim"),
        ({"no_capability": True}, "unknown", "did not advertise request observations"),
        ({"no_device_samples": True}, "unknown", "CUDA physical process observations unavailable"),
        ({"guard_stopped": True}, "failed", "resource guard stopped"),
        ({"unclean": True}, "failed", "cleanup not verified"),
        ({"failure": "RuntimeError: worker lost"}, "failed", "run failed"),
    ],
)
def test_defects_are_never_accepted(options: dict, status: str, fragment: str) -> None:
    evidence = build(**options)
    assert evidence.results.acceptance == status, evidence.results.acceptance_reasons
    assert any(fragment in reason for reason in evidence.results.acceptance_reasons), (
        evidence.results.acceptance_reasons
    )


def test_overlap_is_proven_by_events_or_by_a_joint_snapshot() -> None:
    from hllm_control.qualification.concurrent import derive_overlap

    rounds = [Round(cycle=0, index=0, kind="paired", request_ids=("a", "b"))]
    # Events alone (no joint snapshot) prove overlap through the first retirement.
    without_snapshot = Observations(
        workers=(
            WorkerObservations(
                worker_id="mlx",
                snapshots=tuple(
                    s.model_copy(update={"active_request_ids": (), "requests": ()})
                    for s in timeline("mlx").snapshots
                ),
            ),
            timeline("cuda"),
        ),
        clock_alignment="test",
    )
    proofs = {(o.worker_id, o.proof) for o in derive_overlap(without_snapshot, rounds)}
    assert proofs == {("mlx", "retirement-event"), ("cuda", "retirement-event")}

    # A joint snapshot alone (retirements evicted from history) still proves overlap.
    def strip_retirements(w: WorkerObservations) -> WorkerObservations:
        return w.model_copy(
            update={
                "snapshots": tuple(
                    s.model_copy(
                        update={
                            "events": tuple(
                                e
                                for e in s.events
                                if not (e.kind == "retired" and e.request_id in ("a", "b"))
                            )
                        }
                    )
                    for s in w.snapshots
                )
            }
        )

    without_retirements = Observations(
        workers=(strip_retirements(timeline("mlx")), strip_retirements(timeline("cuda"))),
        clock_alignment="test",
    )
    proofs = {(o.worker_id, o.proof) for o in derive_overlap(without_retirements, rounds)}
    assert proofs == {("mlx", "joint-snapshot"), ("cuda", "joint-snapshot")}


def test_tampered_records_are_rejected_on_load(tmp_path: Path) -> None:
    evidence = build(queued_only=True)
    assert evidence.results.acceptance == "unknown"
    data = evidence.model_dump(mode="json")
    for mutate, message in (
        (lambda d: d["results"].__setitem__("acceptance", "accepted"), "digest mismatch"),
        (lambda d: d["identity"].__setitem__("plan_digest", "6" * 64), "digest mismatch"),
        (lambda d: d.__setitem__("extra", 1), "extra"),
    ):
        tampered = json.loads(json.dumps(data))
        mutate(tampered)
        with pytest.raises(ValueError, match=message):
            ConcurrentServingEvidence.model_validate(tampered)
    # Even with a recomputed digest, claimed acceptance must match the derivation.
    tampered = json.loads(json.dumps(data))
    tampered["results"]["acceptance"] = "accepted"
    tampered["results"]["acceptance_reasons"] = []
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = digest(tampered)
    with pytest.raises(ValueError, match="is not derived"):
        ConcurrentServingEvidence.model_validate(tampered)
    # Claimed overlap that the observations do not show is rejected the same way.
    tampered = json.loads(json.dumps(data))
    tampered["results"]["overlap"] = evidence.model_dump(mode="json")["results"]["overlap"] + [
        {
            "worker_id": "mlx",
            "request_ids": ["a", "b"],
            "start_monotonic_ns": 1,
            "end_monotonic_ns": 2,
            "admission_sequence": 4,
            "proof": "joint-snapshot",
            "detail": "snapshot paired-0-0",
        }
    ]
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = digest(tampered)
    with pytest.raises(ValueError, match="overlap intervals are not what the observations show"):
        ConcurrentServingEvidence.model_validate(tampered)


def test_serving_envelope_falls_back_and_refuses_unsafe_budgets() -> None:
    rows = timeline("mlx").snapshots
    budget = PhysicalBudget(available_bytes=5000, headroom_bytes=10, extra_overhead_bytes=10)
    footprint = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        rows,
        host=budget,
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    conservative = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        rows,
        host=budget,
        device=None,
        mlx_policy=MlxFitPolicy.CONSERVATIVE,
        device_process_peak_bytes=None,
    )
    assert footprint.host_peak_bytes == 1100 and footprint.policy == "mlx-footprint-max-v1"
    assert conservative.host_peak_bytes == 600 + 1050
    assert conservative.policy == "mlx-rss-plus-allocator-v1"
    # A snapshot without footprint telemetry falls back with a note.
    broken = (
        *rows[:1],
        rows[1].model_copy(
            update={
                "process_memory": rows[1].process_memory.model_copy(
                    update={"physical_footprint_bytes": None}
                )
                if rows[1].process_memory
                else None
            }
        ),
        *rows[2:],
    )
    fallback = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        broken,
        host=budget,
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    assert fallback.policy == "mlx-rss-plus-allocator-v1" and fallback.policy_notes
    tight = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        rows,
        host=budget.model_copy(update={"available_bytes": 1200}),
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    assert tight.status == "unsafe"  # 1100 + 110 + 10 + 10 > 1200
