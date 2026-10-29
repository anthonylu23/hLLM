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
    assess_serving_envelope,
    classify_guard,
    derive_overlap,
    make_evidence,
    read_evidence,
    write_evidence,
)

NOW = datetime(2026, 10, 11, 3, 0, tzinfo=UTC)
GATE_AT = NOW - timedelta(minutes=1)
FINISHED = NOW + timedelta(minutes=10)
ORDER = ("mlx", "cuda")
PROMPT = (1, 4, 2, 8, 3)
EXPECTED = (7, 7, 3, 1)
WEIGHTS = {"mlx": MemoryAmounts(unified=1000), "cuda": MemoryAmounts(host=100, device=900)}
CACHE = {"mlx": MemoryAmounts(unified=40), "cuda": MemoryAmounts(host=4, device=36)}
WORK = {"mlx": MemoryAmounts(unified=10), "cuda": MemoryAmounts(host=1, device=9)}
CAPACITY = {"mlx": MemoryAmounts(unified=1200), "cuda": MemoryAmounts(host=200, device=1100)}
BUDGET = PhysicalBudget(available_bytes=5000, headroom_bytes=10, extra_overhead_bytes=10)
TIGHT = BUDGET.model_copy(update={"available_bytes": 1200})  # 1100 + 110 + 10 + 10 > 1200


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
    cycle: int | None = 0,
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
        cycle=cycle,
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
    if options.get("simultaneous"):
        # Both reserved before either allocation completed: neither admission lists a
        # peer, but the joint snapshot and the retirement witness still show overlap.
        pair_events = (*pair_events[:3], event("admitted", "b", 4, 310))
        if options.get("simultaneous") == "snapshot-only":
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
            reason=options.get(
                "victim_reason", "completed" if options.get("victim_completed") else "cancelled"
            ),
        ),
    )
    if not options.get("survivor_unfinished"):
        cancel_events = (
            *cancel_events,
            event(
                "retired", "survivor", 10, 700, reason=options.get("survivor_reason", "completed")
            ),
        )
    process_id = 43 if options.get("other_process") else 42
    snapshots = [
        snapshot(worker, "baseline", 50, (), (), loaded=False, cycle=None, process_id=process_id),
        snapshot(worker, "serial-0", 150, ("serial",), pair_events[:1], process_id=process_id),
        snapshot(worker, "serial-0-cleanup", 250, (), pair_events[:2], process_id=process_id),
        snapshot(
            worker,
            "paired-0-0",
            320,
            joint_live,
            pair_events,
            allocating=("b",) if options.get("allocating") else (),
            process_id=process_id,
        ),
        snapshot(worker, "paired-0-0-cleanup", 520, (), after_pair[:6], process_id=process_id),
        snapshot(
            worker, "cancel-0-1", 620, ("victim", "survivor"), after_pair, process_id=process_id
        ),
        snapshot(
            worker, "cancel-0-1", 660, ("survivor",), cancel_events[:9], process_id=process_id
        ),
        snapshot(worker, "cancel-0-1-cleanup", 720, (), cancel_events, process_id=process_id),
        snapshot(
            worker,
            "unloaded-0",
            800,
            (),
            cancel_events,
            loaded=False,
            process_id=43 if options.get("process_change") else process_id,
            available=not options.get("dropped_telemetry"),
        ),
    ]
    if options.get("second_cycle"):
        snapshots.append(snapshot(worker, "serial-1", 900, ("serial",), pair_events[:1], cycle=1))
    if options.get("no_snapshots") and worker == "cuda":
        snapshots = []
    samples = ()
    if worker == "cuda" and not options.get("no_device_samples"):
        samples = (
            DeviceSample(
                unix_time=NOW.timestamp() + 30,
                process_id=7 if options.get("device_other_process") else 42,
                cuda_process_bytes=950,
            ),
            DeviceSample(unix_time=NOW.timestamp() + 31, process_id=42, cuda_process_bytes=None),
        )
    return WorkerObservations(worker_id=worker, snapshots=tuple(snapshots), device_samples=samples)


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

    if options.get("unsealed"):
        # Internally consistent, but not what the sealed workload expects.
        other = (99, 99, 99, 99)
        return [
            row("serial", "serial", other, other),
            row("a", "paired", other, other),
            row("b", "paired", other, other),
            row("victim", "cancel-victim", other[:3], other[:3]),
            row("survivor", "cancel-survivor", other, other),
        ]
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


def guard(worker: str, **options) -> GuardOutcome:
    started = NOW.timestamp() - 120
    last = FINISHED.timestamp() - 2
    exited = None
    returncode = None
    if options.get("guard_late_start"):
        started = NOW.timestamp() + 5
    if options.get("guard_stale"):
        last = FINISHED.timestamp() - 60
    if options.get("guard_exited_early"):
        exited = FINISHED.timestamp() - 30
        returncode = 1
    return GuardOutcome(
        worker_id=worker,
        record_sha256="5" * 64,
        process_id=999 if options.get("guard_other_process") else 42,
        started_unix_time=started,
        last_observed_unix_time=last,
        exited_unix_time=exited,
        observations=10,
        exit_reason="swap-out" if options.get("guard_stopped") else None,
        returncode=returncode,
        detail="guard still running after 10 observations",
        outcome="unavailable",
        reasons=(),
    )


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
    measured_at = NOW - timedelta(hours=30 if options.get("stale") else 1)
    gates = [
        CycleGate(
            worker_id=w,
            cycle=0,
            sampled_at=GATE_AT,
            status="unsafe" if options.get("gate_unsafe") else "safe",
            reasons=("host/unified: cap exceeds availability",)
            if options.get("gate_unsafe")
            else (),
            host_budget=TIGHT if options.get("tight_budget") else BUDGET,
            device_budget=BUDGET if w == "cuda" else None,
        )
        for w in ORDER
    ]
    if options.get("second_cycle"):
        gates += [
            CycleGate(
                worker_id=w,
                cycle=1,
                sampled_at=NOW + timedelta(minutes=5),
                status="safe",
                reasons=(),
                host_budget=TIGHT if w == "mlx" else BUDGET,
                device_budget=BUDGET if w == "cuda" else None,
            )
            for w in ORDER
        ]
    baseline = BaselineSummary(
        scope="f16-reload-qualification",
        completed=not options.get("baseline_incomplete"),
        plan_digest="e" * 64 if options.get("baseline_other_plan") else "d" * 64,
        reference_sha256="2" * 64,
        worker_binary_digests=tuple(
            (w, "0" * 64 if options.get("baseline_other_binaries") else "3" * 64) for w in ORDER
        ),
        cycles=6,
        requests=24,
        all_exact=True,
        finished_unix_time=NOW.timestamp() - 3600,
    )
    preconditions = Preconditions(
        baseline_report_sha256="4" * 64,
        baseline=baseline,
        started_at=NOW,
        profiles=tuple(
            ProfilePrecondition(
                worker_id=w,
                artifact_digest="a" * 64,
                measured_at=measured_at,
                requested_mlx_policy=MlxFitPolicy.FOOTPRINT if w == "mlx" else None,
            )
            for w in ORDER
        ),  # type: ignore[arg-type]
        gates=() if options.get("no_gate") else tuple(gates),
        guard="1 GiB floor, 256 MiB/60 s swap-out, 30 minutes",
    )
    observations = Observations(
        workers=(timeline("mlx", **options), timeline("cuda", **options)),
        clock_alignment="per-worker steady clocks only",
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
        guards=(guard("mlx", **options), guard("cuda", **options)),
        cleanup=cleanup,  # type: ignore[arg-type]
        finished_at=FINISHED,
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
        assert interval.admission_sequence in (4, 8)
    # Shared weights are counted once; two reservations are summed.
    mlx, cuda = evidence.results.logical_peaks
    assert mlx.combined == MemoryAmounts(unified=1000 + 2 * 40 + 2 * 10)
    assert cuda.combined == MemoryAmounts(host=100 + 2 * 5, device=900 + 2 * 45)
    # One envelope per worker and gated cycle, derived from the record itself.
    assert [(p.worker_id, p.cycle, p.status) for p in evidence.results.physical] == [
        ("mlx", 0, "safe"),
        ("cuda", 0, "safe"),
    ]
    assert evidence.results.physical[0].policy == "mlx-footprint-max-v1"
    assert evidence.results.physical[1].policy == "cuda-rss-device-v1"
    assert evidence.results.physical[1].device_peak_bytes == 1050  # max(950, allocator)
    assert all(g.outcome == "normal" and not g.reasons for g in evidence.results.guards)
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
        ({"unsealed": True}, "failed", "expected tokens differ from the sealed reference"),
        ({"unsealed": True}, "failed", "victim: expected tokens are not a proper prefix"),
        ({"dropped_telemetry": True}, "unknown", "request observations missing"),
        ({"process_change": True}, "failed", "worker process differs from the initial identity"),
        ({"other_process": True}, "failed", "worker process differs from the initial identity"),
        ({"device_other_process": True}, "failed", "device observations belong to another"),
        ({"stale": True}, "failed", "profile stale or future-dated"),
        ({"no_independent_release": True}, "unknown", "independent retirement of victim"),
        (
            {"victim_completed": True},
            "unknown",
            "no cancelled retirement of victim observed (it retired as completed)",
        ),
        # A deadline or an error is not evidence that cancellation was independent.
        ({"victim_reason": "expired"}, "unknown", "(it retired as expired)"),
        ({"victim_reason": "failed"}, "unknown", "(it retired as failed)"),
        ({"survivor_reason": "failed"}, "unknown", "completed retirement of survivor"),
        ({"survivor_unfinished": True}, "unknown", "completed retirement of survivor"),
        ({"no_capability": True}, "unknown", "did not advertise request observations"),
        ({"no_device_samples": True}, "unknown", "CUDA physical process observations unavailable"),
        ({"no_snapshots": True}, "unknown", "cuda: no snapshots recorded"),
        ({"no_gate": True}, "unknown", "no fit gate recorded before the first cycle"),
        ({"gate_unsafe": True}, "failed", "fit gate before cycle 0 was unsafe"),
        ({"tight_budget": True}, "failed", "mlx cycle 0: host/unified: serving envelope"),
        ({"second_cycle": True}, "failed", "mlx cycle 1: host/unified: serving envelope"),
        ({"baseline_incomplete": True}, "failed", "baseline single-request report was incomplete"),
        ({"baseline_other_plan": True}, "failed", "produced for a different plan"),
        ({"baseline_other_binaries": True}, "failed", "different worker binaries"),
        ({"guard_stopped": True}, "failed", "guard stopped the worker: swap-out"),
        ({"guard_exited_early": True}, "failed", "exited before the experiment finished"),
        ({"guard_other_process": True}, "unknown", "guard watched process 999"),
        ({"guard_late_start": True}, "unknown", "guard started after the experiment began"),
        ({"guard_stale": True}, "unknown", "guard observations end"),
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


def test_second_cycle_is_scored_against_its_own_budget() -> None:
    evidence = build(second_cycle=True)
    by_cycle = {(p.worker_id, p.cycle): p.status for p in evidence.results.physical}
    assert by_cycle == {
        ("mlx", 0): "safe",
        ("mlx", 1): "unsafe",
        ("cuda", 0): "safe",
        ("cuda", 1): "unknown",  # No device samples fell in the second window.
    }


def test_requests_that_reserve_together_still_show_overlap() -> None:
    witnessed = build(simultaneous=True)
    assert witnessed.results.acceptance == "accepted", witnessed.results.acceptance_reasons
    intervals = {(o.worker_id, o.request_ids): o for o in witnessed.results.overlap}
    interval = intervals[("mlx", ("a", "b"))]
    assert interval.proof == "retirement-event" and interval.start_monotonic_ns == 310
    assert "reserved together" in interval.detail
    polled = build(simultaneous="snapshot-only")
    assert polled.results.acceptance == "accepted", polled.results.acceptance_reasons
    interval = {(o.worker_id, o.request_ids): o for o in polled.results.overlap}[
        ("mlx", ("a", "b"))
    ]
    assert interval.proof == "joint-snapshot" and interval.start_monotonic_ns == 310


def test_overlap_is_proven_by_events_or_by_a_joint_snapshot() -> None:
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
    def strip(w: WorkerObservations, kinds: tuple[str, ...]) -> WorkerObservations:
        return w.model_copy(
            update={
                "snapshots": tuple(
                    s.model_copy(
                        update={
                            "events": tuple(
                                e
                                for e in s.events
                                if not (e.kind in kinds and e.request_id in ("a", "b"))
                            )
                        }
                    )
                    for s in w.snapshots
                )
            }
        )

    without_retirements = Observations(
        workers=(strip(timeline("mlx"), ("retired",)), strip(timeline("cuda"), ("retired",))),
        clock_alignment="test",
    )
    proofs = {(o.worker_id, o.proof) for o in derive_overlap(without_retirements, rounds)}
    assert proofs == {("mlx", "joint-snapshot"), ("cuda", "joint-snapshot")}
    # Even admissions evicted: the joint snapshot's rows carry the admission times.
    nothing = Observations(
        workers=(
            strip(timeline("mlx"), ("retired", "admitted")),
            strip(timeline("cuda"), ("retired", "admitted")),
        ),
        clock_alignment="test",
    )
    intervals = derive_overlap(nothing, rounds)
    assert {o.proof for o in intervals} == {"joint-snapshot"}
    assert all(o.admission_sequence is None and o.start_monotonic_ns == 319 for o in intervals)


def test_tampered_records_are_rejected_on_load(tmp_path: Path) -> None:
    evidence = build(queued_only=True)
    assert evidence.results.acceptance == "unknown"
    data = evidence.model_dump(mode="json")

    def resealed(mutate) -> dict:
        tampered = json.loads(json.dumps(data))
        mutate(tampered)
        tampered.pop("evidence_digest")
        tampered["evidence_digest"] = digest(tampered)
        return tampered

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
    with pytest.raises(ValueError, match="is not derived"):
        ConcurrentServingEvidence.model_validate(
            resealed(lambda d: d["results"].update(acceptance="accepted", acceptance_reasons=[]))
        )
    # Claimed overlap that the observations do not show is rejected the same way.
    extra = {
        "worker_id": "mlx",
        "request_ids": ["a", "b"],
        "start_monotonic_ns": 1,
        "end_monotonic_ns": 2,
        "admission_sequence": 4,
        "proof": "joint-snapshot",
        "detail": "snapshot paired-0-0",
    }
    with pytest.raises(ValueError, match="overlap intervals are not what the observations show"):
        ConcurrentServingEvidence.model_validate(
            resealed(
                lambda d: d["results"].__setitem__("overlap", d["results"]["overlap"] + [extra])
            )
        )
    # A safe envelope cannot be carried over a budget the observations no longer fit.
    with pytest.raises(ValueError, match="physical envelopes are not what the observations show"):
        ConcurrentServingEvidence.model_validate(
            resealed(
                lambda d: [
                    g["host_budget"].__setitem__("available_bytes", 1200)
                    for g in d["preconditions"]["gates"]
                ]
            )
        )
    # Nor can a guard outcome be claimed over fields that do not support it.
    with pytest.raises(ValueError, match="guard outcome is not derived"):
        ConcurrentServingEvidence.model_validate(
            resealed(lambda d: d["results"]["guards"][0].__setitem__("process_id", 999))
        )
    # The stated allowances must be the ones every fit gate used.
    for field, value in (
        ("host_headroom_bytes", 11),
        ("extra_overhead_bytes", 11),
        ("safety_fraction", 0.2),
    ):
        with pytest.raises(ValueError, match="allowances other than the recorded configuration"):
            ConcurrentServingEvidence.model_validate(
                resealed(lambda d, f=field, v=value: d["configuration"].__setitem__(f, v))
            )
    with pytest.raises(ValueError, match="allowances other than the recorded configuration"):
        ConcurrentServingEvidence.model_validate(
            resealed(
                lambda d: d["preconditions"]["gates"][-1]["device_budget"].__setitem__(
                    "headroom_bytes", 11
                )
            )
        )


def test_guard_classification_requires_coverage_of_this_worker() -> None:
    normal = guard("mlx")
    assert classify_guard(normal, 42, NOW, FINISHED) == ("normal", ())
    missing = normal.model_copy(update={"started_unix_time": None})
    assert classify_guard(missing, 42, NOW, FINISHED)[0] == "unavailable"
    # A stop reason beats coverage gaps, and both are reported.
    outcome, reasons = classify_guard(
        guard("mlx", guard_stopped=True, guard_other_process=True), 42, NOW, FINISHED
    )
    assert outcome == "stopped" and len(reasons) == 2
    # A guard that exits cleanly after the experiment is still normal.
    done = normal.model_copy(update={"exited_unix_time": FINISHED.timestamp() + 5, "returncode": 0})
    assert classify_guard(done, 42, NOW, FINISHED) == ("normal", ())


def test_serving_envelope_falls_back_and_refuses_unsafe_budgets() -> None:
    rows = timeline("mlx").snapshots
    footprint = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        rows,
        cycle=0,
        host=BUDGET,
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    conservative = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        rows,
        cycle=0,
        host=BUDGET,
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
        cycle=0,
        host=BUDGET,
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    assert fallback.policy == "mlx-rss-plus-allocator-v1" and fallback.policy_notes
    tight = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        rows,
        cycle=0,
        host=TIGHT,
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    assert tight.status == "unsafe"  # 1100 + 110 + 10 + 10 > 1200
    empty = assess_serving_envelope(
        "mlx",
        Backend.MLX,
        (),
        cycle=3,
        host=BUDGET,
        device=None,
        mlx_policy=MlxFitPolicy.FOOTPRINT,
        device_process_peak_bytes=None,
    )
    assert empty.status == "unknown" and "no serving snapshots" in empty.reasons[0]


def test_run_directory_summary_reads_only_preserved_files(tmp_path: Path) -> None:
    from scripts.validation.assemble_two_host_results import main as assemble

    run = tmp_path / "hllm-run-20261014"
    (run / "inputs").mkdir(parents=True)
    (run / "mac" / "rejected-1").mkdir(parents=True)
    (run / "mac" / "final").mkdir()
    (run / "concurrent-attempt-1").mkdir()
    (run / "concurrent-attempt-1" / "README.txt").write_text("refused at the cap\n")
    (run / "inputs" / "plan.json").write_text("{}\n")
    identity = dict(base_revision="abc", source_digest="1" * 64, binaries={"w": "2" * 64})
    (run / "inputs" / "source-identity-concurrent.json").write_text(json.dumps(identity))
    profile_key = dict(environment=dict(source_revision="r", binary_digest="3" * 64))
    (run / "mac" / "rejected-1" / "mlx-memory.preflight.json").write_text(
        json.dumps(
            dict(
                status="refused",
                reasons=["host availability below the cap plus headroom"],
                schema_version="1.2",
                host_budget=dict(available_bytes=5),
                key=profile_key,
                admission_capacity=dict(unified=9),
            )
        )
    )
    guard_rows = [
        {"event": "start", "unix_time": 1.0, "pid": 7},
        {"unix_time": 2.0, "pid": 7, "available_bytes": 2**30, "swapout_bytes": 0},
        {"unix_time": 4.0, "pid": 7, "available_bytes": 2**31, "swapout_bytes": 2**20},
        {"event": "exit", "unix_time": 5.0, "reason": None, "returncode": 0},
    ]
    guard_text = "".join(json.dumps(r) + "\n" for r in guard_rows)
    (run / "baseline-cuda.guard.jsonl").write_text(guard_text)

    def guard_record(reason: str) -> str:
        stop = dict(guard_rows[-1], reason=reason, returncode=-15)
        return "".join(json.dumps(r) + "\n" for r in [*guard_rows[:-1], stop])

    # The launcher stopped the MLX guard; the CUDA guard stopped its own worker, after
    # the soak had read its record.
    (run / "mac" / "final" / "concurrent-mlx.guard.jsonl").write_text(
        guard_record("guard received signal 15")
    )
    (run / "concurrent-cuda.guard.jsonl").write_text(
        guard_record("new swap-out exceeds configured 60-second limit")
    )
    # A baseline refused before its first request.
    (run / "baseline-refused").mkdir()
    (run / "baseline-refused" / "baseline-reload-soak.json").write_text(
        json.dumps(
            dict(
                completed=False,
                scope="f16-reload-qualification",
                plan=dict(plan_digest="p"),
                cycles=[dict(requests=[], preflight={})],
                workers=dict(cuda=dict(binary_digest="4" * 64)),
            )
        )
    )
    (run / "baseline-reload-soak.json").write_text(
        json.dumps(
            dict(
                completed=True,
                scope="f16-reload-qualification",
                plan=dict(plan_digest="p"),
                cycles=[dict(requests=[dict(exact_reference_match=True)])],
                workers=dict(cuda=dict(binary_digest="4" * 64)),
            )
        )
    )
    evidence = build()
    write_evidence(run / "concurrent.json", evidence)
    output = tmp_path / "summary.json"
    assemble([str(run), str(output), "--link", "night_summary=docs/earlier.json"])
    summary = json.loads(output.read_text())
    # Only the directory's base name is recorded, never a host path.
    assert summary["run_directory"] == "hllm-run-20261014"
    assert str(tmp_path) not in output.read_text()
    assert summary["night_summary"] == "docs/earlier.json"
    linux = summary["source_identities"]["linux"]
    assert linux["source-identity-concurrent"]["base_revision"] == "abc"
    assert list(summary["inputs"]) == ["plan.json", "source-identity-concurrent.json"]
    rejected = summary["p1"]["mlx_rejected"][0]
    assert rejected["preflight"]["status"] == "refused" and "artifact" not in rejected
    assert "cuda" not in summary["p1"]  # Absent evidence is absent, not filled in.
    final = summary["baselines"]["final"]
    assert final["all_exact"] and final["guards"]["cuda"]["observations"] == 2
    refused = summary["baselines"]["refused"]
    assert (refused["requests"], refused["all_exact"]) == (0, False)
    assert final["guards"]["cuda"]["swapout_mib"] == 1 and final["guards"]["mlx"] is None
    attempts = summary["p5_attempts"]
    assert attempts[0] == dict(
        directory="concurrent-attempt-1", note="refused at the cap", evidence_present=False
    )
    assert summary["p5"] == attempts[-1]
    assert summary["p5"]["acceptance"] == evidence.results.acceptance == "accepted"
    assert summary["p5"]["evidence_digest"] == evidence.evidence_digest
    assert summary["p5"]["exact"] == summary["p5"]["comparisons"]
    assert set(summary["p5"]["files"]) == {
        "concurrent-cuda.guard.jsonl",
        "mac/final/concurrent-mlx.guard.jsonl",
    }
    # The sealed record may predate a guard's exit; the final records do not.
    assert summary["p5"]["final_guards"]["mlx"]["stop_reason"] == "guard received signal 15"
    assert summary["p5"]["guard_stops"] == [
        dict(worker="cuda", reason="new swap-out exceeds configured 60-second limit")
    ]
    with pytest.raises(FileExistsError):
        assemble([str(run), str(output)])
