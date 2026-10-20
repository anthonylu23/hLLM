from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hllm_control.models import Backend, ConnectionType, DType, WorkloadProfile
from hllm_control.profiling.memory import MlxFitPolicy, PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    MEMORY_PHASES,
    AllocatorSample,
    ComputeMeasurement,
    ComputeRunMeasurement,
    Conditions,
    ConversionMeasurement,
    Environment,
    LinkMeasurement,
    MemoryAmounts,
    MemoryMeasurement,
    MemorySample,
    PhysicalSample,
    ProcessMemoryObservation,
    ProfileArtifact,
    ProfileKey,
    TimingRecord,
    check_compatibility,
    digest,
    make_artifact,
)
from hllm_control.profiling.runner import checkpoint_digest

from tests.process_helpers import plan, write_model


@pytest.fixture
def key(tmp_path: Path) -> ProfileKey:
    root = tmp_path / "model"
    manifest = write_model(root)
    workload = WorkloadProfile(
        workload_id="memory-test",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    return ProfileKey(
        manifest_digest=manifest.manifest_digest,
        checkpoint_digest=checkpoint_digest(root, manifest),
        environment=Environment(
            backend=Backend.CPU,
            device_identity="cpu-1",
            device_name="CPU",
            backend_version="hllm-native",
            driver_version="not-applicable",
            allocator="system",
            allocator_config="default",
            source_revision="test",
            binary_digest="a" * 64,
            compiler="clang",
            os="test-os",
        ),
        assignment=plan(manifest).stages[0],
        workload=workload,
        workload_digest=digest(workload),
        execution_dtype=DType.F32,
        transport_mode="pageable",
        input_kind="synthetic-shape",
        input_digest="b" * 64,
    )


def conditions() -> Conditions:
    return Conditions(
        measured_at=datetime(2026, 9, 8, tzinfo=UTC),
        warmup_cycles=0,
        measured_cycles=1,
        process_policy="fresh-process-then-reloads",
        concurrent_load="idle",
    )


def memory(key: ProfileKey, *, allocator: bool = False) -> MemoryMeasurement:
    return MemoryMeasurement(
        admission_capacity=MemoryAmounts(host=1000, device=1000),
        completed=True,
        samples=tuple(
            MemorySample(
                cycle=0,
                phase=phase,
                weights=MemoryAmounts(),
                cache=MemoryAmounts(),
                workspace=MemoryAmounts(),
                allocator=AllocatorSample(
                    active_bytes=100, cached_bytes=100, peak_bytes=300, peak_scope="phase"
                )
                if allocator
                else None,
                rss_bytes=400,
                rss_lifetime_peak_bytes=500,
                device_available_bytes=2000,
                completed_steps=key.workload.output_tokens
                if phase in ("decode", "request_cleanup", "unload")
                else 0,
            )
            for phase in MEMORY_PHASES
        ),
        physical_samples=(
            PhysicalSample(elapsed_seconds=1, rss_bytes=400, device_process_bytes=350),
        ),
        physical_sample_interval_seconds=0.25,
    )


@pytest.mark.parametrize("kind", ["memory", "compute", "conversion", "link"])
def test_artifact_round_trip_and_tamper_detection(key: ProfileKey, kind: str) -> None:
    measurement = {
        "memory": memory(key),
        "compute": ComputeMeasurement(
            phase="prefill", context_tokens=0, component="stage", samples_ms=(1.0,)
        ),
        "conversion": ConversionMeasurement(
            direction="to-wire", payload_bytes=64, samples_ms=(1.0,)
        ),
        "link": LinkMeasurement(
            source_worker_id="a",
            target_worker_id="b",
            source_environment=key.environment,
            target_environment=key.environment,
            connection_type=ConnectionType.DIRECT,
            payload_bytes=64,
            stream_policy="persistent",
            attribution="round-trip-including-feedback",
            samples_ms=(1.0,),
        ),
    }[kind]
    artifact = make_artifact(key, conditions(), measurement)
    assert ProfileArtifact.model_validate_json(artifact.model_dump_json()) == artifact
    assert make_artifact(key, conditions(), measurement).artifact_digest == artifact.artifact_digest
    modified = artifact.model_dump(mode="json")
    modified["key"]["checkpoint_digest"] = "f" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        ProfileArtifact.model_validate(modified)
    modified["schema_version"] = "2.0"
    with pytest.raises(ValueError):
        ProfileArtifact.model_validate(modified)


@pytest.mark.parametrize(
    "field,value",
    [("checkpoint_digest", "c" * 64), ("manifest_digest", "c" * 64), ("input_digest", "c" * 64)],
)
def test_incompatible_identity(key: ProfileKey, field: str, value: str) -> None:
    artifact = make_artifact(key, conditions(), memory(key))
    changed = ProfileKey.model_validate({**key.model_dump(), field: value})
    assert check_compatibility(artifact, changed, kind="memory").status == "incompatible"


def test_device_ownership_and_shape_compatibility(key: ProfileKey) -> None:
    artifact = make_artifact(key, conditions(), memory(key))
    for field in ("device_identity", "backend_version", "driver_version", "allocator_config"):
        environment = {**key.environment.model_dump(), field: "different"}
        changed = ProfileKey.model_validate({**key.model_dump(), "environment": environment})
        assert check_compatibility(artifact, changed, kind="memory").status == "incompatible"
    assignment = {**key.assignment.model_dump(), "layer_end": key.assignment.layer_end + 1}
    changed = ProfileKey.model_validate({**key.model_dump(), "assignment": assignment})
    assert check_compatibility(artifact, changed, kind="memory").status == "incompatible"
    workload = key.workload.model_copy(update={"total_cached_tokens": 8})
    changed = ProfileKey.model_validate(
        {**key.model_dump(), "workload": workload, "workload_digest": digest(workload)}
    )
    assert check_compatibility(artifact, changed, kind="memory").status == "out-of-range"
    assert check_compatibility(None, key, kind="memory").status == "missing"
    assert check_compatibility(artifact, key, kind="memory").status == "compatible"


@pytest.mark.parametrize(
    "field,value",
    [
        ("concurrency", 2),
        ("total_cached_tokens", 2),
        ("activation_dtype", "F32"),
        ("kv_dtype", "F16"),
    ],
)
def test_unsupported_workload(key: ProfileKey, field: str, value: object) -> None:
    workload = WorkloadProfile.model_validate({**key.workload.model_dump(), field: value})
    with pytest.raises(ValueError):
        ProfileKey.model_validate(
            {**key.model_dump(), "workload": workload, "workload_digest": digest(workload)}
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_timings(value: float) -> None:
    with pytest.raises(ValueError):
        ConversionMeasurement(direction="to-wire", payload_bytes=1, samples_ms=(value,))


def test_timing_scope_requires_exact_payload_and_direction(key: ProfileKey) -> None:
    m = ConversionMeasurement(direction="to-wire", payload_bytes=64, samples_ms=(1.0,))
    a = make_artifact(key, conditions(), m)
    scope = m.model_dump(mode="json", exclude={"samples_ms"})
    assert check_compatibility(a, key, kind="conversion", scope=scope).status == "compatible"
    assert check_compatibility(a, key, kind="conversion").status == "incompatible"
    assert (
        check_compatibility(a, key, kind="conversion", scope={**scope, "payload_bytes": 128}).status
        == "incompatible"
    )


def test_memory_completeness_and_failure(key: ProfileKey) -> None:
    m = memory(key)
    with pytest.raises(ValueError, match="all ordered phases"):
        make_artifact(key, conditions(), m.model_copy(update={"samples": m.samples[:-1]}))
    failed = m.model_copy(update={"completed": False, "error": "allocation failed"})
    a = make_artifact(key, conditions(), failed)
    assert check_compatibility(a, key, kind="memory").status == "incomplete"
    sample = m.samples[-1].model_copy(update={"cache": MemoryAmounts(host=1)})
    with pytest.raises(ValueError, match="reservations did not retire"):
        make_artifact(
            key, conditions(), m.model_copy(update={"samples": (*m.samples[:-1], sample)})
        )


def test_fit_keeps_reservations_and_physical_usage_separate(key: ProfileKey) -> None:
    m = memory(key)
    a = make_artifact(key, conditions(), m)
    budget = PhysicalBudget(available_bytes=1000, headroom_bytes=100, extra_overhead_bytes=50)
    result = assess_fit(a, m.admission_capacity, budget)
    assert result.status == "safe"
    assert result.host_envelope_bytes == 600  # 500 * 1.1 + 50; no added reservation bytes.
    assert (
        assess_fit(
            a, m.admission_capacity, budget.model_copy(update={"available_bytes": 650})
        ).status
        == "unsafe"
    )
    assert assess_fit(a, MemoryAmounts(host=999, device=1000), budget).status == "unknown"
    assert (
        assess_fit(
            a, m.admission_capacity, budget.model_copy(update={"available_bytes": None})
        ).status
        == "unknown"
    )
    missing = m.model_copy(update={"physical_samples": ()})
    cuda_key = ProfileKey.model_validate(
        {**key.model_dump(), "environment": {**key.environment.model_dump(), "backend": "cuda"}}
    )
    cuda = make_artifact(cuda_key, conditions(), missing)
    result = assess_fit(cuda, m.admission_capacity, budget, budget)
    assert result.status == "unknown"
    assert "CUDA physical process observations unavailable" in result.reasons


def test_checkpoint_hash_detects_payload_change(tmp_path: Path) -> None:
    root = tmp_path / "model"
    manifest = write_model(root)
    first = checkpoint_digest(root, manifest)
    path = root / manifest.tensor_files[0].name
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    assert checkpoint_digest(root, manifest) != first


def test_milestone_workload_preserves_long_capacity_example() -> None:
    from hllm_control.planner.config import load_settings, load_workload

    w = load_workload(Path("examples/workloads/milestone-5.yaml"))
    assert (w.prompt_tokens, w.output_tokens, w.total_cached_tokens) == (512, 256, 768)
    assert load_workload(Path("examples/workloads/interactive.yaml")).total_cached_tokens == 32768
    weights = load_settings(Path("examples/profiles/milestone-5-objective.yaml")).objective_weights
    assert (weights.ttft, weights.itl, weights.pipeline_period, weights.memory_pressure) == (
        1,
        255,
        0,
        0,
    )


def test_mlx_unknown_overlap_uses_conservative_union_bound(key: ProfileKey) -> None:
    mlx = ProfileKey.model_validate(
        {**key.model_dump(), "environment": {**key.environment.model_dump(), "backend": "mlx"}}
    )
    m = memory(mlx, allocator=True)
    a = make_artifact(mlx, conditions(), m)
    b = PhysicalBudget(available_bytes=2000, headroom_bytes=100, extra_overhead_bytes=50)
    result = assess_fit(a, m.admission_capacity, b)
    assert result.status == "safe"
    assert result.host_envelope_bytes == 930  # (RSS 500 + allocator 300) * 1.1 + 50.


@pytest.mark.parametrize("footprint", [None, 0, 600])
def test_process_observations_are_sealed_but_do_not_change_fit(
    key: ProfileKey, footprint: int | None
) -> None:
    mlx = ProfileKey.model_validate(
        {**key.model_dump(), "environment": {**key.environment.model_dump(), "backend": "mlx"}}
    )
    original = memory(mlx, allocator=True)
    p = ProcessMemoryObservation(
        process_id=42,
        observed_at_unix_ns=123,
        rss_bytes=400,
        rss_lifetime_peak_bytes=500,
        physical_footprint_bytes=footprint,
        physical_footprint_lifetime_peak_bytes=footprint,
    )
    m = original.model_copy(
        update={
            "samples": tuple(s.model_copy(update={"process_memory": p}) for s in original.samples)
        }
    )
    a = make_artifact(mlx, conditions(), m)
    assert (a.schema_version, a.profiler_version) == ("1.3", "0.3.0")
    assert ProfileArtifact.model_validate_json(a.model_dump_json()) == a
    assert p.model_dump()["physical_footprint_bytes"] == footprint
    budget = PhysicalBudget(available_bytes=2000, headroom_bytes=100, extra_overhead_bytes=50)
    # Under conservative-v1 the sealed OS observations are inert; footprint-v1 (the
    # default) is covered separately in test_footprint_fit.
    conservative = MlxFitPolicy.CONSERVATIVE
    assert assess_fit(a, m.admission_capacity, budget, mlx_policy=conservative) == assess_fit(
        make_artifact(mlx, conditions(), original),
        original.admission_capacity,
        budget,
        mlx_policy=conservative,
    )
    modified = a.model_dump(mode="json")
    modified["measurement"]["samples"][0]["process_memory"]["physical_footprint_bytes"] = 999
    with pytest.raises(ValueError, match="digest mismatch"):
        ProfileArtifact.model_validate(modified)
    for mutation, message in (
        ("legacy-schema", "require profile schema 1.3"),
        ("missing-observation", "requires process observations"),
        ("different-process", "single process"),
        ("inconsistent-rss", "match legacy RSS"),
    ):
        invalid = a.model_dump(mode="json", exclude={"artifact_digest"})
        first = invalid["measurement"]["samples"][0]
        if mutation == "legacy-schema":
            invalid["schema_version"] = "1.0"
        elif mutation == "missing-observation":
            del first["process_memory"]
        elif mutation == "different-process":
            first["process_memory"]["process_id"] = 43
        else:
            first["process_memory"]["rss_bytes"] = 401
        with pytest.raises(ValueError, match=message):
            ProfileArtifact.model_validate({**invalid, "artifact_digest": digest(invalid)})


def test_historical_memory_artifacts_retain_digests() -> None:
    import json

    historical = Path(__file__).parents[2] / "docs/validation/milestone-5-memory"
    checked = 0
    for path in historical.glob("*.json"):
        saved = json.loads(path.read_text())
        if "artifact_digest" not in saved:
            continue
        parsed = ProfileArtifact.model_validate(saved)
        assert parsed.model_dump(mode="json") == saved
        assert isinstance(parsed.measurement, MemoryMeasurement)
        assert all(s.process_memory is None for s in parsed.measurement.samples)
        checked += 1
    assert checked == 4, "historical memory artifacts missing; this test must not pass vacuously"


def test_preflight_refuses_to_start_and_saves_report(tmp_path: Path, monkeypatch) -> None:
    from hllm_control.profiling.runner import run_memory_profile

    from tests.process_helpers import plan, write_model

    root = tmp_path / "model"
    manifest = write_model(root)
    monkeypatch.setattr("hllm_control.profiling.runner.host_available_bytes", lambda: 10)
    w = WorkloadProfile(
        workload_id="preflight",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    with pytest.raises(ValueError, match="exceeds availability"):
        run_memory_profile(
            binary=Path(sys.executable),
            root=root,
            manifest=manifest,
            plan=plan(manifest),
            stage_index=0,
            workload=w,
            backend=Backend.CPU,
            capacity=MemoryAmounts(host=1000),
            output=tmp_path / "rejected.json",
            source_revision="test",
            concurrent_load="test",
        )
    report = (tmp_path / "rejected.preflight.json").read_text()
    assert '"status": "rejected"' in report
    assert not (tmp_path / "rejected.native.jsonl").exists()


def test_preflight_samples_availability_before_reading_the_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    """The decision uses a sample taken before the runner's own checkpoint read.

    On macOS the free-plus-inactive proxy does not count file pages this process has
    just read (they are active), so a sample taken after the digest would charge the
    tool's own read of the checkpoint against the cap, which can be gigabytes for a
    full checkpoint.
    """
    import json

    from hllm_control.profiling import runner

    from tests.process_helpers import plan, write_model

    root = tmp_path / "model"
    manifest = write_model(root)
    calls: list[str] = []
    samples = iter([10**10, 10])  # enough before the read, not enough after it
    real_digest = runner.checkpoint_digest

    def sample() -> int:
        calls.append("sample")
        return next(samples)

    def read_checkpoint(*args, **kwargs):
        calls.append("checkpoint")
        return real_digest(*args, **kwargs)

    monkeypatch.setattr(runner, "host_available_bytes", sample)
    monkeypatch.setattr(runner, "checkpoint_digest", read_checkpoint)
    w = WorkloadProfile(
        workload_id="preflight",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    # The binary is a shell that exits at once, so after an accepted preflight the
    # native run fails and the run raises; the preflight record is what matters here.
    with pytest.raises(RuntimeError):
        runner.run_memory_profile(
            binary=Path("/bin/sh"),
            root=root,
            manifest=manifest,
            plan=plan(manifest),
            stage_index=0,
            workload=w,
            backend=Backend.CPU,
            capacity=MemoryAmounts(host=1000),
            output=tmp_path / "ordered.json",
            source_revision="test",
            concurrent_load="test",
            timeout_seconds=5,
        )
    assert calls == ["sample", "checkpoint", "sample"]
    report = json.loads((tmp_path / "ordered.preflight.json").read_text())
    assert report["status"] == "accepted"
    assert report["availability_sampled_before_checkpoint_read"] is True
    assert report["host_budget"]["available_bytes"] == 10**10
    assert report["host_available_bytes_after_checkpoint_read"] == 10
    assert report["availability_sampled_at"] <= report["observed_at"]
    assert report["schema_version"] == "1.2"
    assert report["device_budget"] is None
    assert report["device_availability_decided_by"] is None


def test_preflight_refreshes_cuda_device_availability_after_the_checkpoint_read(
    tmp_path: Path, monkeypatch
) -> None:
    """Device memory can be taken by another process while the checkpoint is hashed.

    The host rationale (the runner's own read pollutes the macOS proxy) does not apply
    to VRAM, so the device decision uses the smaller of the samples taken before and
    after the digest: free VRAM that falls from 8 GiB to 256 MiB during hashing must
    reject a 4 GiB probe rather than launch it.
    """
    import json

    from hllm_control.profiling import runner

    from tests.process_helpers import plan, write_model

    root = tmp_path / "model"
    manifest = write_model(root)
    calls: list[str] = []
    device_samples = iter([8 * 1024**3, 256 * 1024**2])
    real_digest = runner.checkpoint_digest

    def device_sample(device_id: int) -> int:
        calls.append(f"device:{device_id}")
        return next(device_samples)

    def read_checkpoint(*args, **kwargs):
        calls.append("checkpoint")
        return real_digest(*args, **kwargs)

    monkeypatch.setattr(runner, "host_available_bytes", lambda: 10**12)
    monkeypatch.setattr(runner, "device_available_bytes", device_sample)
    monkeypatch.setattr(runner, "checkpoint_digest", read_checkpoint)
    w = WorkloadProfile(
        workload_id="preflight",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    with pytest.raises(ValueError, match="device: configured probe cap plus physical"):
        runner.run_memory_profile(
            binary=Path("/bin/sh"),
            root=root,
            manifest=manifest,
            plan=plan(manifest),
            stage_index=0,
            workload=w,
            backend=Backend.CUDA,
            capacity=MemoryAmounts(host=1000, device=4 * 1024**3),
            output=tmp_path / "stale-device.json",
            source_revision="test",
            concurrent_load="test",
            device_id=3,
            timeout_seconds=5,
        )
    assert calls == ["device:3", "checkpoint", "device:3"]
    report = json.loads((tmp_path / "stale-device.preflight.json").read_text())
    assert report["status"] == "rejected"
    assert report["device_budget"]["available_bytes"] == 256 * 1024**2
    assert report["device_available_bytes_before_checkpoint_read"] == 8 * 1024**3
    assert report["device_available_bytes_after_checkpoint_read"] == 256 * 1024**2
    assert (
        report["device_availability_decided_by"]
        == "minimum-of-samples-before-and-after-checkpoint-read"
    )
    assert not (tmp_path / "stale-device.native.jsonl").exists()


def test_preflight_treats_a_missing_device_sample_as_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    from hllm_control.profiling import runner

    from tests.process_helpers import plan, write_model

    root = tmp_path / "model"
    manifest = write_model(root)
    device_samples = iter([8 * 1024**3, None])
    monkeypatch.setattr(runner, "host_available_bytes", lambda: 10**12)
    monkeypatch.setattr(runner, "device_available_bytes", lambda _: next(device_samples))
    w = WorkloadProfile(
        workload_id="preflight",
        prompt_tokens=4,
        output_tokens=3,
        total_cached_tokens=7,
        kv_dtype=DType.F32,
    )
    with pytest.raises(ValueError, match="device: current availability unavailable"):
        runner.run_memory_profile(
            binary=Path("/bin/sh"),
            root=root,
            manifest=manifest,
            plan=plan(manifest),
            stage_index=0,
            workload=w,
            backend=Backend.CUDA,
            capacity=MemoryAmounts(host=1000, device=1000),
            output=tmp_path / "no-device.json",
            source_revision="test",
            concurrent_load="test",
            timeout_seconds=5,
        )


def _chunked_run(
    key: ProfileKey, *, final: bool, mutate: str | None = None
) -> tuple[ProfileKey, ComputeRunMeasurement]:
    """A CPU compute run for prompt 5 in chunks of 2 (positions 0, 2, 4) plus 2 decodes."""
    workload = key.workload.model_copy(
        update={"prompt_tokens": 5, "output_tokens": 3, "total_cached_tokens": 8}
    )
    workload = workload.model_copy(update={"prefill_chunk_tokens": 2})
    assignment = key.assignment
    if final:
        assignment = assignment.model_copy(
            update={
                "stage_index": 1,
                "layer_start": 1,
                "layer_end": 2,
                "owns_token_embedding": False,
                "owns_final_norm": True,
                "owns_lm_head": True,
                "owns_sampling": True,
            }
        )
    k = ProfileKey.model_validate(
        {
            **key.model_dump(),
            "assignment": assignment,
            "workload": workload,
            "workload_digest": digest(workload),
        }
    )
    records: list[TimingRecord] = []
    executions = [(0, 0, 0), (0, 1, 2), (0, 2, 4), (1, None, 5), (2, None, 6)]
    for mode in ("whole-stage", "component-synchronized"):
        for step, chunk, position in executions:
            names: list[tuple[str, int | None, float]] = [("stage", None, 3.0)]
            if chunk == 0:
                names.append(("sequence_allocation", None, 0.5))
            if mode == "component-synchronized":
                last = chunk is None or chunk == 2 or mutate == "early-sampling"
                tail = (
                    [("final_norm", None, 0.5), ("lm_head", None, 0.5), ("sampling", None, 0.5)]
                    if final and last
                    else [("to-wire", None, 1.5)]
                    if not final
                    else [("runtime_overhead", None, 1.5)]
                )
                names += [("from-wire" if final else "embedding", None, 0.5)]
                names += [("layer", assignment.layer_start, 1.0)]
                names += tail
                if tail[-1][0] != "runtime_overhead":
                    names.append(("runtime_overhead", None, 3.0 - 1.5 - sum(t[2] for t in tail)))
            for component, layer, ms in names:
                records.append(
                    TimingRecord(
                        cycle=0,
                        step=step,
                        context_tokens=position + (1 if mutate == "position" and chunk == 1 else 0),
                        phase="prefill" if step == 0 else "decode",
                        component=component,  # type: ignore[arg-type]
                        layer_index=layer if component == "layer" else None,
                        timing_mode=mode,  # type: ignore[arg-type]
                        milliseconds=ms,
                        chunk_index=None if mutate == "unindexed" else chunk,
                    )
                )
    return k, ComputeRunMeasurement(
        admission_capacity=MemoryAmounts(host=1),
        records=tuple(records),
        identical_output_cycles=(0,),
        completed=True,
    )


@pytest.mark.parametrize("final", [False, True])
def test_chunked_compute_run_validates_per_chunk_records(key: ProfileKey, final: bool) -> None:
    k, run = _chunked_run(key, final=final)
    artifact = make_artifact(
        k, conditions().model_copy(update={"process_policy": "loaded-stage-paired-passes"}), run
    )
    stage = [r for r in run.records if r.component == "stage" and r.timing_mode == "whole-stage"]
    assert [(r.chunk_index, r.context_tokens) for r in stage] == [
        (0, 0),
        (1, 2),
        (2, 4),
        (None, 5),
        (None, 6),
    ]
    assert ProfileArtifact.model_validate_json(artifact.model_dump_json()) == artifact


@pytest.mark.parametrize(
    "final,mutate",
    [(True, "early-sampling"), (False, "position"), (False, "unindexed")],
)
def test_chunked_compute_run_rejects_wrong_chunk_records(
    key: ProfileKey, final: bool, mutate: str
) -> None:
    k, run = _chunked_run(key, final=final, mutate=mutate)
    with pytest.raises(ValueError):
        make_artifact(
            k,
            conditions().model_copy(update={"process_policy": "loaded-stage-paired-passes"}),
            run,
        )


def test_whole_prompt_timing_records_omit_the_chunk_index() -> None:
    record = TimingRecord(
        cycle=0,
        step=0,
        context_tokens=0,
        phase="prefill",
        component="stage",
        layer_index=None,
        timing_mode="whole-stage",
        milliseconds=1.0,
    )
    assert "chunk_index" not in record.model_dump(mode="json")
