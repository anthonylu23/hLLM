from __future__ import annotations

# ruff: noqa: F811 -- imported pytest fixture
from pathlib import Path

import pytest
from hllm_control.models import Backend
from hllm_control.profiling.memory import FitResult, MlxFitPolicy, PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    MemoryMeasurement,
    ProfileArtifact,
    ProfileKey,
    make_artifact,
)

from tests.python.test_profiling import conditions, key, memory  # noqa: F401


def footprint_measurement(key: ProfileKey) -> MemoryMeasurement:
    """Coherent schema-1.3 telemetry: conservative envelope 930, footprint envelope 710."""
    raw = memory(key, allocator=True).model_dump()
    for index, sample in enumerate(raw["samples"]):
        sample["process_memory"] = dict(
            process_id=42,
            observed_at_unix_ns=index + 100,
            rss_bytes=sample["rss_bytes"],
            rss_lifetime_peak_bytes=sample["rss_lifetime_peak_bytes"],
            physical_footprint_bytes=550,
            physical_footprint_lifetime_peak_bytes=600,
        )
    return MemoryMeasurement.model_validate(raw)


def footprint_artifact(key: ProfileKey, *, backend: Backend = Backend.MLX) -> ProfileArtifact:
    key = key.model_copy(
        update={"environment": key.environment.model_copy(update={"backend": backend})}
    )
    return make_artifact(key, conditions(), footprint_measurement(key))


BUDGET = PhysicalBudget(available_bytes=900, headroom_bytes=100, extra_overhead_bytes=50)


def test_opt_in_uses_maximum_and_preserves_allowances(key: ProfileKey) -> None:
    a = footprint_artifact(key)
    m = a.measurement
    assert isinstance(m, MemoryMeasurement)
    budget = PhysicalBudget(available_bytes=900, headroom_bytes=100, extra_overhead_bytes=50)
    old = assess_fit(a, m.admission_capacity, budget, mlx_policy=MlxFitPolicy.CONSERVATIVE)
    new = assess_fit(a, m.admission_capacity, budget, mlx_policy=MlxFitPolicy.FOOTPRINT)
    assert old.status == "unsafe" and old.host_envelope_bytes == 930
    assert new.status == "safe" and new.host_envelope_bytes == 710
    assert assess_fit(a, m.admission_capacity, budget) == new  # footprint-v1 is the default
    assert new.policy == "mlx-footprint-max-v1"
    assert new.requested_mlx_policy == MlxFitPolicy.FOOTPRINT and not new.policy_notes
    assert (
        assess_fit(
            a,
            m.admission_capacity,
            budget.model_copy(update={"available_bytes": 809}),
            mlx_policy=MlxFitPolicy.FOOTPRINT,
        ).status
        == "unsafe"
    )  # 710 plus the same 100-byte headroom
    assert (
        assess_fit(
            a,
            m.admission_capacity,
            budget.model_copy(update={"available_bytes": None}),
            mlx_policy=MlxFitPolicy.FOOTPRINT,
        ).status
        == "unknown"
    )
    assert (
        assess_fit(
            a,
            m.admission_capacity.model_copy(update={"host": 999}),
            budget,
            mlx_policy=MlxFitPolicy.FOOTPRINT,
        ).status
        == "unknown"
    )  # a smaller admission cap still requires reprofiling


@pytest.mark.parametrize("view", ["rss", "allocator"])
def test_other_accounting_views_can_dominate(key: ProfileKey, view: str) -> None:
    a = footprint_artifact(key)
    assert isinstance(a.measurement, MemoryMeasurement)
    raw = a.measurement.model_dump()
    if view == "rss":
        raw["physical_samples"][0]["rss_bytes"] = 900
    else:
        raw["samples"][2]["allocator"]["cached_bytes"] = 800  # active + cached, not peak alone
    m = MemoryMeasurement.model_validate(raw)
    a = make_artifact(a.key, a.conditions, m)
    b = PhysicalBudget(available_bytes=1000, headroom_bytes=100, extra_overhead_bytes=50)
    r = assess_fit(a, m.admission_capacity, b, mlx_policy=MlxFitPolicy.FOOTPRINT)
    assert r.policy == "mlx-footprint-max-v1"
    assert r.host_envelope_bytes == 1040 and r.status == "unsafe"


@pytest.mark.parametrize(
    "defect",
    [
        "legacy",
        "null",
        "zero",
        "missing-rss",
        "decreasing-peak",
        "current-exceeds-peak",
        "clock-backwards",
        "allocator-absent",
        "allocator-peak-unavailable",
        "incomplete",
        "wrong-process-policy",
    ],
)
def test_unqualified_evidence_falls_back_without_lowering_envelope(
    key: ProfileKey, defect: str
) -> None:
    a = footprint_artifact(key)
    assert isinstance(a.measurement, MemoryMeasurement)
    raw = a.measurement.model_dump()
    p = raw["samples"][1]["process_memory"]
    c = a.conditions
    if defect == "legacy":
        for s in raw["samples"]:
            s.pop("process_memory")
    elif defect in ("null", "zero"):
        p["physical_footprint_bytes"] = None if defect == "null" else 0
    elif defect == "missing-rss":
        p["rss_bytes"] = raw["samples"][1]["rss_bytes"] = None
    elif defect == "decreasing-peak":
        p["physical_footprint_lifetime_peak_bytes"] = 599
    elif defect == "current-exceeds-peak":
        p["physical_footprint_bytes"] = 601
    elif defect == "clock-backwards":
        p["observed_at_unix_ns"] = 99
    elif defect == "allocator-absent":
        raw["samples"][1]["allocator"] = None
    elif defect == "allocator-peak-unavailable":
        raw["samples"][1]["allocator"]["peak_scope"] = "unavailable"
    elif defect == "incomplete":
        raw.update(completed=False, error="interrupted")
    else:
        c = c.model_copy(update={"process_policy": "loaded-stage-paired-passes"})
    m = MemoryMeasurement.model_validate(raw)
    a = make_artifact(a.key, c, m)
    budget = PhysicalBudget(available_bytes=2000, headroom_bytes=100, extra_overhead_bytes=50)
    old = assess_fit(a, m.admission_capacity, budget, mlx_policy=MlxFitPolicy.CONSERVATIVE)
    new = assess_fit(a, m.admission_capacity, budget, mlx_policy=MlxFitPolicy.FOOTPRINT)
    assert (new.status, new.host_envelope_bytes) == (old.status, old.host_envelope_bytes)
    assert new.policy == "mlx-rss-plus-allocator-v1" and new.policy_notes


@pytest.mark.parametrize("backend", [Backend.CPU, Backend.CUDA])
def test_policy_does_not_change_other_backends(key: ProfileKey, backend: Backend) -> None:
    a = footprint_artifact(key, backend=backend)
    assert isinstance(a.measurement, MemoryMeasurement)
    b = PhysicalBudget(available_bytes=2000, headroom_bytes=100, extra_overhead_bytes=50)
    assert assess_fit(a, a.measurement.admission_capacity, b, b) == assess_fit(
        a, a.measurement.admission_capacity, b, b, mlx_policy=MlxFitPolicy.FOOTPRINT
    )


HISTORICAL = Path(__file__).parents[2] / "docs/validation/milestone-5-memory"


def test_historical_fit_reports_still_round_trip_and_replay() -> None:
    import json

    reports = sorted(HISTORICAL.glob("*.fit.json"))
    assert len(reports) == 4, "historical fit reports missing; this test must not pass vacuously"
    for path in reports:
        report = json.loads(path.read_text())
        saved = report["assessment"]
        assert FitResult.model_validate(saved).model_dump(mode="json") == saved
        artifact = ProfileArtifact.model_validate_json(
            path.with_name(path.name.removesuffix(".fit.json") + ".json").read_text()
        )
        assert artifact.artifact_digest == report["profile_digest"]
        assert isinstance(artifact.measurement, MemoryMeasurement)
        host = PhysicalBudget.model_validate(report["host_budget"])
        device = (
            PhysicalBudget.model_validate(report["device_budget"])
            if report["device_budget"]
            else None
        )
        for policy in MlxFitPolicy:
            # The default formula reproduces the saved decision exactly; requesting the
            # footprint policy on legacy evidence falls back to the same decision.
            replayed = assess_fit(
                artifact, artifact.measurement.admission_capacity, host, device, mlx_policy=policy
            )
            assert (
                replayed.status,
                replayed.host_envelope_bytes,
                replayed.device_envelope_bytes,
            ) == (saved["status"], saved["host_envelope_bytes"], saved["device_envelope_bytes"])
            if policy == MlxFitPolicy.FOOTPRINT and artifact.key.environment.backend == Backend.MLX:
                assert replayed.policy == "mlx-rss-plus-allocator-v1" and replayed.policy_notes


def test_policy_is_hashed_and_changes_mlx_placement_and_fresh_activation(
    tmp_path, key, monkeypatch
):
    from unittest.mock import MagicMock

    from hllm_control.planner import activation
    from hllm_control.planner.measured import BundleContent, ProfileBundle, seal_bundle
    from hllm_control.proto import common_pb2, control_pb2, profile_pb2

    from tests.python.test_measured_planner import bundle_fixture, report_for

    manifest, original = bundle_fixture(
        tmp_path, key, backend=Backend.MLX, memory_measurement=footprint_measurement, host=BUDGET
    )
    assert all("mlx_fit_policy" not in w for w in original.model_dump(mode="json")["workers"])
    # The conservative sum (930 + 100 headroom) exceeds the 900 bytes available.
    conservative = report_for(manifest, original)
    assert conservative.plan is None and conservative.candidates
    assert all(
        any("exceeds current availability" in r for r in c.rejection_reasons)
        for c in conservative.candidates
    )
    data = original.model_dump(mode="json", exclude={"bundle_digest"})
    for w in data["workers"]:
        w["mlx_fit_policy"] = "footprint-v1"
    with pytest.raises(ValueError, match="digest mismatch"):
        ProfileBundle.model_validate({**data, "bundle_digest": original.bundle_digest})
    bundle = seal_bundle(BundleContent.model_validate(data))
    assert bundle.bundle_digest != original.bundle_digest
    report = report_for(manifest, bundle)
    plan = report.plan
    assert plan is not None
    selected = next(c for c in report.candidates if c.candidate_id == plan.selected_candidate_id)
    assert selected.performance is not None
    assert selected.performance.host_envelope_bytes == {"a": 710, "b": 710}
    controls = []
    for assignment in plan.stages:
        w = next(w for w in bundle.workers if w.worker.worker_id == assignment.worker_id)
        control = MagicMock()
        control.GetCapabilities.return_value = control_pb2.Capabilities(
            worker=profile_pb2.WorkerProfile(
                worker_id=w.worker.worker_id,
                endpoint=w.worker.endpoint,
                backend=profile_pb2.BACKEND_MLX,
                supported_architectures=[manifest.architecture.architecture_id],
                supported_execution_dtypes=[common_pb2.DATA_TYPE_F32],
                memory_budgets=[
                    profile_pb2.MemoryBudget(
                        domain=profile_pb2.MEMORY_DOMAIN_UNIFIED,
                        capacity_bytes=w.admission_capacity.unified,
                    )
                ],
            )
        )
        control.GetQualificationState.return_value = control_pb2.QualificationState(
            available_host_bytes=123456,
            binary_digest=w.runtime_binary_digest,
            device_identity=w.runtime_device_identity,
            backend_version=w.compute_environment.backend_version,
            driver_version=w.runtime_driver_version,
            allocator=w.compute_environment.allocator,
            device_fingerprint=w.compute_environment.device_identity,
            boundary_transport_mode=w.transport_mode,
        )
        control.GetMemoryReport.return_value = control_pb2.MemoryReport()
        controls.append(control)
    clock = MagicMock()
    clock.now.return_value = bundle.evaluated_at
    monkeypatch.setattr(activation, "datetime", clock)
    fresh_calls = []

    def fresh_fit(a, capacity, host, device, **kwargs):
        fresh_calls.append((host.available_bytes, kwargs["mlx_policy"]))
        return assess_fit(a, capacity, host, device, **kwargs)

    monkeypatch.setattr(activation, "assess_fit", fresh_fit)
    activation.validate_activation(manifest, plan, bundle, controls)
    assert fresh_calls == [(123456, MlxFitPolicy.FOOTPRINT)] * 2
    # The same fresh availability under the conservative bundle policy is refused.
    with pytest.raises(ValueError):
        activation.validate_activation(
            manifest,
            plan,
            seal_bundle(
                BundleContent.model_validate(
                    original.model_dump(mode="json", exclude={"bundle_digest"})
                )
            ),
            controls,
        )


def test_explicit_policy_is_rejected_on_non_mlx_workers_and_evidence(tmp_path, key):
    from hllm_control.planner.measured import BundleContent
    from hllm_control.qualification.sweep import MemoryExclusion

    from tests.python.test_measured_planner import bundle_fixture

    _, bundle = bundle_fixture(tmp_path, key)
    data = bundle.model_dump(mode="json", exclude={"bundle_digest"})
    data["workers"][0]["mlx_fit_policy"] = "footprint-v1"
    with pytest.raises(ValueError, match="MLX workers only"):
        BundleContent.model_validate(data)
    with pytest.raises(ValueError, match="MLX workers only"):
        native_worker("a", Backend.CPU, MlxFitPolicy.FOOTPRINT, tmp_path)
    a = footprint_artifact(key, backend=Backend.CPU)
    assert isinstance(a.measurement, MemoryMeasurement)
    with pytest.raises(ValueError, match="MLX evidence only"):
        MemoryExclusion(
            profile=a,
            capacity=a.measurement.admission_capacity,
            host=BUDGET,
            mlx_fit_policy=MlxFitPolicy.FOOTPRINT,
        )


def native_worker(name: str, backend: Backend, policy: MlxFitPolicy, root: Path):
    from hllm_control.profiling.models import MemoryAmounts
    from hllm_control.qualification.native import NativeWorker

    return NativeWorker(
        worker_id=name,
        backend=backend,
        endpoint="127.0.0.1:" + str(50000 + len(name)),
        binary=str(root / "worker"),
        binary_digest="d" * 64,
        memory_binary=str(root / "profiler"),
        memory_binary_digest="d" * 64,
        model_root=str(root / "model"),
        evidence_root=str(root / "evidence"),
        capacity=MemoryAmounts(host=1000, device=1000),
        host_headroom_bytes=BUDGET.headroom_bytes,
        extra_overhead_bytes=BUDGET.extra_overhead_bytes,
        mlx_fit_policy=policy,
    )


def test_probe_host_forwards_the_bound_policy(tmp_path, key, monkeypatch, capsys):
    import io
    import json
    import sys

    from hllm_control.qualification import host

    from tests.process_helpers import plan, write_model

    manifest = write_model(tmp_path / "model")
    deployment = plan(manifest)
    seen = []

    def fake_profile(**kwargs):
        seen.append(kwargs["mlx_fit_policy"])
        output = kwargs["output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".fit.json").write_text(
            json.dumps({"assessment": {"requested_mlx_policy": kwargs["mlx_fit_policy"].value}})
        )
        return footprint_artifact(key)

    monkeypatch.setattr(host, "run_memory_profile", fake_profile)
    monkeypatch.setattr(host, "sha256_file", lambda path: "d" * 64)
    monkeypatch.setattr(host, "package_digest", lambda: "p" * 64)
    for policy in (MlxFitPolicy.FOOTPRINT, MlxFitPolicy.CONSERVATIVE):
        worker = native_worker("a", Backend.MLX, policy, tmp_path).model_dump(mode="json")
        # The default is omitted from the payload, so the host must apply it itself.
        assert ("mlx_fit_policy" in worker) == (policy == MlxFitPolicy.FOOTPRINT)
        payload = dict(
            package_digest="p" * 64,
            worker=worker,
            manifest=manifest.model_dump(mode="json"),
            plan=deployment.model_dump(mode="json"),
            workload=key.workload.model_dump(mode="json"),
            stage_index=0,
            run_id="run-" + policy.value,
            source_revision="test",
            concurrent_load="idle",
        )
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload) + "\n"))
        monkeypatch.setattr(sys, "argv", ["host", "probe"])
        host.main()
        out = json.loads(capsys.readouterr().out)
        assert seen[-1] == policy
        assert json.loads(out["files"]["a.fit.json"])["assessment"] == {
            "requested_mlx_policy": policy.value
        }


def test_executor_requires_matching_policies_and_assesses_probes_under_them(
    tmp_path, key, monkeypatch
):
    import json
    import subprocess

    from hllm_control.qualification.native import NativeExecutor
    from hllm_control.qualification.sweep import memory_basis, schedule

    from tests.python.test_sweep import sweep_fixture

    def executor(policy):
        return NativeExecutor(
            workers=tuple(native_worker(n, Backend.MLX, policy, tmp_path) for n in "ab"),
            source_revision="test",
        )

    recorded: dict[str, str] = {}
    current = {}

    def probe(command, *, input, **kwargs):
        payload = json.loads(input)
        spec, plan = current["spec"], current["plan"]
        stage = plan.stages[payload["stage_index"]]
        k = key.model_copy(
            update={
                "manifest_digest": spec.manifest.manifest_digest,
                "checkpoint_digest": spec.reference.checkpoint_digest,
                "assignment": stage,
                "workload": spec.workload,
            }
        )
        fit = {
            "host_budget": BUDGET.model_dump(),
            "device_budget": None,
            "assessment": {"requested_mlx_policy": recorded[payload["worker"]["worker_id"]]},
        }
        stdout = json.dumps(
            {
                "artifact": footprint_artifact(k).model_dump(mode="json"),
                "files": {"a.fit.json": json.dumps(fit)},
            }
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    def no_serve(*args, **kwargs):
        raise RuntimeError("serving disabled in this test")

    monkeypatch.setattr(subprocess, "run", probe)
    monkeypatch.setattr(subprocess, "Popen", no_serve)

    def run(name, ex, policies, recorded_policies):
        (tmp_path / name).mkdir()
        spec = sweep_fixture(
            tmp_path / name, key, executor_digest=ex.identity(), mlx_fit_policies=policies
        )
        job, plan = schedule(spec, 0)[1]
        current.update(spec=spec, plan=plan)
        recorded.clear()
        recorded.update(recorded_policies)
        directory = tmp_path / name / "job"
        directory.mkdir()
        return ex(spec, job, plan, directory)

    footprint = executor(MlxFitPolicy.FOOTPRINT)
    bound = {"a": MlxFitPolicy.FOOTPRINT, "b": MlxFitPolicy.FOOTPRINT}
    both = {"a": "footprint-v1", "b": "footprint-v1"}
    # Probes assessed under the bound policy admit the candidate through to serving.
    result = run("matching", footprint, bound, both)
    assert result.status == "unknown" and "serving disabled" in result.detail
    assert result.host_envelope_bytes == {"a": 710, "b": 710}
    assert result.memory_exclusion is None
    # A probe that applied a different policy is not re-assessed under this one.
    result = run("probe-drift", footprint, bound, {**both, "a": "conservative-v1"})
    assert "probe fit policy" in result.detail and result.host_envelope_bytes == {}
    # An executor whose policy differs from the frozen bundle binding never runs.
    with pytest.raises(ValueError, match="differs from the frozen"):
        run("unbound", footprint, {}, both)
    # The conservative sum excludes the same evidence and records that policy.
    result = run("conservative", executor(MlxFitPolicy.CONSERVATIVE), {}, {"a": "conservative-v1"})
    assert result.status == "memory-excluded" and result.host_envelope_bytes == {"a": 930}
    assert result.memory_exclusion is not None
    assert result.memory_exclusion.mlx_fit_policy == MlxFitPolicy.CONSERVATIVE
    assert "Conservative" in memory_basis({}) and "footprint-v1" in memory_basis(bound)


def test_sweep_exclusions_replay_the_recorded_policy(tmp_path, key):
    from hllm_control.qualification.sweep import MemoryExclusion, schedule, validate_result

    from tests.python.test_sweep import fake, sweep_fixture

    spec = sweep_fixture(tmp_path, key)
    job, plan = schedule(spec, 0)[1]
    k = key.model_copy(
        update={
            "manifest_digest": spec.manifest.manifest_digest,
            "checkpoint_digest": spec.reference.checkpoint_digest,
            "assignment": plan.stages[0],
            "workload": spec.workload,
        }
    )
    a = footprint_artifact(k)
    assert isinstance(a.measurement, MemoryMeasurement)
    exclusion = MemoryExclusion(
        profile=a,
        capacity=a.measurement.admission_capacity,
        host=PhysicalBudget(available_bytes=900, headroom_bytes=100, extra_overhead_bytes=50),
    )
    result = fake(spec, job, plan, tmp_path).model_copy(
        update={
            "status": "memory-excluded",
            "memory_exclusion": exclusion,
            "sample": None,
        }
    )
    validate_result(spec, result, job, plan)  # conservative formula excludes this candidate
    footprint = exclusion.model_copy(update={"mlx_fit_policy": MlxFitPolicy.FOOTPRINT})
    with pytest.raises(ValueError, match="unsupported independent memory exclusion"):
        validate_result(spec, result.model_copy(update={"memory_exclusion": footprint}), job, plan)
