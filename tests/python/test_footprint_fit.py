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


def footprint_artifact(key: ProfileKey, *, backend: Backend = Backend.MLX) -> ProfileArtifact:
    key = key.model_copy(
        update={"environment": key.environment.model_copy(update={"backend": backend})}
    )
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
    return make_artifact(key, conditions(), MemoryMeasurement.model_validate(raw))


def test_opt_in_uses_maximum_and_preserves_allowances(key: ProfileKey) -> None:
    a = footprint_artifact(key)
    m = a.measurement
    assert isinstance(m, MemoryMeasurement)
    budget = PhysicalBudget(available_bytes=900, headroom_bytes=100, extra_overhead_bytes=50)
    old = assess_fit(a, m.admission_capacity, budget)
    new = assess_fit(a, m.admission_capacity, budget, mlx_policy=MlxFitPolicy.FOOTPRINT)
    assert old.status == "unsafe" and old.host_envelope_bytes == 930
    assert new.status == "safe" and new.host_envelope_bytes == 710
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
    old = assess_fit(a, m.admission_capacity, budget)
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


def test_historical_fit_reports_still_round_trip() -> None:
    import json

    for path in Path("docs/validation/milestone-5-memory").glob("*.fit.json"):
        saved = json.loads(path.read_text())["assessment"]
        assert FitResult.model_validate(saved).model_dump(mode="json") == saved


def test_policy_is_hashed_and_propagates_to_planning_and_fresh_activation(
    tmp_path, key, monkeypatch
):
    from unittest.mock import MagicMock

    from hllm_control.planner import activation, measured
    from hllm_control.planner.measured import BundleContent, ProfileBundle, seal_bundle
    from hllm_control.proto import common_pb2, control_pb2, profile_pb2

    from tests.python.test_measured_planner import bundle_fixture, report_for

    manifest, original = bundle_fixture(tmp_path, key)
    assert all("mlx_fit_policy" not in w for w in original.model_dump(mode="json")["workers"])
    data = original.model_dump(mode="json", exclude={"bundle_digest"})
    for w in data["workers"]:
        w["mlx_fit_policy"] = "footprint-v1"
    with pytest.raises(ValueError, match="digest mismatch"):
        ProfileBundle.model_validate({**data, "bundle_digest": original.bundle_digest})
    bundle = seal_bundle(BundleContent.model_validate(data))
    assert bundle.bundle_digest != original.bundle_digest
    recorded = []

    def record_fit(*args, **kwargs):
        recorded.append(kwargs["mlx_policy"])
        return assess_fit(*args, **kwargs)

    monkeypatch.setattr(measured, "assess_fit", record_fit)
    plan = report_for(manifest, bundle).plan
    assert plan is not None and recorded and set(recorded) == {MlxFitPolicy.FOOTPRINT}
    controls = []
    for assignment in plan.stages:
        w = next(w for w in bundle.workers if w.worker.worker_id == assignment.worker_id)
        control = MagicMock()
        control.GetCapabilities.return_value = control_pb2.Capabilities(
            worker=profile_pb2.WorkerProfile(
                worker_id=w.worker.worker_id,
                endpoint=w.worker.endpoint,
                backend=profile_pb2.BACKEND_CPU,
                supported_architectures=[manifest.architecture.architecture_id],
                supported_execution_dtypes=[common_pb2.DATA_TYPE_F32],
                memory_budgets=[
                    profile_pb2.MemoryBudget(
                        domain=profile_pb2.MEMORY_DOMAIN_HOST,
                        capacity_bytes=w.admission_capacity.host,
                    ),
                    profile_pb2.MemoryBudget(
                        domain=profile_pb2.MEMORY_DOMAIN_DEVICE,
                        capacity_bytes=w.admission_capacity.device,
                    ),
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
        fresh_calls.append(host.available_bytes)
        assert kwargs["mlx_policy"] == MlxFitPolicy.FOOTPRINT
        return assess_fit(a, capacity, host, device, **kwargs)

    monkeypatch.setattr(activation, "assess_fit", fresh_fit)
    activation.validate_activation(manifest, plan, bundle, controls)
    assert fresh_calls == [123456, 123456]


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
