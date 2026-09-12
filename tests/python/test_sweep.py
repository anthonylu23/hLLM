from __future__ import annotations

# ruff: noqa: F811 -- imported pytest fixture
import json
from pathlib import Path

import pytest
from hllm_control.profiling.models import ProfileKey, digest
from hllm_control.profiling.runner import write_exclusive
from hllm_control.qualification.sweep import (
    JobResult,
    Reference,
    Sample,
    SweepContent,
    freeze,
    placements,
    run,
    schedule,
    summarize,
    validate_result,
)

from tests.python.test_measured_planner import bundle_fixture, report_for
from tests.python.test_profiling import key  # noqa: F401


def sweep_fixture(path: Path, key: ProfileKey):
    manifest, bundle = bundle_fixture(path, key)
    report = report_for(manifest, bundle)
    assert report.plan
    reference = Reference(
        manifest_digest=manifest.manifest_digest,
        checkpoint_digest=key.checkpoint_digest,
        producer="synthetic-test",
        producer_artifact_digest="a" * 64,
        prompt_ids=(1,) * 4,
        generated_ids=(2,) * 3,
    )
    return freeze(
        SweepContent(
            manifest=manifest,
            workload=key.workload,
            reference=reference,
            selected_plan=report.plan,
            planning_report_digest=digest(report),
            profile_bundle_digest=bundle.bundle_digest,
            executor_digest="b" * 64,
            concurrent_load="idle",
        ),
        path / "sweep",
    )


def fake(spec, job, plan, directory):
    latency = 100 if plan.selected_candidate_id == spec.selected_plan.selected_candidate_id else 110
    sample = Sample(
        token_ids=spec.reference.generated_ids,
        client_arrivals_ms=(10, 50, latency),
        native_arrivals_ms=(9, 49, latency - 1),
        terminal_ms=latency + 2,
    )
    raw: dict[str, object] = dict(test=True, sample=sample.model_dump(mode="json"))
    write_exclusive(directory / "evidence.json", raw)
    return JobResult(
        sweep_digest=spec.sweep_digest,
        job_id=job,
        candidate_id=plan.selected_candidate_id,
        status="measured",
        warmups=(sample,) * spec.warmups,
        sample=sample,
        evidence_digest=digest(raw),
        cleanup_passed=True,
        selected_health_passed=True,
        detail="test",
    )


def test_frozen_schedule_resume_complete_and_tamper(tmp_path: Path, key: ProfileKey):
    spec = sweep_fixture(tmp_path, key)
    assert len(placements(spec)) == 6
    assert schedule(spec, 0) == schedule(spec, 0)
    assert schedule(spec, 0) != schedule(spec, 1)
    directory = tmp_path / "sweep"
    partial = run(spec, directory, fake, maximum_jobs=3)
    assert partial["decision"] == "incomplete"
    final = run(spec, directory, fake)
    assert final["complete_coverage"]
    assert final["decision"] == "pass"

    def unexpected(*args):
        raise AssertionError("resume should not run finished jobs")

    assert run(spec, directory, unexpected) == final
    evidence = next(directory.glob("*/evidence.json"))
    evidence.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        run(spec, directory, unexpected)


def test_wrong_workload_and_divergence_cannot_win(tmp_path: Path, key: ProfileKey):
    spec = sweep_fixture(tmp_path, key)
    data = spec.model_dump(exclude={"sweep_digest"})
    data["reference"]["prompt_ids"] = [1]
    with pytest.raises(ValueError, match="acceptance workload"):
        SweepContent.model_validate(data)
    job, plan = schedule(spec, 0)[1]
    output = tmp_path / "one"
    output.mkdir()
    result = fake(spec, job, plan, output)
    assert result.sample
    bad = result.model_copy(
        update={"sample": result.sample.model_copy(update={"token_ids": (3, 3, 3)})}
    )
    with pytest.raises(ValueError, match="divergent"):
        validate_result(spec, bad, job, plan)
    unsupported = result.model_copy(update={"status": "memory-excluded"})
    with pytest.raises(ValueError, match="memory evidence"):
        validate_result(spec, unsupported, job, plan)
    assert summarize(spec, [result])["decision"] == "incomplete"


def test_drift_does_not_pass(tmp_path: Path, key: ProfileKey):
    spec = sweep_fixture(tmp_path, key)
    directory = tmp_path / "sweep"

    def drifting(spec, job, plan, directory):
        result = fake(spec, job, plan, directory)
        if job.endswith("reference-after"):
            assert result.sample
            result = result.model_copy(
                update={
                    "sample": result.sample.model_copy(
                        update={"client_arrivals_ms": (10, 50, 200), "terminal_ms": 202}
                    )
                }
            )
        return result

    final = run(spec, directory, drifting)
    assert final["decision"] == "unstable"
    assert json.loads((directory / "summary.json").read_text())["decision"] == "unstable"


def test_mixed_sweep_preserves_precision_contract(tmp_path: Path, key: ProfileKey):
    from hllm_control.models import DeploymentPlan, DType

    spec = sweep_fixture(tmp_path, key)
    selected = spec.selected_plan.model_copy(
        update={"schema_version": "1.2", "weight_dtype": DType.F16}
    )
    spec = spec.model_copy(update={"selected_plan": selected})
    for candidate in placements(spec):
        assert candidate.schema_version == "1.2"
        assert candidate.weight_dtype == DType.F16
        assert candidate.execution_dtype == DType.F32
        assert candidate.plan_digest == digest(
            candidate.model_dump(mode="json", exclude={"plan_id", "plan_digest"})
        )
        assert candidate.plan_id == "plan-" + candidate.plan_digest[:16]
        assert DeploymentPlan.model_validate(candidate.model_dump()) == candidate
