"""Frozen sweep protocol and conservative independent comparison.

An executor receives placements and inputs, never planner timing samples. Every
job starts fresh workers, performs two warmups, and records client/native arrivals.
Results are immutable. Failed/incomplete jobs remain visible across resumes.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path
from statistics import median
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from hllm_control.models import (
    DeploymentPlan,
    ModelManifest,
    PerformanceEstimate,
    PlanningMode,
    WorkloadProfile,
)
from hllm_control.planner.planner import assignments_for
from hllm_control.profiling.memory import PhysicalBudget, assess_fit
from hllm_control.profiling.models import (
    Digest,
    MemoryAmounts,
    MemoryMeasurement,
    Milliseconds,
    ProfileArtifact,
    ProfileModel,
    digest,
)
from hllm_control.profiling.runner import write_exclusive


class Reference(ProfileModel):
    manifest_digest: Digest
    checkpoint_digest: Digest
    producer: str
    producer_artifact_digest: Digest
    prompt_ids: Annotated[tuple[int, ...], Field(min_length=1)]
    generated_ids: Annotated[tuple[int, ...], Field(min_length=1)]
    policy: Literal["exact-greedy-token-sequence"] = "exact-greedy-token-sequence"


class SweepContent(ProfileModel):
    schema_version: Literal["1.0"] = "1.0"
    manifest: ModelManifest
    workload: WorkloadProfile
    reference: Reference
    selected_plan: DeploymentPlan
    planning_report_digest: Digest
    profile_bundle_digest: Digest
    executor_digest: Digest
    predictions: dict[str, PerformanceEstimate] = Field(default_factory=dict)
    concurrent_load: str
    seed: int = 5506
    warmups: Annotated[int, Field(ge=2)] = 2
    repetitions: Annotated[int, Field(ge=5)] = 5
    maximum_repetitions: Annotated[int, Field(ge=5, le=100)] = 15
    regret_limit: float = Field(default=0.15, gt=0, lt=1)
    drift_limit: float = Field(default=0.10, gt=0, lt=1)

    @model_validator(mode="after")
    def scope(self) -> Self:
        p, w, r = self.selected_plan, self.workload, self.reference
        if (
            p.manifest_digest != self.manifest.manifest_digest
            or r.manifest_digest != p.manifest_digest
            or p.planning_mode != PlanningMode.MEASURED
            or p.profile_bundle_digest != self.profile_bundle_digest
            or p.workload_digest != digest(w)
            or len(p.stages) != 2
        ):
            raise ValueError("selection/reference/bundle scope mismatch")
        if p.plan_digest != digest(p.model_dump(mode="json", exclude={"plan_id", "plan_digest"})):
            raise ValueError("frozen selected plan hash mismatch")
        if len(r.prompt_ids) != w.prompt_tokens or len(r.generated_ids) != w.output_tokens:
            raise ValueError("reference is not the acceptance workload")
        if any(
            not 0 <= t < self.manifest.config.vocabulary_size
            for t in (*r.prompt_ids, *r.generated_ids)
        ):
            raise ValueError("reference token outside vocabulary")
        if w.concurrency != 1 or w.total_cached_tokens != w.prompt_tokens + w.output_tokens:
            raise ValueError("sweep requires concurrency one and exact request cache capacity")
        if self.maximum_repetitions < self.repetitions:
            raise ValueError("maximum repetitions below initial count")
        return self


class SweepSpec(SweepContent):
    sweep_digest: Digest

    @model_validator(mode="after")
    def seal(self) -> Self:
        if self.sweep_digest != digest(self.model_dump(mode="json", exclude={"sweep_digest"})):
            raise ValueError("sweep digest mismatch")
        return self


def freeze(content: SweepContent, directory: Path) -> SweepSpec:
    data = content.model_dump(mode="json")
    spec = SweepSpec.model_validate({**data, "sweep_digest": digest(data)})
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sweep.json"
    if path.exists():
        if SweepSpec.model_validate_json(path.read_text()) != spec:
            raise ValueError("resume inputs differ from frozen sweep")
    else:
        write_exclusive(path, spec.model_dump(mode="json"))
    return spec


def placements(spec: SweepSpec) -> tuple[DeploymentPlan, ...]:
    names = tuple(s.worker_id for s in spec.selected_plan.stages)
    result: list[DeploymentPlan] = []
    for first, final in (names, names[::-1]):
        for split in range(1, spec.manifest.config.num_layers):
            cid = f"{first}--{final}-m{split:03d}"
            unsigned = dict(
                planner_version="qualification-1.0",
                deployment_version=1,
                manifest_digest=spec.manifest.manifest_digest,
                workload_id=spec.workload.workload_id,
                planning_mode=PlanningMode.FEASIBILITY,
                execution_dtype=spec.selected_plan.execution_dtype,
                activation_dtype=spec.selected_plan.activation_dtype,
                split_layer=split,
                stages=assignments_for(first, final, split, spec.manifest.config.num_layers),
                duplicated_tensor_groups=spec.selected_plan.duplicated_tensor_groups,
                selected_candidate_id=cid,
            )
            provisional = DeploymentPlan.model_validate(
                {"plan_id": "pending", "plan_digest": "pending", **unsigned}
            )
            d = digest(provisional.model_dump(mode="json", exclude={"plan_id", "plan_digest"}))
            result.append(
                provisional.model_copy(update={"plan_id": "sweep-" + d[:16], "plan_digest": d})
            )
    return tuple(result)


class Sample(ProfileModel):
    token_ids: tuple[int, ...]
    client_arrivals_ms: tuple[Milliseconds, ...]
    native_arrivals_ms: tuple[Milliseconds, ...]
    terminal_ms: Milliseconds
    native_request_setup_ms: Milliseconds | None = None

    @model_validator(mode="after")
    def timing(self) -> Self:
        if not self.token_ids or len(self.token_ids) != len(self.client_arrivals_ms):
            raise ValueError("missing client token arrival measurements")
        if self.native_arrivals_ms and len(self.native_arrivals_ms) != len(self.token_ids):
            raise ValueError("incomplete native token arrival measurements")
        for arrivals in (self.client_arrivals_ms, self.native_arrivals_ms):
            if any(b < a for a, b in pairwise(arrivals)):
                raise ValueError("arrival clock moved backwards")
        if self.terminal_ms < self.client_arrivals_ms[-1]:
            raise ValueError("terminal preceded last token")
        return self


class MemoryExclusion(ProfileModel):
    profile: ProfileArtifact
    capacity: MemoryAmounts
    host: PhysicalBudget
    device: PhysicalBudget | None = None


class JobResult(ProfileModel):
    sweep_digest: Digest
    job_id: str
    candidate_id: str
    status: Literal["measured", "memory-excluded", "unknown", "oom", "correctness-failed"]
    warmups: tuple[Sample, ...] = ()
    sample: Sample | None = None
    # Paths/digests refer to raw executor evidence retained in the job directory.
    evidence_digest: Digest
    memory_exclusion: MemoryExclusion | None = None
    host_envelope_bytes: dict[str, int] = Field(default_factory=dict)
    device_envelope_bytes: dict[str, int] = Field(default_factory=dict)
    cleanup_passed: bool
    selected_health_passed: bool | None = None
    detail: str


def schedule(spec: SweepSpec, round_index: int) -> list[tuple[str, DeploymentPlan]]:
    candidates = list(placements(spec))
    random.Random(spec.seed + round_index).shuffle(candidates)
    selected = next(
        p for p in candidates if p.selected_candidate_id == spec.selected_plan.selected_candidate_id
    )
    return (
        [(f"r{round_index:03d}-reference-before", selected)]
        + [(f"r{round_index:03d}-c{i:03d}", p) for i, p in enumerate(candidates)]
        + [(f"r{round_index:03d}-reference-after", selected)]
    )


def validate_result(spec: SweepSpec, result: JobResult, job: str, plan: DeploymentPlan) -> None:
    if (result.sweep_digest, result.job_id, result.candidate_id) != (
        spec.sweep_digest,
        job,
        plan.selected_candidate_id,
    ):
        raise ValueError("result belongs to another job/sweep/placement")
    if result.status == "memory-excluded":
        e = result.memory_exclusion
        if (
            e is None
            or not isinstance(e.profile.measurement, MemoryMeasurement)
            or not e.profile.measurement.completed
        ):
            raise ValueError(
                "memory exclusion requires complete independent native memory evidence"
            )
        if (
            e.profile.key.assignment not in plan.stages
            or e.profile.key.workload != spec.workload
            or e.profile.key.checkpoint_digest != spec.reference.checkpoint_digest
            or e.profile.key.manifest_digest != spec.manifest.manifest_digest
            or e.profile.conditions.process_policy != "fresh-process-then-reloads"
            or e.profile.conditions.concurrent_load != spec.concurrent_load
            or assess_fit(e.profile, e.capacity, e.host, e.device).status != "unsafe"
        ):
            raise ValueError("unsupported independent memory exclusion")
    if result.status == "measured":
        if (
            not result.cleanup_passed
            or result.sample is None
            or len(result.warmups) != spec.warmups
        ):
            raise ValueError("measured result lacks successful warmups, timing or cleanup")
        if any(
            s.token_ids != spec.reference.generated_ids for s in (*result.warmups, result.sample)
        ):
            raise ValueError("divergent output cannot be a measured feasible result")


def _interval(values: list[float]) -> tuple[float, float]:
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def summarize(spec: SweepSpec, results: list[JobResult]) -> dict[str, object]:
    ids = [p.selected_candidate_id for p in placements(spec)]
    samples: dict[str, list[Sample]] = {cid: [] for cid in ids}
    statuses: dict[str, list[str]] = {cid: [] for cid in ids}
    refs: list[float] = []
    for result in results:
        if result.candidate_id not in samples or result.sweep_digest != spec.sweep_digest:
            raise ValueError("foreign sweep result")
        if "reference" in result.job_id:
            if result.status == "measured" and result.sample:
                refs.append(result.sample.client_arrivals_ms[-1])
            continue
        statuses[result.candidate_id].append(result.status)
        if result.status == "measured" and result.sample:
            samples[result.candidate_id].append(result.sample)
    measured = {
        cid: [s.client_arrivals_ms[-1] for s in rows] for cid, rows in samples.items() if rows
    }
    complete = all(
        (len(samples[cid]) >= spec.repetitions and set(statuses[cid]) == {"measured"})
        or (len(statuses[cid]) >= spec.repetitions and set(statuses[cid]) == {"memory-excluded"})
        for cid in ids
    )
    reference_jobs = [r for r in results if "reference" in r.job_id]
    drift = max(refs) / min(refs) - 1 if refs and min(refs) > 0 else None
    health = (
        all(
            r.selected_health_passed is True
            for r in reference_jobs
            if r.job_id.endswith("reference-after")
        )
        if reference_jobs
        else False
    )
    stable = (
        len(refs) >= 2 * spec.repetitions
        and len(refs) == len(reference_jobs)
        and drift is not None
        and drift <= spec.drift_limit
    )
    selected = spec.selected_plan.selected_candidate_id
    best = min(measured, key=lambda cid: (median(measured[cid]), cid)) if measured else None
    regret = (
        median(measured[selected]) / median(measured[best]) - 1
        if best and selected in measured and median(measured[best]) > 0
        else None
    )
    interval = None
    if complete and selected in measured:
        rng = random.Random(spec.seed)
        ratios: list[float] = []
        for _ in range(2000):
            medians = {
                cid: median(rng.choices(rows, k=len(rows))) for cid, rows in measured.items()
            }
            if min(medians.values()) > 0:
                ratios.append(medians[selected] / min(medians.values()) - 1)
        if ratios:
            interval = _interval(ratios)
    decision = "incomplete"
    if complete:
        decision = "unstable" if not stable else "inconclusive"
        if stable and interval and health:
            decision = (
                "pass"
                if interval[1] <= spec.regret_limit
                else "fail"
                if interval[0] > spec.regret_limit
                else "inconclusive"
            )
    return dict(
        sweep_digest=spec.sweep_digest,
        candidate_count=len(ids),
        complete_coverage=complete,
        selected_health_passed=health,
        decision=decision,
        selected_candidate_id=selected,
        best_candidate_id=best,
        observed_regret=regret,
        bootstrap_95_interval=interval,
        reference_drift_fraction=drift,
        prediction_errors={
            cid: prediction_error(
                spec.predictions.get(cid),
                rows,
                [
                    r
                    for r in results
                    if r.candidate_id == cid
                    and r.status == "measured"
                    and "reference" not in r.job_id
                ],
            )
            for cid, rows in samples.items()
            if rows
        },
        statistics={
            cid: dict(
                count=len(rows),
                median_generation_ms=median([s.client_arrivals_ms[-1] for s in rows]),
                median_ttft_ms=median([s.client_arrivals_ms[0] for s in rows]),
                median_itl_ms=median(
                    [b - a for s in rows for a, b in pairwise(s.client_arrivals_ms)]
                )
                if spec.workload.output_tokens > 1
                else 0,
                statuses=statuses[cid],
            )
            for cid, rows in samples.items()
            if rows
        },
        unresolved=[
            cid
            for cid in ids
            if not statuses[cid]
            or any(s not in ("measured", "memory-excluded") for s in statuses[cid])
        ],
        uncertainty_method=(
            "Seeded percentile bootstrap of selected / best over all measured "
            "candidates; conditional on observed samples. Drift is a separate gate."
        ),
    )


Executor = Callable[[SweepSpec, str, DeploymentPlan, Path], JobResult]


def run(
    spec: SweepSpec, directory: Path, execute: Executor, *, maximum_jobs: int | None = None
) -> dict[str, object]:
    results: list[JobResult] = []
    launched = 0
    for round_index in range(spec.maximum_repetitions):
        for job, plan in schedule(spec, round_index):
            job_dir = directory / job
            job_dir.mkdir(parents=True, exist_ok=True)
            output = job_dir / "result.json"
            if output.exists():
                saved = json.loads(output.read_text())
                if saved["digest"] != digest(saved["result"]):
                    raise ValueError("saved sweep result digest mismatch")
                result = JobResult.model_validate(saved["result"])
            else:
                if maximum_jobs is not None and launched >= maximum_jobs:
                    summary = summarize(spec, results)
                    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                    return summary
                result = execute(spec, job, plan, job_dir)
                validate_result(spec, result, job, plan)
                data = result.model_dump(mode="json")
                write_exclusive(output, {"result": data, "digest": digest(data)})
                launched += 1
            validate_result(spec, result, job, plan)
            evidence = json.loads((job_dir / "evidence.json").read_text())
            if digest(evidence) != result.evidence_digest:
                raise ValueError("raw sweep evidence was changed")
            results.append(result)
            if result.detail.startswith("interrupted"):
                return summarize(spec, results)
        summary = summarize(spec, results)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if round_index + 1 >= spec.repetitions and summary["decision"] != "inconclusive":
            return summary
    return summarize(spec, results)


def prediction_error(
    prediction: PerformanceEstimate | None, samples: list[Sample], results: list[JobResult]
) -> dict[str, object]:
    if prediction is None or prediction.ttft_ms is None or prediction.generation_ms is None:
        return {"status": "prediction-unavailable"}

    def difference(predicted: float, actual: float) -> dict[str, float | None]:
        return {
            "predicted": predicted,
            "observed": actual,
            "relative_error": predicted / actual - 1 if actual > 0 else None,
        }

    observed_itl = (
        median(
            [
                (s.client_arrivals_ms[-1] - s.client_arrivals_ms[0]) / (len(s.token_ids) - 1)
                for s in samples
                if len(s.token_ids) > 1
            ]
        )
        if len(samples[0].token_ids) > 1
        else 0
    )
    return dict(
        ttft_ms=difference(prediction.ttft_ms, median([s.client_arrivals_ms[0] for s in samples])),
        mean_itl_ms=difference(
            sum(prediction.decode_ms) / len(prediction.decode_ms) if prediction.decode_ms else 0,
            observed_itl,
        ),
        generation_ms=difference(
            prediction.generation_ms, median([s.client_arrivals_ms[-1] for s in samples])
        ),
        host_envelope_bytes={
            worker: difference(
                value,
                median(
                    [
                        r.host_envelope_bytes[worker]
                        for r in results
                        if worker in r.host_envelope_bytes
                    ]
                ),
            )
            for worker, value in prediction.host_envelope_bytes.items()
            if any(worker in r.host_envelope_bytes for r in results)
        },
        device_envelope_bytes={
            worker: difference(
                value,
                median(
                    [
                        r.device_envelope_bytes[worker]
                        for r in results
                        if worker in r.device_envelope_bytes
                    ]
                ),
            )
            for worker, value in prediction.device_envelope_bytes.items()
            if any(worker in r.device_envelope_bytes for r in results)
        },
        memory_basis=(
            "Conservative envelopes including safety/extra overhead; "
            "not allocator plus reservations or claimed physical peaks."
        ),
    )
