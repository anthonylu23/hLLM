"""Human-readable planning reports."""

from __future__ import annotations

from hllm_control.models import PlanCandidate, PlanningReport, StageMemory


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _stage_line(stage: StageMemory) -> str:
    return (
        f"  {stage.worker_id}: layers [{stage.layer_start},{stage.layer_end}), "
        f"{_gib(stage.required_bytes)} / {_gib(stage.usable_bytes)} "
        f"{stage.domain.value.lower()} ({stage.pressure:.1%})"
    )


def _candidate_summary(candidate: PlanCandidate) -> str:
    if candidate.feasible:
        detail = f"rank {candidate.rank}, score {candidate.score:.4f}"
        if candidate.performance and candidate.performance.generation_ms is not None:
            detail += (
                f", TTFT {candidate.performance.ttft_ms:.2f} ms, "
                f"generation {candidate.performance.generation_ms:.2f} ms"
            )
        return detail
    return (
        candidate.measurement_status + ": " if candidate.measurement_status else ""
    ) + ", ".join(candidate.rejection_reasons)


def explain_report(report: PlanningReport, *, rejected_limit: int = 8) -> str:
    lines = [
        f"Manifest: {report.manifest_digest}",
        f"Workload: {report.workload_id}",
        f"Mode: {report.mode.value}",
    ]
    if report.selected_candidate_id is None:
        lines.append("Selected: none")
    else:
        selected = next(
            item for item in report.candidates if item.candidate_id == report.selected_candidate_id
        )
        lines.extend(
            [
                "",
                f"Selected: {selected.stage_zero_worker_id} -> "
                f"{selected.final_stage_worker_id}, split at layer {selected.split_layer}",
                *(_stage_line(stage) for stage in selected.stages),
                f"  {_candidate_summary(selected)}",
            ]
        )
        runners_up = sorted(
            (
                item
                for item in report.candidates
                if item.feasible and item.candidate_id != report.selected_candidate_id
            ),
            key=lambda item: item.rank or 10**9,
        )[:3]
        if runners_up:
            lines.extend(["", "Runners-up:"])
            lines.extend(
                f"  {item.candidate_id}: {_candidate_summary(item)}" for item in runners_up
            )

    rejected = [item for item in report.candidates if not item.feasible]
    if rejected:
        lines.extend(["", f"Rejected candidates ({len(rejected)}):"])
        lines.extend(
            f"  {item.candidate_id}: {_candidate_summary(item)}"
            for item in rejected[:rejected_limit]
        )
        if len(rejected) > rejected_limit:
            lines.append(f"  ... {len(rejected) - rejected_limit} more in the JSON report")
    if report.notes:
        lines.extend(["", "Notes:"])
        lines.extend(f"  {note}" for note in report.notes)
    return "\n".join(lines) + "\n"
