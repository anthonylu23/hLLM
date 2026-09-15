"""Summarize paired native timings without hiding instrumentation overhead."""

from __future__ import annotations

from statistics import median

from hllm_control.profiling.models import ComputeRunMeasurement, ProfileArtifact


def summarize_compute(artifact: ProfileArtifact) -> dict[str, object]:
    m = artifact.measurement
    if not isinstance(m, ComputeRunMeasurement):
        raise ValueError("expected a compute run")
    groups: dict[tuple[str, int, str, int | None, str], list[float]] = {}
    for r in m.records:
        if r.cycle < artifact.conditions.warmup_cycles:
            continue
        group = (r.phase, r.context_tokens, r.component, r.layer_index, r.timing_mode)
        groups.setdefault(group, []).append(r.milliseconds)
    buckets = [
        {
            "phase": g[0],
            "context_tokens": g[1],
            "component": g[2],
            "layer_index": g[3],
            "timing_mode": g[4],
            "samples_ms": values,
            "median_ms": median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for g, values in groups.items()
    ]
    comparisons: list[dict[str, object]] = []
    for (phase, context, component, _layer, mode), values in groups.items():
        if component != "stage" or mode != "whole-stage":
            continue
        key = (phase, context, component, None, "component-synchronized")
        instrumented = groups.get(key)
        if instrumented:
            baseline = median(values)
            detailed = median(instrumented)
            comparisons.append(
                {
                    "phase": phase,
                    "context_tokens": context,
                    "whole_stage_ms": baseline,
                    "instrumented_ms": detailed,
                    "overhead_fraction": detailed / baseline - 1 if baseline > 0 else None,
                }
            )
    return {
        "schema_version": "1.0",
        "profile_digest": artifact.artifact_digest,
        "completed": m.completed,
        "error": m.error,
        "warmup_cycles_excluded": artifact.conditions.warmup_cycles,
        "buckets": buckets,
        "instrumentation_comparison": comparisons,
        "notes": [
            "Whole-stage timings include boundary conversion; do not add conversion twice.",
            "Component-synchronized timings are diagnostic; "
            "use measured overhead before prediction.",
            "Every decode context is measured; no context interpolation is assumed.",
        ],
    }
