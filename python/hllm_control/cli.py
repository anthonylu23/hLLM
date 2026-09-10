"""Model preparation, placement, and native CPU generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DeploymentPlan, ModelManifest, PlanningMode, PlanningReport
from hllm_control.planner.config import load_links, load_settings, load_workers, load_workload
from hllm_control.planner.explain import explain_report
from hllm_control.planner.measured import read_profile_bundle
from hllm_control.planner.planner import create_plan
from hllm_control.prepare.manifest import HashMode, prepare_model
from hllm_control.serialization import read_artifact, write_artifact

app = typer.Typer(
    name="hllm",
    no_args_is_help=True,
    help="Prepare model manifests and create heterogeneous deployment plans.",
)


@app.command("prepare")
def prepare_command(
    model_path: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    model_id: Annotated[str | None, typer.Option(help="Stable source model identifier.")] = None,
    revision: Annotated[str | None, typer.Option(help="Pinned source repository revision.")] = None,
    hash_mode: Annotated[HashMode, typer.Option(case_sensitive=False)] = HashMode.METADATA,
) -> None:
    """Inspect a Llama-compatible or dense Qwen3 Safetensors model and emit its manifest."""
    manifest = prepare_model(
        model_path,
        model_id=model_id,
        revision=revision,
        hash_mode=hash_mode,
    )
    write_artifact(output, manifest)
    typer.echo(
        f"Wrote {output}: {manifest.config.num_layers} layers, "
        f"{len(manifest.tensors)} tensors, {manifest.total_storage_bytes} bytes"
    )


@app.command("plan")
def plan_command(
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    workers_path: Annotated[Path, typer.Option("--workers", exists=True, dir_okay=False)],
    workload_path: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    report_path: Annotated[Path, typer.Option("--report")],
    links_path: Annotated[Path | None, typer.Option("--links", exists=True, dir_okay=False)] = None,
    profile_bundle_path: Annotated[
        Path | None, typer.Option("--profile-bundle", exists=True, dir_okay=False)
    ] = None,
    mode: Annotated[PlanningMode | None, typer.Option("--mode")] = None,
    settings_path: Annotated[
        Path | None, typer.Option("--settings", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Enumerate all two-worker orders and contiguous split points."""
    manifest = read_artifact(manifest_path, ModelManifest)
    settings = load_settings(settings_path)
    if mode is not None:
        settings = settings.model_copy(update={"mode": mode})
    report = create_plan(
        manifest,
        load_workers(workers_path),
        load_links(links_path) if links_path else (),
        load_workload(workload_path),
        settings,
        profile_bundle=read_profile_bundle(profile_bundle_path)
        if profile_bundle_path
        else None,
    )
    write_artifact(report_path, report)
    if report.plan is None:
        typer.echo(explain_report(report), err=True)
        raise typer.Exit(2)
    write_artifact(output, report.plan)
    typer.echo(explain_report(report))
    typer.echo(f"Wrote plan to {output} and full report to {report_path}")


@app.command("explain")
def explain_command(
    report_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    rejected_limit: Annotated[int, typer.Option(min=0)] = 8,
) -> None:
    """Render an existing JSON planning report for a human."""
    report = read_artifact(report_path, PlanningReport)
    typer.echo(explain_report(report, rejected_limit=rejected_limit))


@app.command("generate")
def generate_command(
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    plan_path: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    workers_path: Annotated[Path, typer.Option("--workers", exists=True, dir_okay=False)],
    token_ids: Annotated[str, typer.Option(help="Comma-separated prompt token IDs.")],
    profile_bundle_path: Annotated[
        Path | None, typer.Option("--profile-bundle", exists=True, dir_okay=False)
    ] = None,
    maximum_new_tokens: Annotated[int, typer.Option("--max-new-tokens", min=1)] = 32,
    timeout: Annotated[float, typer.Option(min=0.001, max=3600)] = 60,
) -> None:
    """Load CPU stages, stream greedy token IDs, and unload the deployment."""
    manifest = read_artifact(manifest_path, ModelManifest)
    plan = read_artifact(plan_path, DeploymentPlan)
    workers = load_workers(workers_path)
    try:
        tokens = [int(item.strip()) for item in token_ids.split(",")]
    except ValueError as error:
        raise typer.BadParameter("token IDs must be comma-separated integers") from error
    with DeploymentSession(
        manifest,
        plan,
        {w.worker_id: w.endpoint for w in workers},
        profile_bundle=read_profile_bundle(profile_bundle_path)
        if profile_bundle_path
        else None,
    ) as session:
        events = session.generate(tokens, maximum_new_tokens=maximum_new_tokens, timeout=timeout)
        try:
            for event in events:
                if event.HasField("token"):
                    typer.echo(str(event.token.token_id) + " ", nl=False)
        finally:
            events.close()
    typer.echo()


@app.command("profile-memory")
def profile_memory_command(
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    plan_path: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    workload_path: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    model_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    binary: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    backend: Annotated[Backend, typer.Option()],
    output: Annotated[Path, typer.Option()],
    source_revision: Annotated[
        str, typer.Option(help="Revision/source snapshot used to build binary.")
    ],
    concurrent_load: Annotated[
        str, typer.Option(help="Observed competing workloads during probe.")
    ],
    stage_index: Annotated[int, typer.Option(min=0, max=1)] = 0,
    host_bytes: Annotated[int, typer.Option(min=0)] = 0,
    device_bytes: Annotated[int, typer.Option(min=0)] = 0,
    unified_bytes: Annotated[int, typer.Option(min=0)] = 0,
    pinned_bytes: Annotated[int, typer.Option(min=0)] = 0,
    pinned: Annotated[bool, typer.Option()] = False,
    device_id: Annotated[int, typer.Option(min=0)] = 0,
    warmup_cycles: Annotated[int, typer.Option(min=0, max=31)] = 0,
    cycles: Annotated[int, typer.Option(min=1, max=32)] = 2,
    timeout: Annotated[float, typer.Option(min=0.01, max=86400)] = 600,
    host_headroom_bytes: Annotated[int, typer.Option(min=0)] = 1024**3,
    device_headroom_bytes: Annotated[int, typer.Option(min=0)] = 512 * 1024**2,
    extra_overhead_bytes: Annotated[int, typer.Option(min=1)] = 256 * 1024**2,
) -> None:
    """Profile one assignment in a fresh native process; never connect to serving workers."""
    from hllm_control.profiling.models import MemoryAmounts
    from hllm_control.profiling.runner import run_memory_profile

    artifact = run_memory_profile(
        binary=binary,
        root=model_root,
        manifest=read_artifact(manifest_path, ModelManifest),
        plan=read_artifact(plan_path, DeploymentPlan),
        stage_index=stage_index,
        workload=load_workload(workload_path),
        backend=backend,
        capacity=MemoryAmounts(
            host=host_bytes, device=device_bytes, unified=unified_bytes, pinned=pinned_bytes
        ),
        output=output,
        source_revision=source_revision,
        concurrent_load=concurrent_load,
        transport_mode="pinned" if pinned else "pageable",
        device_id=device_id,
        warmup_cycles=warmup_cycles,
        measured_cycles=cycles,
        timeout_seconds=timeout,
        host_headroom_bytes=host_headroom_bytes,
        device_headroom_bytes=device_headroom_bytes,
        extra_overhead_bytes=extra_overhead_bytes,
    )
    from hllm_control.profiling.memory import FitResult

    fit = FitResult.model_validate(
        json.loads(output.with_suffix(".fit.json").read_text())["assessment"]
    )
    typer.echo(f"Wrote {output}: {artifact.artifact_digest}; physical fit: {fit.status}")
    if getattr(artifact.measurement, "completed", False) is not True:
        raise typer.Exit(2)
    if fit.status != "safe":
        raise typer.Exit(3)


@app.command("profile-compute")
def profile_compute_command(
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    plan_path: Annotated[Path, typer.Option("--plan", exists=True, dir_okay=False)],
    workload_path: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    model_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    binary: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    backend: Annotated[Backend, typer.Option()],
    output: Annotated[Path, typer.Option()],
    source_revision: Annotated[
        str, typer.Option(help="Revision/source snapshot used to build binary.")
    ],
    concurrent_load: Annotated[
        str, typer.Option(help="Observed competing workloads during probe.")
    ],
    stage_index: Annotated[int, typer.Option(min=0, max=1)] = 0,
    host_bytes: Annotated[int, typer.Option(min=0)] = 0,
    device_bytes: Annotated[int, typer.Option(min=0)] = 0,
    unified_bytes: Annotated[int, typer.Option(min=0)] = 0,
    pinned_bytes: Annotated[int, typer.Option(min=0)] = 0,
    pinned: Annotated[bool, typer.Option()] = False,
    device_id: Annotated[int, typer.Option(min=0)] = 0,
    warmup_cycles: Annotated[int, typer.Option(min=0, max=31)] = 2,
    cycles: Annotated[int, typer.Option(min=1, max=32)] = 5,
    timeout: Annotated[float, typer.Option(min=0.01, max=86400)] = 600,
    host_headroom_bytes: Annotated[int, typer.Option(min=0)] = 1024**3,
    device_headroom_bytes: Annotated[int, typer.Option(min=0)] = 512 * 1024**2,
    extra_overhead_bytes: Annotated[int, typer.Option(min=1)] = 256 * 1024**2,
) -> None:
    """Measure paired whole-stage/component timings and conversion on identical inputs."""
    from hllm_control.profiling.models import MemoryAmounts
    from hllm_control.profiling.runner import run_memory_profile

    artifact = run_memory_profile(
        mode="compute",
        binary=binary,
        root=model_root,
        manifest=read_artifact(manifest_path, ModelManifest),
        plan=read_artifact(plan_path, DeploymentPlan),
        stage_index=stage_index,
        workload=load_workload(workload_path),
        backend=backend,
        capacity=MemoryAmounts(
            host=host_bytes, device=device_bytes, unified=unified_bytes, pinned=pinned_bytes
        ),
        output=output,
        source_revision=source_revision,
        concurrent_load=concurrent_load,
        transport_mode="pinned" if pinned else "pageable",
        device_id=device_id,
        warmup_cycles=warmup_cycles,
        measured_cycles=cycles,
        timeout_seconds=timeout,
        host_headroom_bytes=host_headroom_bytes,
        device_headroom_bytes=device_headroom_bytes,
        extra_overhead_bytes=extra_overhead_bytes,
    )
    typer.echo(
        f"Wrote {output}: {artifact.artifact_digest}; "
        f"summary: {output.with_suffix('.summary.json')}"
    )
    if getattr(artifact.measurement, "completed", False) is not True:
        raise typer.Exit(2)


@app.command("serve-link-probe")
def serve_link_probe_command(
    binary: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    worker_id: Annotated[str, typer.Option()],
    listen: Annotated[str, typer.Option(help="Numeric IP:port to bind.")],
    source_revision: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option(help="New server configuration file.")],
    peer: Annotated[list[str], typer.Option(help="Allowed worker-id=IP:port; repeat per peer.")],
) -> None:
    """Start an isolated native transport probe in the foreground (Ctrl-C to stop)."""
    import os

    from hllm_control.profiling.link import server_config

    peers: dict[str, str] = {}
    for item in peer:
        name, separator, endpoint = item.partition("=")
        if not separator or not name or name in peers:
            raise typer.BadParameter("expected unique worker-id=IP:port peers")
        peers[name] = endpoint
    binary = binary.resolve(strict=True)
    server_config(
        binary=binary,
        worker_id=worker_id,
        listen=listen,
        peers=peers,
        source_revision=source_revision,
        output=config,
    )
    os.execv(str(binary), [str(binary), str(config.resolve())])


@app.command("profile-link")
def profile_link_command(
    source_endpoint: Annotated[str, typer.Option()],
    target_endpoint: Annotated[str, typer.Option()],
    target_worker_id: Annotated[str, typer.Option()],
    hidden_size: Annotated[int, typer.Option(min=1, max=8192)],
    output: Annotated[Path, typer.Option()],
    concurrent_load: Annotated[str, typer.Option()],
    tailscale: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    prompt_tokens: Annotated[int, typer.Option(min=1, max=4096)] = 512,
    output_tokens: Annotated[int, typer.Option(min=1, max=1024)] = 256,
    warmup_cycles: Annotated[int, typer.Option(min=0, max=31)] = 2,
    cycles: Annotated[int, typer.Option(min=1, max=32)] = 5,
    timeout: Annotated[float, typer.Option(min=0.01, max=120)] = 120,
) -> None:
    """Measure a directional native stream; run this command on the source host."""
    from hllm_control.profiling.link import run_link_profile

    artifact = run_link_profile(
        source_endpoint=source_endpoint,
        target_endpoint=target_endpoint,
        target_worker_id=target_worker_id,
        hidden_size=hidden_size,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        warmup_cycles=warmup_cycles,
        measured_cycles=cycles,
        output=output,
        concurrent_load=concurrent_load,
        tailscale=tailscale,
        timeout_seconds=timeout,
    )
    typer.echo(f"Wrote {output}: {artifact.artifact_digest}; qualified: {artifact.qualified}")
    if not artifact.qualified:
        typer.echo(artifact.error)
        raise typer.Exit(2)


if __name__ == "__main__":
    app()


@app.command("qualify-sweep")
def qualify_sweep_command(
    spec_path: Annotated[Path, typer.Option("--spec", exists=True, dir_okay=False)],
    executor_path: Annotated[Path, typer.Option("--executor", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
    maximum_jobs: Annotated[int | None, typer.Option("--max-jobs", min=1)] = None,
) -> None:
    """Freeze or resume independent all-split qualification with owned native workers."""
    from hllm_control.qualification.native import NativeExecutor
    from hllm_control.qualification.sweep import SweepContent, freeze, run

    content = read_artifact(spec_path, SweepContent)
    executor = read_artifact(executor_path, NativeExecutor)
    if content.executor_digest != executor.identity():
        raise typer.BadParameter("executor identity differs from frozen inputs")
    spec = freeze(content, output)
    summary = run(spec, output, executor, maximum_jobs=maximum_jobs)
    typer.echo(json.dumps(summary, indent=2))
    if summary["decision"] != "pass":
        raise typer.Exit(2)


@app.command("seal-profile-bundle")
def seal_profile_bundle_command(
    input_path: Annotated[Path, typer.Option("--input", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Validate and content-address an assembled BundleContent JSON document."""
    from hllm_control.planner.measured import BundleContent, seal_bundle
    from hllm_control.profiling.runner import write_exclusive

    bundle = seal_bundle(read_artifact(input_path, BundleContent))
    write_exclusive(output, bundle.model_dump(mode="json"))
    typer.echo(bundle.bundle_digest)


@app.command("freeze-sweep")
def freeze_sweep_command(
    report_path: Annotated[Path, typer.Option("--report", exists=True, dir_okay=False)],
    bundle_path: Annotated[Path, typer.Option("--profile-bundle", exists=True, dir_okay=False)],
    manifest_path: Annotated[Path, typer.Option("--manifest", exists=True, dir_okay=False)],
    reference_path: Annotated[Path, typer.Option("--reference", exists=True, dir_okay=False)],
    executor_path: Annotated[Path, typer.Option("--executor", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Bind automatic selection to an independent checkpoint reference before sweeping."""
    from hllm_control.profiling.models import digest
    from hllm_control.profiling.runner import write_exclusive
    from hllm_control.qualification.native import NativeExecutor
    from hllm_control.qualification.sweep import Reference, SweepContent
    from hllm_control.serialization import sha256_file

    report = read_artifact(report_path, PlanningReport)
    bundle = read_profile_bundle(bundle_path)
    manifest = read_artifact(manifest_path, ModelManifest)
    executor = read_artifact(executor_path, NativeExecutor)
    raw = json.loads(reference_path.read_text())
    if report.plan is None or report.plan.profile_bundle_digest != bundle.bundle_digest:
        raise typer.BadParameter("report must select a measured plan from this bundle")
    if raw.get("checkpoint_digest") != bundle.checkpoint_digest:
        raise typer.BadParameter("reference checkpoint identity differs from profile bundle")
    generation = raw["long_generation"]
    reference = Reference(
        manifest_digest=manifest.manifest_digest,
        checkpoint_digest=bundle.checkpoint_digest,
        producer=json.dumps(raw["producer"], sort_keys=True),
        producer_artifact_digest=sha256_file(reference_path),
        prompt_ids=tuple(generation["token_ids"]),
        generated_ids=tuple(generation["generated_ids"]),
    )
    content = SweepContent(
        manifest=manifest,
        workload=bundle.workload,
        reference=reference,
        selected_plan=report.plan,
        planning_report_digest=digest(report),
        profile_bundle_digest=bundle.bundle_digest,
        executor_digest=executor.identity(),
        concurrent_load=bundle.concurrent_load,
        predictions={
            c.candidate_id: c.performance for c in report.candidates if c.performance is not None
        },
    )
    write_exclusive(output, content.model_dump(mode="json"))
    typer.echo(f"Frozen selection inputs: {output}")
