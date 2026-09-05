"""Model preparation, placement, and native CPU generation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from hllm_control.controller import DeploymentSession
from hllm_control.models import DeploymentPlan, ModelManifest, PlanningReport
from hllm_control.planner.config import load_links, load_settings, load_workers, load_workload
from hllm_control.planner.explain import explain_report
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
    links_path: Annotated[Path, typer.Option("--links", exists=True, dir_okay=False)],
    workload_path: Annotated[Path, typer.Option("--workload", exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    report_path: Annotated[Path, typer.Option("--report")],
    settings_path: Annotated[
        Path | None, typer.Option("--settings", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """Enumerate all two-worker orders and contiguous split points."""
    manifest = read_artifact(manifest_path, ModelManifest)
    report = create_plan(
        manifest,
        load_workers(workers_path),
        load_links(links_path),
        load_workload(workload_path),
        load_settings(settings_path),
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
    with DeploymentSession(manifest, plan, {w.worker_id: w.endpoint for w in workers}) as session:
        events = session.generate(tokens, maximum_new_tokens=maximum_new_tokens, timeout=timeout)
        try:
            for event in events:
                if event.HasField("token"):
                    typer.echo(str(event.token.token_id) + " ", nl=False)
        finally:
            events.close()
    typer.echo()


if __name__ == "__main__":
    app()
