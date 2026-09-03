from __future__ import annotations

from pathlib import Path

from hllm_control.cli import app
from hllm_control.models import ModelManifest, PlanningReport
from hllm_control.serialization import read_artifact
from typer.testing import CliRunner


def test_prepare_plan_and_explain_cli(tiny_model: Path, tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    runner = CliRunner()
    manifest_path = tmp_path / "manifest.json"
    plan_path = tmp_path / "plan.json"
    report_path = tmp_path / "report.json"

    prepared = runner.invoke(
        app,
        [
            "prepare",
            str(tiny_model),
            "--model-id",
            "test/tiny-llama",
            "--revision",
            "deadbeef",
            "--output",
            str(manifest_path),
        ],
    )
    assert prepared.exit_code == 0, prepared.output
    assert read_artifact(manifest_path, ModelManifest).config.num_layers == 4

    planned = runner.invoke(
        app,
        [
            "plan",
            "--manifest",
            str(manifest_path),
            "--workers",
            str(root / "examples/profiles/workers-m3pro-3060ti.yaml"),
            "--links",
            str(root / "examples/profiles/links-m3pro-3060ti.yaml"),
            "--workload",
            str(root / "examples/workloads/interactive.yaml"),
            "--output",
            str(plan_path),
            "--report",
            str(report_path),
        ],
    )
    assert planned.exit_code == 0, planned.output
    report = read_artifact(report_path, PlanningReport)
    assert report.plan is not None
    assert len(report.candidates) == 6
    assert plan_path.exists()

    explained = runner.invoke(app, ["explain", str(report_path)])
    assert explained.exit_code == 0, explained.output
    assert "Selected:" in explained.output
