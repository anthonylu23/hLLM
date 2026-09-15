from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hllm_control.cli import app
from hllm_control.models import ModelManifest, PlanningReport
from hllm_control.serialization import read_artifact
from typer.testing import CliRunner


def test_module_and_installed_cli_register_all_commands() -> None:
    expected = (
        "prepare",
        "plan",
        "explain",
        "generate",
        "profile-memory",
        "profile-compute",
        "serve-link-probe",
        "profile-link",
        "qualify-sweep",
        "seal-profile-bundle",
        "freeze-sweep",
    )
    for command in (
        [sys.executable, "-m", "hllm_control.cli"],
        [str(Path(sys.executable).with_name("hllm"))],
    ):
        help_result = subprocess.run(
            [*command, "--help"], capture_output=True, text=True, check=True, timeout=10
        )
        for name in expected:
            assert name in help_result.stdout
        for name in ("qualify-sweep", "seal-profile-bundle", "freeze-sweep"):
            subprocess.run([*command, name, "--help"], capture_output=True, check=True, timeout=10)


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
