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
        for name in ("qualify-sweep", "seal-profile-bundle", "freeze-sweep", "profile-memory"):
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


def test_profile_memory_cli_forwards_policy_and_reports_fallback(tmp_path: Path, monkeypatch):
    import json
    import shutil
    from types import SimpleNamespace

    from hllm_control.profiling import runner
    from hllm_control.profiling.memory import FitResult, MlxFitPolicy

    from tests.process_helpers import plan, write_model

    manifest = write_model(tmp_path / "model")
    (tmp_path / "manifest.json").write_text(manifest.model_dump_json())
    (tmp_path / "plan.json").write_text(plan(manifest).model_dump_json())
    root = Path(__file__).parents[2]
    shutil.copy(root / "examples/workloads/interactive.yaml", tmp_path / "workload.yaml")
    (tmp_path / "binary").write_text("")
    seen = []

    def fake_profile(**kwargs):
        seen.append(kwargs["mlx_fit_policy"])
        fit = FitResult(
            status="safe",
            reasons=(),
            host_envelope_bytes=10,
            device_envelope_bytes=None,
            policy="mlx-rss-plus-allocator-v1",
            requested_mlx_policy=kwargs["mlx_fit_policy"],
            policy_notes=("footprint policy requires a complete schema-1.3 memory exercise",)
            if kwargs["mlx_fit_policy"] == MlxFitPolicy.FOOTPRINT
            else (),
        )
        kwargs["output"].with_suffix(".fit.json").write_text(
            json.dumps({"assessment": fit.model_dump()})
        )
        return SimpleNamespace(
            artifact_digest="a" * 64, measurement=SimpleNamespace(completed=True)
        )

    monkeypatch.setattr(runner, "run_memory_profile", fake_profile)
    base = [
        "profile-memory",
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--plan",
        str(tmp_path / "plan.json"),
        "--workload",
        str(tmp_path / "workload.yaml"),
        "--model-root",
        str(tmp_path / "model"),
        "--binary",
        str(tmp_path / "binary"),
        "--backend",
        "mlx",
        "--output",
        str(tmp_path / "out.json"),
        "--source-revision",
        "test",
        "--concurrent-load",
        "idle",
    ]
    runner_ = CliRunner()
    result = runner_.invoke(app, [*base, "--mlx-fit-policy", "footprint-v1"])
    assert result.exit_code == 0, result.output
    assert seen == [MlxFitPolicy.FOOTPRINT]
    assert "policy: mlx-rss-plus-allocator-v1; requested: footprint-v1" in result.output
    assert "policy note: footprint policy requires" in result.output
    result = runner_.invoke(app, base)
    assert result.exit_code == 0, result.output
    assert seen[-1] == MlxFitPolicy.FOOTPRINT  # default since the October 10 promotion
    result = runner_.invoke(app, [*base, "--mlx-fit-policy", "conservative-v1"])
    assert result.exit_code == 0, result.output
    assert seen[-1] == MlxFitPolicy.CONSERVATIVE and "policy note" not in result.output
    assert runner_.invoke(app, [*base, "--mlx-fit-policy", "bogus-v9"]).exit_code != 0
