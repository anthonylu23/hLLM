"""Real CPU/MLX pipeline parity; requires both native binaries and a Metal GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
from hllm_control.controller import DeploymentSession

from tests.mlx.helpers import mixed_workers
from tests.process_helpers import ROOT, Workers, plan, tokens, wait_clean, write_model


@pytest.mark.parametrize("family", ["llama", "qwen3"])
def test_mixed_all_splits_storage_and_heads(tmp_path: Path, family: str) -> None:
    for storage, tied in (("F32", False), ("F32", True), ("F16", True), ("BF16", False)):
        manifest = write_model(tmp_path, family, storage, tied=tied)
        expected = {}
        with Workers(tmp_path) as reference:
            for split in range(1, manifest.config.num_layers):
                with DeploymentSession(
                    manifest, plan(manifest, split), reference.endpoints
                ) as session:
                    expected[split] = tokens(session)
        with mixed_workers(tmp_path) as workers:
            for split, output in expected.items():
                for reverse in (False, True):
                    with DeploymentSession(
                        manifest, plan(manifest, split, reverse), workers.endpoints
                    ) as session:
                        assert tokens(session) == output
                        assert tokens(session) == output
                        assert tokens(session, stop_token_ids=[output[0]]) == output[:1]
                    wait_clean(workers, loaded=False)


def test_mixed_long_decode(tmp_path: Path) -> None:
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as reference:
        with DeploymentSession(manifest, plan(manifest), reference.endpoints) as session:
            expected = tokens(session, count=256)
    with mixed_workers(tmp_path) as workers:
        for reverse in (False, True):
            with DeploymentSession(
                manifest, plan(manifest, reverse=reverse), workers.endpoints
            ) as session:
                assert tokens(session, count=256) == expected
            wait_clean(workers, loaded=False)


def test_mixed_cli_profile(tmp_path: Path) -> None:
    from hllm_control.cli import app
    from typer.testing import CliRunner

    write_model(tmp_path)
    with mixed_workers(tmp_path) as workers:
        config = (ROOT / "examples/profiles/workers-cpu-mlx-local.yaml").read_text()
        for name, default in (("cpu-a", "127.0.0.1:50051"), ("cpu-b", "127.0.0.1:50053")):
            config = config.replace(default, workers.endpoints[name])
        worker_path = tmp_path / "workers.yaml"
        worker_path.write_text(config)
        manifest_path, plan_path = tmp_path / "manifest.json", tmp_path / "plan.json"
        runner = CliRunner()
        result = runner.invoke(app, ["prepare", str(tmp_path), "--output", str(manifest_path)])
        assert result.exit_code == 0, result.output
        result = runner.invoke(
            app,
            [
                "plan",
                "--manifest",
                str(manifest_path),
                "--workers",
                str(worker_path),
                "--links",
                str(ROOT / "examples/profiles/links-cpu-local.yaml"),
                "--workload",
                str(ROOT / "examples/workloads/cpu-demo.yaml"),
                "--settings",
                str(ROOT / "examples/profiles/planner-cpu.yaml"),
                "--output",
                str(plan_path),
                "--report",
                str(tmp_path / "report.json"),
            ],
        )
        assert result.exit_code == 0, result.output
        result = runner.invoke(
            app,
            [
                "generate",
                "--manifest",
                str(manifest_path),
                "--plan",
                str(plan_path),
                "--workers",
                str(worker_path),
                "--token-ids",
                "1,4,2",
                "--max-new-tokens",
                "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert len([int(value) for value in result.output.split()]) == 4
        wait_clean(workers, loaded=False)
