"""Launcher behavior for scripts/validation/two_host, with SSH replaced by local commands."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TWO_HOST = Path(__file__).resolve().parents[2] / "scripts" / "validation" / "two_host"


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def _matching_pids(tmp_path: Path) -> set[int]:
    env_file = tmp_path / "empty.env"
    env_file.write_text("")
    result = subprocess.run(
        [
            "bash",
            "-c",
            'ENV_FILE=$1; . "$2"; pgrep -f "$HLLM_PROCESS_PATTERN" || true',
            "check",
            str(env_file),
            str(TWO_HOST / "lib.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {int(line) for line in result.stdout.split()}


@pytest.mark.skipif(shutil.which("pgrep") is None, reason="needs pgrep")
def test_running_worker_is_found_by_full_name_on_any_port(tmp_path: Path) -> None:
    # Linux truncates process names to 15 characters; this name has 16.
    worker = tmp_path / "build" / "cpp" / "hllm-worker-cuda"
    _executable(worker, "#!/bin/sh\nwhile :; do sleep 1; done\n")
    log = tmp_path / "hllm-worker-cuda.log"
    processes = [
        subprocess.Popen(["sh", str(worker), "--listen", "127.0.0.1:59999"]),
        subprocess.Popen(["sh", "-c", "sleep 60; :", str(log)]),
    ]
    try:
        found = _matching_pids(tmp_path)
        assert processes[0].pid in found
        assert processes[1].pid not in found
    finally:
        for process in processes:
            process.kill()
            process.wait()


def _fake_hosts(tmp_path: Path, profile_status: int, fit_status: str) -> tuple[Path, dict]:
    """A run directory, a Mac checkout and a PATH whose ssh/scp act on local paths."""
    bin_dir = tmp_path / "bin"
    _executable(
        bin_dir / "ssh",
        '#!/bin/bash\nwhile [ "$1" = -o ]; do shift 2; done\nshift\nexec bash -c "$*"\n',
    )
    _executable(
        bin_dir / "scp",
        '#!/bin/bash\nargs=()\nfor a in "$@"; do case $a in\n'
        "  -q) ;;\n  fake-mac:*) args+=(${a#fake-mac:}) ;;\n"
        '  *) args+=("$a") ;;\nesac; done\nexec cp "${args[@]}"\n',
    )
    for name in ("vm_stat", "sysctl", "caffeinate"):
        _executable(bin_dir / name, "#!/bin/sh\n")
    # profile.sh polls the Mac every 15 seconds.
    _executable(bin_dir / "sleep", "#!/bin/sh\nexec /bin/sleep 0.1\n")

    mac_repo = tmp_path / "mac-repo"
    scripts = mac_repo / "scripts" / "validation"
    (scripts / "two_host").mkdir(parents=True)
    for name in ("lib.sh", "mlx-profile.sh"):
        shutil.copy2(TWO_HOST / name, scripts / "two_host" / name)
    (scripts / "source_identity.py").write_text("print('digest')\n")
    # Like the profiler, write the artifact first and then exit with the refusal status.
    (scripts / "resource_guard.py").write_text(
        "import json, sys\n"
        "output = sys.argv[sys.argv.index('--output') + 1]\n"
        "json.dump({'measurement': {'completed': True}}, open(output, 'w'))\n"
        "fit = output.removesuffix('.json') + '.fit.json'\n"
        f"json.dump({{'assessment': {{'status': {fit_status!r}}}}}, open(fit, 'w'))\n"
        f"sys.exit({profile_status})\n"
    )
    source = tmp_path / "source"
    for checkout in (mac_repo, source):
        (checkout / ".venv" / "bin").mkdir(parents=True)
        (checkout / ".venv" / "bin" / "python").symlink_to(sys.executable)

    run = tmp_path / "run"
    (run / "inputs").mkdir(parents=True)
    plan = {
        "stages": [{"stage_index": 0, "worker_id": "cuda"}, {"stage_index": 1, "worker_id": "mlx"}]
    }
    (run / "inputs" / "plan.json").write_text(json.dumps(plan))
    (run / "inputs" / "manifest.json").write_text("{}")
    (run / "inputs" / "workload.yaml").write_text("{}")
    settings = {
        "RUN_DIR": run,
        "SOURCE_DIR": source,
        "MAC_SSH": "fake-mac",
        "MAC_REPO": mac_repo,
        "MAC_RUN_DIR": tmp_path / "mac-run",
        "MANIFEST": "inputs/manifest.json",
        "PLAN": "inputs/plan.json",
        "WORKLOAD": "inputs/workload.yaml",
        "MLX_BUILD_DIR": tmp_path / "mlx-build",
        "MLX_MODEL_ROOT": tmp_path / "model",
        "MLX_LIMIT_BYTES": 1,
        "PROFILE_CYCLES": 1,
        "PROFILE_GUARD_SECONDS": 200,
        "CONCURRENT_LOAD": "none",
    }
    env_file = run / "two-host.env"
    env_file.write_text("".join(f"{key}='{value}'\n" for key, value in settings.items()))
    environment = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    return env_file, environment


@pytest.mark.parametrize(
    ("profile_status", "fit_status", "expected"),
    [(0, "safe", 0), (2, "safe", 2), (3, "unsafe", 3), (0, "unsafe", 1)],
)
def test_mlx_profile_reports_the_profiler_refusal(
    tmp_path: Path, profile_status: int, fit_status: str, expected: int
) -> None:
    env_file, environment = _fake_hosts(tmp_path, profile_status, fit_status)
    result = subprocess.run(
        ["bash", str(TWO_HOST / "profile.sh"), str(env_file), "mlx"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected, result.stderr
    # The artifacts are still copied back for diagnosis.
    assert (tmp_path / "run" / "mac" / "mlx-memory.json").is_file()
    assert (tmp_path / "run" / "mac" / "mlx-profile.exit").read_text().strip() == str(
        profile_status
    )
