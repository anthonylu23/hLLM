"""Failure-path coverage for qualification report publication and process sampling."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.validation import memory_watch, resource_guard
from scripts.validation.redact_report import redact


def test_report_redaction_keeps_evidence_without_network_identifiers() -> None:
    report = {
        "paths": [
            "pong from test-machine (100.64.0.2) via [2001:db8::1]:41641 in 52ms",
            "ip daddr 192.0.2.10/32 udp sport 41641 counter packets 17 bytes 4096 drop",
            "ip6 daddr 2001:db8:1::/64 counter packets 5 bytes 40 drop",
            "host.test-tailnet.ts.net.",
        ],
        "peak": 1234,
        "time": "12:34:56",
        "torch": "2.13.0",
        "hash": "a" * 64,
    }
    result = redact(report)
    assert result["paths"] == [
        "pong from peer (<IPv4 address>) via [<IPv6 address>]:41641 in 52ms",
        "ip daddr <IPv4 prefix/32> udp sport 41641 counter packets 17 bytes 4096 drop",
        "ip6 daddr <IPv6 prefix/64> counter packets 5 bytes 40 drop",
        "<tailnet hostname>",
    ]
    assert {k: v for k, v in result.items() if k != "paths"} == {
        k: v for k, v in report.items() if k != "paths"
    }


def test_cuda_sampler_recovers_after_failed_or_unavailable_samples(monkeypatch) -> None:
    run = Mock(
        side_effect=[
            subprocess.CalledProcessError(1, "nvidia-smi"),
            subprocess.CompletedProcess([], 0, stdout="42, [N/A]\n"),
            subprocess.TimeoutExpired("nvidia-smi", 5),
            subprocess.CompletedProcess([], 0, stdout="42, 128\n"),
        ]
    )
    monkeypatch.setattr(memory_watch.subprocess, "run", run)
    assert [memory_watch.cuda_memory_bytes(42) for _ in range(4)] == [
        None,
        None,
        None,
        128 * 1024**2,
    ]


def test_sampler_tolerates_process_exit_before_status_read(tmp_path, monkeypatch) -> None:
    output = tmp_path / "samples.jsonl"
    monkeypatch.setattr(sys, "argv", ["memory_watch", "42", str(output)])
    monkeypatch.setattr(
        memory_watch.subprocess,
        "run",
        Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0, stdout="1000\n"),
                subprocess.CompletedProcess([], 1, stdout=""),
            ]
        ),
    )
    monkeypatch.setattr(memory_watch.time, "sleep", lambda _: None)
    read_text = Path.read_text

    def disappearing_status(path, *args, **kwargs):
        if str(path) == "/proc/42/status":
            raise FileNotFoundError(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", disappearing_status)
    memory_watch.main()
    assert json.loads(output.read_text())["rss_bytes"] == 1000 * 1024


def test_memory_guard_enforces_pressure_availability_and_swap_window():
    guard = resource_guard.MemoryGuard(100, 256, 16)
    assert guard.check(0, dict(available_bytes=1000, swapout_bytes=0, pressure=2)) is None
    assert guard.check(15, dict(available_bytes=1000, swapout_bytes=0, pressure=2)) is None
    assert "pressure" in str(
        guard.check(16, dict(available_bytes=1000, swapout_bytes=0, pressure=2))
    )
    assert guard.check(17, dict(available_bytes=1000, swapout_bytes=0, pressure=1)) is None
    assert guard.check(18, dict(available_bytes=1000, swapout_bytes=0, pressure=2)) is None
    assert "floor" in str(guard.check(19, dict(available_bytes=99, swapout_bytes=0)))
    # A burst straddling the 60-second boundary must not disappear when pruning.
    guard = resource_guard.MemoryGuard(100, 256, 16)
    assert guard.check(0, dict(available_bytes=1000, swapout_bytes=0)) is None
    assert guard.check(59, dict(available_bytes=1000, swapout_bytes=200)) is None
    assert "swap-out" in str(guard.check(61, dict(available_bytes=1000, swapout_bytes=300)))


def test_guard_does_not_spawn_when_evidence_would_be_overwritten(tmp_path, monkeypatch):
    import pytest

    output = tmp_path / "guard.jsonl"
    output.write_text("original evidence")
    monkeypatch.setattr(sys, "argv", ["guard", "--record", str(output), "--", sys.executable])
    spawn = Mock()
    monkeypatch.setattr(resource_guard.subprocess, "Popen", spawn)
    with pytest.raises(FileExistsError):
        resource_guard.main()
    spawn.assert_not_called()
    assert output.read_text() == "original evidence"


def test_guard_timeout_reaps_owned_child(tmp_path, monkeypatch):
    import os

    import pytest

    output = tmp_path / "guard.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guard",
            "--record",
            str(output),
            "--seconds",
            "0.1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
    )
    monkeypatch.setattr(
        resource_guard,
        "system_memory",
        lambda: dict(available_bytes=4 * 1024**3, swapout_bytes=0, pressure=1),
    )
    assert resource_guard.main() == 124
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[-1]["reason"] == "experiment time limit"
    with pytest.raises(ProcessLookupError):
        os.kill(rows[1]["pid"], 0)


def test_guard_sampling_failure_reaps_owned_child(tmp_path, monkeypatch):
    import os

    import pytest

    output = tmp_path / "guard.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "guard",
            "--record",
            str(output),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
        ],
    )
    monkeypatch.setattr(
        resource_guard,
        "system_memory",
        Mock(
            side_effect=[
                dict(available_bytes=4 * 1024**3, swapout_bytes=0, pressure=1),
                RuntimeError("sampler unavailable"),
            ]
        ),
    )
    with pytest.raises(RuntimeError, match="sampler unavailable"):
        resource_guard.main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert "sampler unavailable" in rows[-1]["reason"]
    with pytest.raises(ProcessLookupError):
        os.kill(rows[1]["pid"], 0)


@pytest.mark.parametrize("defect", ["config", "shard", "digest", "token", "producer", "capacity"])
def test_reload_harness_rejects_mismatched_reference_before_execution(tmp_path, defect):
    from hllm_control.prepare.manifest import HashMode, prepare_model
    from hllm_control.profiling.models import digest

    from scripts.validation.reload_soak import reference_tokens
    from tests.process_helpers import write_model

    write_model(tmp_path)
    manifest = prepare_model(tmp_path, hash_mode=HashMode.FULL)
    files = {
        "config.json": manifest.source.config_sha256,
        **{f.name: f.sha256 for f in manifest.tensor_files},
    }
    reference: dict = dict(
        checkpoint_files=files,
        checkpoint_digest=digest(files),
        producer=dict(dtype="f16", attention="eager", tf32=False),
        long_generation=dict(token_ids=[1, 2], generated_ids=[3, 4, 5, 6]),
    )
    assert reference_tokens(reference, manifest) == ([1, 2], [3, 4, 5, 6])
    if defect == "config":
        files["config.json"] = "a" * 64
    elif defect == "shard":
        files[manifest.tensor_files[0].name] = "a" * 64
    elif defect == "digest":
        reference["checkpoint_digest"] = "a" * 64
    elif defect == "token":
        reference["long_generation"]["token_ids"] = [True]
    elif defect == "producer":
        reference["producer"]["tf32"] = True
    else:
        reference["long_generation"]["token_ids"] = [1] * manifest.config.maximum_sequence_length
    with pytest.raises(ValueError):
        reference_tokens(reference, manifest)


@pytest.mark.parametrize("interruption", ["SIGTERM", "SIGINT", "SIGHUP", "leader-exit"])
def test_guard_retires_process_tree_on_interruption(tmp_path, interruption):
    import os
    import signal
    import time

    pidfile = tmp_path / "descendant.pid"
    leader = tmp_path / "leader.py"
    # The descendant deliberately ignores TERM. The guard must kill the whole
    # process group even if the leader exits first.
    leader.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(child.pid))\n"
        + ("time.sleep(0.3)\n" if interruption == "leader-exit" else "time.sleep(60)\n")
    )
    record = tmp_path / "guard.jsonl"
    # Fixed resources isolate signal behavior from the developer machine's load.
    code = (
        "from scripts.validation import resource_guard as g; "
        "g.system_memory=lambda:dict(available_bytes=4*1024**3,swapout_bytes=0,pressure=1); "
        "raise SystemExit(g.main())"
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            code,
            "--record",
            str(record),
            "--seconds",
            "10",
            "--",
            sys.executable,
            str(leader),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    descendant = None
    try:
        until = time.monotonic() + 5
        while not pidfile.exists() or not pidfile.read_text():
            assert time.monotonic() < until and process.poll() is None
            time.sleep(0.02)
        descendant = int(pidfile.read_text())
        if interruption != "leader-exit":
            process.send_signal(getattr(signal, interruption))
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == (0 if interruption == "leader-exit" else 124), stderr
        rows = [json.loads(line) for line in record.read_text().splitlines()]
        assert rows[-1]["event"] == "exit"
        if interruption != "leader-exit":
            assert "received signal" in rows[-1]["reason"]
        until = time.monotonic() + 3
        while True:
            status = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(descendant)], capture_output=True, text=True
            ).stdout.strip()
            # An orphan zombie cannot execute; its reaping belongs to the host init.
            if not status or status.startswith("Z"):
                break
            assert time.monotonic() < until, status
            time.sleep(0.02)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)
        if descendant is not None:
            try:
                os.kill(descendant, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize("refused", [True, False])
def test_cpu_rehearsal_records_guard_preflight_rejection(tmp_path, monkeypatch, refused):
    from scripts.validation import cpu_rehearsal

    monkeypatch.setenv("HLLM_CPU_WORKER", sys.executable)
    monkeypatch.setenv("HLLM_MEMORY_PROFILER", sys.executable)
    monkeypatch.setattr(sys, "argv", ["rehearsal", "--output-root", str(tmp_path)])
    error = (resource_guard.PreflightRefused if refused else RuntimeError)(
        "Mac pressure is already non-normal"
    )
    monkeypatch.setattr(cpu_rehearsal.resource_guard, "main", Mock(side_effect=error))
    # Only a guard refusal maps to the CTest skip code; other failures stay failures.
    code = cpu_rehearsal.SKIPPED if refused else 125
    assert code == (77 if refused else 125)
    assert cpu_rehearsal.main() == code
    report = json.loads(next(tmp_path.glob("*/result.json")).read_text())
    assert report["exit_code"] == code and report["scope"] == "cpu-rehearsal"
    assert not report["accelerator_qualified"]
    assert "pressure" in report["skipped" if refused else "error"]
    assert ("error" in report) != refused


@pytest.mark.parametrize("condition", ["pressure", "availability"])
def test_guard_preflight_refusal_records_reason_without_spawning(tmp_path, monkeypatch, condition):
    output = tmp_path / "guard.jsonl"
    monkeypatch.setattr(sys, "argv", ["guard", "--record", str(output), "--", sys.executable])
    baseline = (
        dict(available_bytes=4 * 1024**3, swapout_bytes=0, pressure=2)
        if condition == "pressure"
        else dict(available_bytes=1024**3 - 1, swapout_bytes=0, pressure=1)
    )
    monkeypatch.setattr(resource_guard, "system_memory", lambda: baseline)
    spawn = Mock()
    monkeypatch.setattr(resource_guard.subprocess, "Popen", spawn)
    expected = "already non-normal" if condition == "pressure" else "below configured floor"
    with pytest.raises(resource_guard.PreflightRefused, match=expected):
        resource_guard.main()
    spawn.assert_not_called()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["preflight"]
    assert (
        expected in rows[0]["reason"] and rows[0]["available_bytes"] == baseline["available_bytes"]
    )


def test_stop_group_tolerates_an_unreapable_leader(monkeypatch):
    import subprocess

    monkeypatch.setattr(resource_guard.os, "killpg", Mock())
    child = Mock(pid=12345)
    child.wait.side_effect = subprocess.TimeoutExpired(cmd="leader", timeout=5)
    resource_guard.stop_group(child)  # must not raise from the guard's finally block
    assert child.wait.call_count == 2


def test_reload_harness_credits_only_inactive_device_allocator_cache() -> None:
    from hllm_control.proto import control_pb2, profile_pb2

    from scripts.validation.reload_soak import effective_device_availability

    state = control_pb2.QualificationState(available_device_bytes=2_000)
    device_cache = control_pb2.WorkerMetrics(
        allocator=control_pb2.AllocatorMetrics(
            domain=profile_pb2.MEMORY_DOMAIN_DEVICE, active_bytes=10, cached_bytes=5_000
        )
    )
    assert effective_device_availability(state, device_cache) == dict(
        reported_available_bytes=2_000,
        same_process_cached_allocator_bytes=5_000,
        effective_available_bytes=7_000,
    )
    host_cache = control_pb2.WorkerMetrics(
        allocator=control_pb2.AllocatorMetrics(
            domain=profile_pb2.MEMORY_DOMAIN_HOST, cached_bytes=5_000
        )
    )
    assert effective_device_availability(state, host_cache)["effective_available_bytes"] == 2_000
    assert (
        effective_device_availability(state, control_pb2.WorkerMetrics())[
            "effective_available_bytes"
        ]
        == 2_000
    )
    assert effective_device_availability(control_pb2.QualificationState(), device_cache) is None
