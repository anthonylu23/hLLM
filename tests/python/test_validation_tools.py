"""Failure-path coverage for qualification report publication and process sampling."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

from scripts.validation import memory_watch
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
