"""Host-identifier redaction of committed evidence, including sealed link archives."""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from hllm_control.profiling.link import LinkArtifact, LinkContent
from hllm_control.profiling.models import digest

from scripts.validation.redact_report import Redactor, dump, json_format, redact, redact_files

ROOT = Path(__file__).parents[2]
NOW = datetime(2026, 9, 8, tzinfo=UTC)
CUDA_TAILNET, MAC_TAILNET = "100.100.1.1", "100.100.2.2"
MAC_PUBLIC = "9.9.9.9"
MAC_DNS = "10-1-2-3.dynamic.example.edu"
NODE = "nAbCdEf123CNTRL"


def identity(worker_id: str, host: str, ip: str) -> dict[str, object]:
    return {
        "worker_id": worker_id,
        "binary_digest": "a" * 64,
        "source_revision": "test",
        "compiler": "clang",
        "grpc_version": "1.77",
        "protobuf_version": "6030000",
        "host": host,
        "os": "test-os",
        "endpoint": f"{ip}:50543",
    }


def link_archive() -> dict[str, object]:
    ping = f"pong from fixture-mac ({MAC_TAILNET}) via {MAC_PUBLIC}:41641 in 5ms"
    path = {
        "observed_at": NOW,
        "kind": "tailscale",
        "source_ip": CUDA_TAILNET,
        "target_ip": MAC_TAILNET,
        "peer_id": NODE,
        "connection_type": "direct",
        "endpoint": f"{MAC_PUBLIC}:41641",
        "online": True,
        "evidence": json.dumps({"CurAddr": f"{MAC_PUBLIC}:41641", "ping": ping}),
    }
    result = {
        "source": identity("cuda-host", "fixture-pc", CUDA_TAILNET),
        "target": identity("mlx-host", MAC_DNS, MAC_TAILNET),
        "channel_ready_ms": 2.0,
        "prompt_tokens": 4,
        "hidden_size": 8,
        "output_tokens": 1,
        "warmup_cycles": 0,
        "measured_cycles": 1,
        "streams": [{"cycle": 0, "setup_ms": 1.0, "teardown_ms": 1.0}],
        "samples": [
            {
                "cycle": 0,
                "step": 0,
                "payload_bytes": 64,
                "message_bytes": 80,
                "feedback_bytes": 8,
                "sender_encode_ms": 0.5,
                "round_trip_ms": 3.25,
            }
        ],
    }
    unsigned = LinkContent.model_validate(
        {
            "measured_at": NOW,
            "finished_at": NOW,
            "concurrent_load": "none",
            "source_endpoint": f"{CUDA_TAILNET}:50543",
            "target_endpoint": f"{MAC_TAILNET}:50543",
            "path_sample_interval_seconds": 0.5,
            "paths": [path, path],
            "result": result,
            "completed": True,
            "qualified": True,
            "error": None,
        }
    ).model_dump(mode="json")
    return LinkArtifact.model_validate(unsigned | {"artifact_digest": digest(unsigned)}).model_dump(
        mode="json"
    )


def test_link_archive_digests_are_recomputed_and_summary_follows(tmp_path: Path) -> None:
    archive = link_archive()
    gz = tmp_path / "cuda-mlx-link.json.gz"
    gz.write_bytes(gzip.compress((json.dumps(archive, indent=2, sort_keys=True) + "\n").encode()))
    summary = {
        "links": [
            {
                "profile_digest": archive["artifact_digest"],
                "archive_sha256": hashlib.sha256(gz.read_bytes()).hexdigest(),
                "source": archive["result"]["source"],  # type: ignore[index]
                "note": "café",
            }
        ]
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    for path, data in redact_files([summary_path, gz]).items():
        path.write_bytes(data)

    text = gzip.decompress(gz.read_bytes()).decode()
    redacted = LinkArtifact.model_validate_json(text)  # Schema, seal and path qualification.
    assert redacted.artifact_digest != archive["artifact_digest"]
    assert text == json.dumps(redacted.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    assert redacted.result and redacted.result.source.host == "<cuda-host>"
    assert redacted.result.target.host == "<mac-host>"
    assert redacted.result.target.endpoint == "<mac-tailnet-ip>:50543"
    observed = redacted.paths[0]
    assert (observed.source_ip, observed.target_ip) == ("<cuda-tailnet-ip>", "<mac-tailnet-ip>")
    assert (observed.peer_id, observed.endpoint) == ("<mac-node-id>", "<mac-public-ip>:41641")
    assert "pong from <mac-host> (<mac-tailnet-ip>) via <mac-public-ip>:41641" in observed.evidence

    new_summary = json.loads(summary_path.read_text())
    link = new_summary["links"][0]
    assert link["profile_digest"] == redacted.artifact_digest
    assert link["archive_sha256"] == hashlib.sha256(gz.read_bytes()).hexdigest()
    assert link["source"] == redacted.result.source.model_dump(mode="json")
    assert summary_path.read_text() == json.dumps(new_summary, indent=2, ensure_ascii=False) + "\n"
    for raw in (CUDA_TAILNET, MAC_TAILNET, MAC_PUBLIC, MAC_DNS, NODE, "fixture-pc", "fixture-mac"):
        assert raw not in text and raw not in summary_path.read_text()
    # A second pass finds nothing left to redact.
    assert all(d == p.read_bytes() for p, d in redact_files([summary_path, gz]).items())


def test_reseal_refuses_an_artifact_whose_digest_does_not_verify(tmp_path: Path) -> None:
    archive = link_archive() | {"artifact_digest": "0" * 64}
    path = tmp_path / "link.json"
    path.write_text(json.dumps(archive, indent=2) + "\n")
    with pytest.raises(ValueError, match="does not verify"):
        redact_files([path])
    single = LinkArtifact.model_validate(redact(link_archive()))  # INPUT OUTPUT mode reseals.
    assert single.result and single.result.target.host == "<mac-host>"


def test_categories_share_placeholders_across_values() -> None:
    redactor = Redactor()
    peer = {
        "HostName": "fixture-pc",
        "TailscaleIPs": [CUDA_TAILNET, "fd7a:115c:a1e0::1:2"],
        "CurAddr": "[2606:4700:4700::1111]:41641",
    }
    redactor.learn_all([[link_archive()], {"peers": [peer]}])
    assert redactor.redact(peer) == {
        "HostName": "<cuda-host>",
        "TailscaleIPs": ["<cuda-tailnet-ip>", "<cuda-tailnet-ipv6>"],
        "CurAddr": "[<cuda-public-ipv6>]:41641",
    }
    text = redactor.text
    assert text(f"client {MAC_TAILNET}.65524 > {CUDA_TAILNET}.50546") == (
        "client <mac-tailnet-ip>.65524 > <cuda-tailnet-ip>.50546"
    )
    assert text(f"[::ffff:{MAC_TAILNET}]:60682") == "[::ffff:<mac-tailnet-ip>]:60682"
    assert text(f"ssh: connect to host {CUDA_TAILNET} port 22") == (
        "ssh: connect to host <cuda-tailnet-ip> port 22"
    )
    assert text("/home/alice/run/x.cpp:12 /Users/bob/hLLM") == "~/run/x.cpp:12 ~/hLLM"
    assert text("/home/conda/feedstock_root/gtest_main.cc") == (
        "/home/conda/feedstock_root/gtest_main.cc"
    )
    assert text("host 10-9-8-7.campus.example.net") == "host <reverse-dns-host>"
    assert text("peer nZyXwV9876CNTRL nodekey:" + "ab" * 32) == (
        "peer <tailscale-node-id> <tailscale-node-key>"
    )
    assert text("ether 02:42:ac:11:00:02") == "ether <mac-address>"
    assert text("8.8.4.4 and 2001:4860:4860::8844") == "<IPv4 address> and <IPv6 address>"
    unchanged = "127.0.0.1:50051 [::1]:1 0.0.0.0 std::vector 12:34:56 torch 2.13.0"
    assert text(unchanged) == unchanged
    assert redactor.redact({"hostname": "box", "host": 0, "name": "box"}) == {
        "hostname": "<host>",
        "host": 0,
        "name": "box",
    }


def test_json_formatting_is_reproduced_or_refused() -> None:
    for text in ('{"b": 1, "a": "\\u00e9"}', '{\n    "a": [\n        1\n    ]\n}\n'):
        assert dump(json.loads(text), json_format(text)) == text
    with pytest.raises(ValueError, match="formatting"):
        json_format('{"a":1,  "b": 2}')


def test_unchanged_files_keep_their_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "odd.json"
    path.write_bytes(b'{"a":1,  "b": "127.0.0.1"}')
    assert redact_files([path]) == {path: path.read_bytes()}


def test_committed_evidence_contains_no_host_identifiers() -> None:
    names = subprocess.run(
        ["git", "ls-files", "*.json", "*.json.gz"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    paths = [ROOT / name for name in names]
    changed = [p for p, data in redact_files(paths).items() if data != p.read_bytes()]
    assert changed == []


def test_committed_link_archives_match_the_timing_summary() -> None:
    root = ROOT / "docs/validation/milestone-5-timing"
    summary = json.loads((root / "summary.json").read_text())
    assert len(summary["links"]) == 4
    for link in summary["links"]:
        data = (root / Path(link["artifact"]).name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == link["archive_sha256"]
        artifact = LinkArtifact.model_validate_json(gzip.decompress(data))
        assert artifact.artifact_digest == link["profile_digest"]
        assert artifact.result is not None
        assert artifact.result.source.model_dump(mode="json") == link["source"]
