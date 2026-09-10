from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest
from hllm_control.profiling.link import (
    LinkArtifact,
    LinkContent,
    LinkResult,
    PathObservation,
    ProbeIdentity,
    check_link_compatibility,
    path_rejection,
    summarize_link,
)
from hllm_control.profiling.models import digest

NOW = datetime(2026, 9, 8, tzinfo=UTC)


def observation(**updates) -> PathObservation:
    return PathObservation.model_validate(
        dict(
            observed_at=NOW,
            kind="tailscale",
            source_ip="100.64.0.1",
            target_ip="100.64.0.2",
            peer_id="peer-2",
            connection_type="direct",
            endpoint="192.0.2.2:41641",
            online=True,
            evidence="fixture",
        )
        | updates
    )


def identity(name: str) -> ProbeIdentity:
    return ProbeIdentity(
        worker_id=name,
        binary_digest="a" * 64,
        source_revision="test",
        compiler="clang",
        grpc_version="1.77",
        protobuf_version="6030000",
        host=name,
        os="test-os",
        endpoint="100.64.0.1:50051" if name == "a" else "100.64.0.2:50051",
    )


def artifact() -> LinkArtifact:
    result = LinkResult.model_validate(
        dict(
            source=identity("a"),
            target=identity("b"),
            channel_ready_ms=2.0,
            prompt_tokens=512,
            hidden_size=16,
            output_tokens=2,
            warmup_cycles=1,
            measured_cycles=2,
            streams=[dict(cycle=c, setup_ms=2.0, teardown_ms=2.0) for c in range(3)],
            samples=[
                dict(
                    cycle=c,
                    step=s,
                    payload_bytes=16384 if s == 0 else 32,
                    message_bytes=16484 if s == 0 else 132,
                    feedback_bytes=80,
                    sender_encode_ms=1.0,
                    round_trip_ms=1000.0 if c == 0 else 10.0,
                )
                for c in range(3)
                for s in range(2)
            ],
        )
    )
    unsigned = LinkContent(
        measured_at=NOW,
        finished_at=NOW,
        concurrent_load="test",
        source_endpoint=identity("a").endpoint,
        target_endpoint=identity("b").endpoint,
        path_sample_interval_seconds=0.5,
        paths=(observation(), observation()),
        result=result,
        completed=True,
        qualified=True,
        error=None,
    ).model_dump(mode="json")
    return LinkArtifact.model_validate(unsigned | {"artifact_digest": digest(unsigned)})


def test_path_changes_missing_and_unknown_rejected():
    a = observation()
    assert path_rejection((a, a)) is None
    for paths in (
        (a,),
        (a, observation(connection_type="derp-relayed", endpoint="dfw")),
        (a, observation(endpoint="192.0.2.3:41641")),
        (a, observation(connection_type="unknown")),
        (a, observation(online=False)),
        (a, observation(connection_type="peer-relayed"), a),
    ):
        assert path_rejection(paths)


def test_link_scope_and_summary_exclude_warmup_and_do_not_halve_rtt():
    a = artifact()
    summary = summarize_link(a)
    assert cast(list[dict[str, object]], summary["buckets"])[0]["exchange_median_ms"] == 11
    args: dict[str, Any] = dict(
        source=identity("a"),
        target=identity("b"),
        prompt_tokens=512,
        output_tokens=2,
        hidden_size=16,
        path=observation(),
        now=NOW,
        maximum_age_seconds=60,
    )
    assert not check_link_compatibility(a, **args)
    for change in (
        dict(source=identity("b"), target=identity("a")),
        dict(hidden_size=32),
        dict(output_tokens=256),
        dict(path=observation(connection_type="derp-relayed")),
        dict(now=NOW + timedelta(seconds=61)),
        dict(source=identity("a").model_copy(update={"binary_digest": "b" * 64})),
    ):
        assert check_link_compatibility(a, **cast(Any, args | change))
    tampered = a.model_dump(mode="json")
    tampered["qualified"] = False
    with pytest.raises(ValueError):
        LinkArtifact.model_validate(tampered)


def test_missing_or_wrong_payload_samples_rejected():
    a = artifact()
    assert a.result
    data = a.result.model_dump(mode="json")
    data["samples"].pop()
    with pytest.raises(ValueError, match="missing/duplicate"):
        LinkResult.model_validate(data)
    data = a.result.model_dump(mode="json")
    data["samples"][0]["payload_bytes"] = 32
    with pytest.raises(ValueError, match="incorrect F16"):
        LinkResult.model_validate(data)


def test_tailscale_qualification_requires_ping_and_status_agreement(tmp_path, monkeypatch):
    import json

    from hllm_control.profiling.link import observe_path

    binary = tmp_path / "tailscale"
    binary.touch()
    peer = dict(
        TailscaleIPs=["100.64.0.2"],
        ID="peer-2",
        CurAddr="192.0.2.2:41641",
        Relay="dfw",
        PeerRelay="",
        Online=True,
        Active=True,
    )
    ping = "pong from peer (100.64.0.2) via 192.0.2.2:41641 in 20ms"

    def output(command, **kwargs):
        # Starting Tailscale while gRPC runs must use the posix_spawn-compatible path.
        assert kwargs["close_fds"] is False
        if command[1] == "ping":
            return ping
        return json.dumps({"Self": {"TailscaleIPs": ["100.64.0.1"]}, "Peer": {"key": peer}})

    monkeypatch.setattr("hllm_control.profiling.link.subprocess.check_output", output)
    args = ("100.64.0.1:50543", "100.64.0.2:50543", binary)
    assert observe_path(*args, probe=True).connection_type == "direct"
    ping = "pong from peer (100.64.0.2) via DERP(dfw) in 40ms"
    assert observe_path(*args, probe=True).connection_type == "unknown"
    peer["CurAddr"] = ""
    assert observe_path(*args, probe=True).connection_type == "derp-relayed"
    peer["Active"] = False
    assert observe_path(*args, probe=True).connection_type == "unknown"
