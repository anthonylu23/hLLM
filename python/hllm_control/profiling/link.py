"""Native directional transport measurements with observed path qualification."""

from __future__ import annotations

import ipaddress
import json
import platform
import re
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Annotated, Any, Literal, Self

import grpc
from google.protobuf.json_format import MessageToDict
from pydantic import AwareDatetime, Field, model_validator

from hllm_control.profiling.models import Digest, Label, Milliseconds, ProfileModel, digest
from hllm_control.profiling.runner import write_exclusive
from hllm_control.proto import common_pb2, control_pb2, control_pb2_grpc
from hllm_control.serialization import sha256_file


class ProbeIdentity(ProfileModel):
    worker_id: Label
    binary_digest: Digest
    source_revision: Label
    compiler: Label
    grpc_version: Label
    protobuf_version: Label
    host: Label
    os: Label
    endpoint: Label


class LinkSample(ProfileModel):
    cycle: Annotated[int, Field(ge=0)]
    step: Annotated[int, Field(ge=0)]
    payload_bytes: Annotated[int, Field(gt=0)]
    message_bytes: Annotated[int, Field(gt=0)]
    feedback_bytes: Annotated[int, Field(gt=0)]
    sender_encode_ms: Milliseconds
    round_trip_ms: Milliseconds


class StreamSample(ProfileModel):
    cycle: Annotated[int, Field(ge=0)]
    setup_ms: Milliseconds
    teardown_ms: Milliseconds


class LinkResult(ProfileModel):
    source: ProbeIdentity
    target: ProbeIdentity
    channel_ready_ms: Milliseconds
    streams: tuple[StreamSample, ...]
    samples: tuple[LinkSample, ...]
    prompt_tokens: Annotated[int, Field(ge=1, le=4096)]
    hidden_size: Annotated[int, Field(ge=1, le=8192)]
    output_tokens: Annotated[int, Field(ge=1, le=1024)]
    warmup_cycles: Annotated[int, Field(ge=0, le=31)]
    measured_cycles: Annotated[int, Field(ge=1, le=32)]

    @model_validator(mode="after")
    def complete_samples(self) -> Self:
        cycles = self.warmup_cycles + self.measured_cycles
        if cycles > 32 or self.source.worker_id == self.target.worker_id:
            raise ValueError("invalid cycles or direction")
        if [s.cycle for s in self.streams] != list(range(cycles)):
            raise ValueError("missing/duplicate stream timings")
        expected = [(c, s) for c in range(cycles) for s in range(self.output_tokens)]
        if [(s.cycle, s.step) for s in self.samples] != expected:
            raise ValueError("missing/duplicate exchange timings")
        for s in self.samples:
            payload = (self.prompt_tokens if s.step == 0 else 1) * self.hidden_size * 2
            if s.payload_bytes != payload or s.message_bytes <= payload:
                raise ValueError("incorrect F16 payload or serialized message size")
        return self


class PathObservation(ProfileModel):
    observed_at: AwareDatetime
    kind: Literal["loopback", "tailscale"]
    source_ip: str
    target_ip: str
    peer_id: str
    connection_type: Literal["direct", "peer-relayed", "derp-relayed", "unknown"]
    endpoint: str
    online: bool
    evidence: str

    def route(self) -> tuple[object, ...]:
        return (
            self.kind,
            self.source_ip,
            self.target_ip,
            self.peer_id,
            self.connection_type,
            self.endpoint,
            self.online,
        )


def path_rejection(observations: tuple[PathObservation, ...]) -> str | None:
    if len(observations) < 2:
        return "missing before/after path observations"
    if any(not p.online or p.connection_type == "unknown" for p in observations):
        return "path unavailable or unknown during run"
    if len({p.route() for p in observations}) != 1:
        return "path changed or mixed during run"
    return None


class LinkContent(ProfileModel):
    schema_version: Literal["1.1"] = "1.1"
    profiler_version: Literal["0.2.0"] = "0.2.0"
    kind: Literal["native-link-run"] = "native-link-run"
    transport: Literal["grpc-insecure-stage-message-f16-v1"] = "grpc-insecure-stage-message-f16-v1"
    stream_policy: Literal["persistent-per-cycle"] = "persistent-per-cycle"
    measured_at: AwareDatetime
    finished_at: AwareDatetime
    concurrent_load: Label
    source_endpoint: Label
    target_endpoint: Label
    path_sample_interval_seconds: Annotated[float, Field(ge=0.1, le=5)]
    paths: tuple[PathObservation, ...]
    result: LinkResult | None
    completed: bool
    qualified: bool
    error: str | None

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.finished_at < self.measured_at:
            raise ValueError("invalid observation interval")
        if self.completed != (self.result is not None):
            raise ValueError("completion requires complete native samples")
        if self.qualified != (self.completed and path_rejection(self.paths) is None):
            raise ValueError("qualified flag disagrees with observed path")
        if not self.qualified and not self.error:
            raise ValueError("unqualified run requires reason")
        return self


class LinkArtifact(LinkContent):
    artifact_digest: Digest

    @model_validator(mode="after")
    def seal(self) -> Self:
        if self.artifact_digest != digest(
            self.model_dump(mode="json", exclude={"artifact_digest"})
        ):
            raise ValueError("link artifact digest mismatch")
        return self


def summarize_link(artifact: LinkArtifact) -> dict[str, object]:
    r = artifact.result
    buckets: list[dict[str, object]] = []
    if r:
        for phase in ("prefill", "decode"):
            samples = [
                s
                for s in r.samples
                if s.cycle >= r.warmup_cycles and (s.step == 0) == (phase == "prefill")
            ]
            if not samples:
                continue
            rtts = [s.round_trip_ms for s in samples]
            encodes = [s.sender_encode_ms for s in samples]
            buckets.append(
                {
                    "phase": phase,
                    "payload_bytes": samples[0].payload_bytes,
                    "count": len(samples),
                    "round_trip_median_ms": median(rtts),
                    "round_trip_min_ms": min(rtts),
                    "round_trip_max_ms": max(rtts),
                    "sender_encode_median_ms": median(encodes),
                    "exchange_median_ms": median(
                        [a + b for a, b in zip(rtts, encodes, strict=True)]
                    ),
                }
            )
    return {
        "profile_digest": artifact.artifact_digest,
        "qualified": artifact.qualified,
        "error": artifact.error,
        "buckets": buckets,
        "attribution": "Exchange = sender encode + native RPC round trip including feedback. "
        "GPU conversion and model compute are excluded. Never halve RTT or add feedback again.",
        "path_evidence": "Before/during/after source-host observations. Reject observed changes; "
        "sampling cannot rule out transitions shorter than the observation interval.",
    }


def check_link_compatibility(
    artifact: LinkArtifact,
    *,
    source: ProbeIdentity,
    target: ProbeIdentity,
    prompt_tokens: int,
    output_tokens: int,
    hidden_size: int,
    path: PathObservation,
    now: datetime,
    maximum_age_seconds: float,
) -> tuple[str, ...]:
    r = artifact.result
    reasons: list[str] = []
    if not artifact.qualified or r is None:
        return ("unqualified link profile",)
    if r.source != source or r.target != target:
        reasons.append("source/target build or direction mismatch")
    if (r.prompt_tokens, r.output_tokens, r.hidden_size) != (
        prompt_tokens,
        output_tokens,
        hidden_size,
    ):
        reasons.append("payload/workload mismatch")
    if path.route() != artifact.paths[-1].route():
        reasons.append("qualified path mismatch")
    age = (now - artifact.measured_at).total_seconds()
    if maximum_age_seconds < 0 or age < 0 or age > maximum_age_seconds:
        reasons.append("stale or future measurement")
    return tuple(reasons)


def _host(endpoint: str) -> str:
    # Explicit numeric endpoints avoid DNS pointing the observer at a different peer.
    host, _, port = endpoint.rpartition(":")
    if not port.isdigit() or not 1 <= int(port) <= 65535:
        raise ValueError("expected numeric IP:port endpoint")
    return str(ipaddress.ip_address(host.strip("[]")))


def observe_path(
    source_endpoint: str, target_endpoint: str, tailscale: Path | None, *, probe: bool = False
) -> PathObservation:
    source_ip, target_ip = _host(source_endpoint), _host(target_endpoint)
    observed_at = datetime.now(UTC)
    if ipaddress.ip_address(target_ip).is_loopback:
        if not ipaddress.ip_address(source_ip).is_loopback:
            raise ValueError("loopback target requires local source")
        return PathObservation(
            observed_at=observed_at,
            source_ip=source_ip,
            target_ip=target_ip,
            kind="loopback",
            peer_id="loopback",
            connection_type="direct",
            endpoint=target_ip,
            online=True,
            evidence="numeric loopback endpoints",
        )
    if tailscale is None:
        raise ValueError("Tailscale executable required for non-loopback qualification")
    tailscale = tailscale.resolve(strict=True)
    try:
        ping = ""
        if probe:
            ping = subprocess.check_output(
                [
                    str(tailscale),
                    "ping",
                    "--c",
                    "1",
                    "--until-direct=false",
                    "--timeout",
                    "3s",
                    target_ip,
                ],
                text=True,
                timeout=5,
                close_fds=False,  # Use posix_spawn instead of forking live gRPC threads.
            ).strip()
        status = json.loads(
            subprocess.check_output(
                [str(tailscale), "status", "--json"], text=True, timeout=5, close_fds=False
            )
        )
        local_ips = status["Self"]["TailscaleIPs"]
        if source_ip not in local_ips and not ipaddress.ip_address(source_ip).is_loopback:
            raise ValueError("path observer must run on source host")
        peer = next(p for p in status.get("Peer", {}).values() if target_ip in p["TailscaleIPs"])
        direct, relay, peer_relay = (
            peer.get("CurAddr", ""),
            peer.get("Relay", ""),
            peer.get("PeerRelay", ""),
        )
        route = (
            "direct"
            if direct
            else "peer-relayed"
            if peer_relay
            else "derp-relayed"
            if relay
            else "unknown"
        )
        # Relay is the home DERP region even on inactive peers: require an active connection.
        active = bool(peer.get("Online") and peer.get("Active"))
        if probe:
            match = re.search(r" via (.+) in ", ping)
            observed = match.group(1) if match else ""
            expected = direct or f"DERP({relay})"
            if not observed or observed != expected or peer_relay:
                active = False  # Unknown ping encodings never qualify a route.
        return PathObservation(
            observed_at=observed_at,
            source_ip=source_ip,
            target_ip=target_ip,
            kind="tailscale",
            peer_id=peer["ID"],
            connection_type=route if active else "unknown",
            endpoint=direct or peer_relay or relay,
            online=active,
            evidence=json.dumps(
                {
                    **{
                        k: peer.get(k)
                        for k in ("CurAddr", "Relay", "PeerRelay", "Online", "Active")
                    },
                    "ping": ping,
                },
                sort_keys=True,
            ),
        )
    except (
        OSError,
        subprocess.SubprocessError,
        KeyError,
        StopIteration,
        json.JSONDecodeError,
    ) as e:
        return PathObservation(
            observed_at=observed_at,
            source_ip=source_ip,
            target_ip=target_ip,
            kind="tailscale",
            peer_id="",
            connection_type="unknown",
            endpoint="",
            online=False,
            evidence=str(e),
        )


def server_config(
    *,
    binary: Path,
    worker_id: str,
    listen: str,
    peers: dict[str, str],
    source_revision: str,
    output: Path,
) -> None:
    _host(listen)
    for endpoint in peers.values():
        _host(endpoint)
    if worker_id in peers or not worker_id or not source_revision:
        raise ValueError("invalid server identity/peer allowlist")
    write_exclusive(
        output,
        {
            "worker_id": worker_id,
            "listen": listen,
            "peers": peers,
            "source_revision": source_revision,
            "binary_digest": sha256_file(binary),
        },
    )


def run_link_profile(
    *,
    source_endpoint: str,
    target_endpoint: str,
    target_worker_id: str,
    prompt_tokens: int,
    hidden_size: int,
    output_tokens: int,
    output: Path,
    concurrent_load: str,
    tailscale: Path | None = None,
    warmup_cycles: int = 2,
    measured_cycles: int = 5,
    timeout_seconds: float = 120,
    path_sample_interval_seconds: float = 0.5,
) -> LinkArtifact:
    if not (0.01 <= timeout_seconds <= 120 and 0.1 <= path_sample_interval_seconds <= 5):
        raise ValueError("invalid timing limits")
    # Reject runaway requests before opening a channel (native server enforces the same caps).
    if not (
        0 <= warmup_cycles
        and measured_cycles > 0
        and warmup_cycles + measured_cycles <= 32
        and 1 <= prompt_tokens <= 4096
        and 1 <= output_tokens <= 1024
        and 1 <= hidden_size <= 8192
        and prompt_tokens * hidden_size * 2 <= 8 * 1024**2
        and (prompt_tokens + output_tokens - 1)
        * hidden_size
        * 2
        * (warmup_cycles + measured_cycles)
        <= 256 * 1024**2
    ):
        raise ValueError("invalid payload/traffic limits")
    raw = output.with_suffix(".native.json")
    summary = output.with_suffix(".summary.json")
    for p in (output, raw, summary):
        if p.exists():
            raise FileExistsError(p)
    paths = [observe_path(source_endpoint, target_endpoint, tailscale, probe=True)]
    started = datetime.now(UTC)
    result: LinkResult | None = None
    error: str | None = None
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(path_sample_interval_seconds):
            paths.append(observe_path(source_endpoint, target_endpoint, tailscale))

    with grpc.insecure_channel(
        source_endpoint, options=[("grpc.max_receive_message_length", 16 * 1024**2)]
    ) as channel:
        stub: Any = control_pb2_grpc.WorkerControlStub(channel)
        info = stub.GetLinkProbeInfo(common_pb2.Empty(), timeout=5)
        if info.host != platform.node() or info.endpoint != source_endpoint:
            raise ValueError("run the path observer on the native source host")
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            response = stub.QualifyLink(
                control_pb2.LinkQualificationRequest(
                    target_worker_id=target_worker_id,
                    prompt_tokens=prompt_tokens,
                    hidden_size=hidden_size,
                    output_tokens=output_tokens,
                    warmup_cycles=warmup_cycles,
                    measured_cycles=measured_cycles,
                    timeout_ms=int(timeout_seconds * 1000),
                ),
                timeout=timeout_seconds + 1,
            )
            raw_result = MessageToDict(
                response,
                preserving_proto_field_name=True,
                always_print_fields_with_no_presence=True,
            )
            write_exclusive(raw, raw_result)
            if not response.HasField("qualification"):
                raise ValueError("server did not return native qualification samples")
            result = LinkResult.model_validate(raw_result["qualification"])
            if (
                result.source.worker_id != info.worker_id
                or result.target.worker_id != target_worker_id
                or result.source.endpoint != source_endpoint
                or result.target.endpoint != target_endpoint
            ):
                raise ValueError("native direction mismatch")
        except (grpc.RpcError, ValueError) as e:
            result = None
            error = str(e)
            if not raw.exists():
                write_exclusive(
                    raw,
                    {
                        "error": error,
                        "source": MessageToDict(info),
                        "target_worker_id": target_worker_id,
                        "prompt_tokens": prompt_tokens,
                        "hidden_size": hidden_size,
                        "output_tokens": output_tokens,
                    },
                )
        finally:
            stop.set()
            thread.join(timeout=6)
    paths.append(observe_path(source_endpoint, target_endpoint, tailscale, probe=True))
    rejected = path_rejection(tuple(paths))
    data: dict[str, object] = dict(
        schema_version="1.1",
        profiler_version="0.2.0",
        kind="native-link-run",
        measured_at=started.isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        concurrent_load=concurrent_load,
        source_endpoint=source_endpoint,
        target_endpoint=target_endpoint,
        path_sample_interval_seconds=path_sample_interval_seconds,
        paths=[p.model_dump(mode="json") for p in paths],
        result=result.model_dump(mode="json") if result else None,
        completed=result is not None,
        qualified=result is not None and rejected is None,
        error=error or rejected,
    )
    unsigned = LinkContent.model_validate(data).model_dump(mode="json")
    artifact = LinkArtifact.model_validate({**unsigned, "artifact_digest": digest(unsigned)})
    write_exclusive(output, artifact.model_dump(mode="json"))
    write_exclusive(summary, summarize_link(artifact))
    return artifact
