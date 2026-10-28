"""Remove host and network identifiers from qualification evidence before committing it.

Values are replaced with stable placeholders, so one value reads the same in every
file of an invocation. Host roles come from the evidence itself: native link probe
identities (`worker_id` `cuda-host` -> `<cuda-host>`, `<cuda-tailnet-ip>`), their
path observations (`<mac-public-ip>`, `<mac-node-id>`) and Tailscale peer records.
Unattributed addresses fall back to `<IPv4 address>`-style placeholders; loopback and
unspecified addresses are kept. Home directories become `~`.

`--in-place` rewrites every file given (JSON, `.json.gz` archives or plain text),
preserving each JSON file's original formatting. A sealed `artifact_digest` is
recomputed with the profiler's canonical digest, and any other digest of a rewritten
archive (its file SHA-256 in a summary, its artifact digest) is updated wherever it
appears in the same invocation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import ipaddress
import itertools
import json
import re
from pathlib import Path
from typing import Any

from hllm_control.profiling.models import digest

type Address = ipaddress.IPv4Address | ipaddress.IPv6Address

IPV4 = re.compile(
    r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?"
    r"(?!\w)(?=\.\d{1,5}(?![\w.])|\.?(?![\w.]))"  # Also `a.b.c.d.port` (tcpdump).
)
IPV6 = re.compile(r"(?<![\w:.])[0-9a-fA-F]{0,4}(?::[0-9a-fA-F]{0,4}){2,7}(?:/\d{1,3})?(?![\w:.])")
TAILNET = (ipaddress.ip_network("100.64.0.0/10"), ipaddress.ip_network("fd7a:115c:a1e0::/48"))
HOME = re.compile(r"/(?:home|Users)/(?!(?:conda|runner|linuxbrew)/)[^/\s\"'\\]+")
REVERSE_DNS = re.compile(r"(?<![\w.-])\d{1,3}(?:-\d{1,3}){3}\.(?:[a-z0-9-]+\.)*[a-z]{2,}\b", re.I)
NODE_ID = re.compile(r"\bn[A-Za-z0-9]{6,}CNTRL\b")
NODE_KEY = re.compile(r"\b(?:nodekey|mkey|discokey):[0-9a-f]{64}\b")
MAC_ADDRESS = re.compile(r"(?<![\w:])(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}(?![\w:])")
PING = re.compile(r"pong from (\S+) \(([^)\s]+)\)")
HOST_KEYS = {"host", "hostname", "HostName", "DNSName"}
ROLE_NAMES = {"mlx": "mac"}  # Probe worker IDs are backend names; placeholders name machines.


def address(value: str) -> Address | None:
    try:
        return ipaddress.ip_address(value.strip("[]"))
    except ValueError:
        return None


def endpoint_address(value: str) -> Address | None:
    """`ip:port`, `[ipv6]:port` or a bare address."""
    return address(value) or address(value.rsplit(":", 1)[0])


def scope(ip: Address) -> str:
    kind = "tailnet" if any(ip in n for n in TAILNET if n.version == ip.version) else None
    kind = kind or ("public" if ip.is_global else "lan")
    return f"{kind}-ip" if ip.version == 4 else f"{kind}-ipv6"


def kept(ip: Address) -> bool:
    return ip.is_loopback or ip.is_unspecified


class Redactor:
    """Learns host roles from evidence, then redacts values consistently."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.addresses: dict[Address, str] = {}
        self.roles: dict[Address, str] = {}
        self.digests: dict[str, str] = {}  # Old digest -> digest of the redacted bytes.

    def bind(self, role: str, ip: Address | None) -> None:
        if ip is not None and not kept(ip):
            self.roles.setdefault(ip, role)
            self.addresses.setdefault(ip, f"<{role}-{scope(ip)}>")

    def name(self, role: str, value: object, kind: str = "host") -> None:
        if isinstance(value, str) and value and not value.startswith("<"):
            self.names.setdefault(value, f"<{role}-{kind}>")

    def learn(self, value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                self.learn(item)
        if not isinstance(value, dict):
            return
        if {"worker_id", "host", "endpoint"} <= value.keys():  # Native probe identity.
            base = str(value["worker_id"]).removesuffix("-host")
            role = ROLE_NAMES.get(base, base)
            self.name(role, value["host"])
            self.bind(role, endpoint_address(str(value["endpoint"])))
        role = self.role_of(value.get("target_ip"))
        if role and "peer_id" in value:  # Path observation of that target.
            self.name(role, value["peer_id"], "node-id")
            if value.get("connection_type") == "direct":
                self.bind(role, endpoint_address(str(value.get("endpoint", ""))))
        if isinstance(value.get("TailscaleIPs"), list):  # Tailscale peer status.
            ips = [ip for ip in (address(str(v)) for v in value["TailscaleIPs"]) if ip]
            known = [self.roles[ip] for ip in ips if ip in self.roles]
            role = next(iter(known), None) or self.names.get(str(value.get("HostName")), "")
            role = role.strip("<>").removesuffix("-host")
            if role:
                for ip in ips:
                    self.bind(role, ip)
                self.name(role, value.get("HostName"))
                self.bind(role, endpoint_address(str(value.get("CurAddr", ""))))
        for item in value.values():
            for match in PING.finditer(item) if isinstance(item, str) else ():
                if role := self.role_of(match[2]):
                    self.name(role, match[1])
            self.learn(item)

    def role_of(self, value: object) -> str | None:
        ip = address(str(value))
        return self.roles.get(ip) if ip else None

    def learn_all(self, values: list[Any]) -> None:
        while True:  # Roles found in one file can attribute values in another.
            size = (len(self.names), len(self.addresses))
            for value in values:
                self.learn(value)
            if size == (len(self.names), len(self.addresses)):
                return

    def replace_address(self, match: re.Match[str]) -> str:
        value = match.group()
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError:
            return value
        if "/" in value:
            return f"<IPv{network.version} prefix/{network.prefixlen}>"
        ip = network.network_address
        return value if kept(ip) else self.addresses.get(ip, f"<IPv{ip.version} address>")

    def text(self, value: str) -> str:
        for old, new in self.digests.items():
            value = value.replace(old, new)
        value = re.sub(r"[\w.-]+\.ts\.net\.?", "<tailnet hostname>", value)
        for name in sorted(self.names, key=len, reverse=True):
            value = re.sub(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", self.names[name], value)
        value = IPV6.sub(self.replace_address, IPV4.sub(self.replace_address, value))
        value = REVERSE_DNS.sub("<reverse-dns-host>", value)
        value = NODE_KEY.sub("<tailscale-node-key>", NODE_ID.sub("<tailscale-node-id>", value))
        value = MAC_ADDRESS.sub("<mac-address>", value)
        value = HOME.sub("~", value)
        return re.sub(r"(?<=pong from )(?!<)\S+", "peer", value)

    def redact(self, value: Any, key: str | None = None) -> Any:
        if isinstance(value, str):
            redacted = self.text(value)
            generic = value.startswith("<") or value in ("", "localhost") or address(value)
            return "<host>" if key in HOST_KEYS and redacted == value and not generic else redacted
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {self.text(k): self.redact(item, k) for k, item in value.items()}
        return value


def redact(value: Any) -> Any:
    """Redact one report on its own, resealing it if it is a sealed artifact."""
    redactor = Redactor()
    redactor.learn_all([value])
    redacted = redactor.redact(value)
    if isinstance(value, dict) and "artifact_digest" in value and redacted != value:
        return reseal(value, redacted)
    return redacted


def json_format(text: str) -> dict[str, Any]:
    """Serializer settings that reproduce `text` byte for byte."""
    data = json.loads(text)
    for indent, ascii_, sort, end in itertools.product(
        (2, None, 4), (True, False), (False, True), ("\n", "")
    ):
        style = {"indent": indent, "ensure_ascii": ascii_, "sort_keys": sort, "end": end}
        if dump(data, style) == text:
            return style
    raise ValueError("cannot reproduce the original JSON formatting")


def dump(data: Any, style: dict[str, Any]) -> str:
    options = {k: v for k, v in style.items() if k != "end"}
    return json.dumps(data, **options) + style["end"]


def reseal(original: dict[str, Any], redacted: dict[str, Any]) -> dict[str, Any]:
    """Recompute a sealed artifact's digest the way the profiler computes it."""
    old = original["artifact_digest"]
    if digest({k: v for k, v in original.items() if k != "artifact_digest"}) != old:
        raise ValueError("original artifact digest does not verify; refusing to reseal")
    unsigned = {k: v for k, v in redacted.items() if k != "artifact_digest"}
    return {**redacted, "artifact_digest": digest(unsigned)}


def read(path: Path) -> str:
    data = path.read_bytes()
    return (gzip.decompress(data) if path.suffix == ".gz" else data).decode()


def is_json(path: Path) -> bool:
    return path.name.endswith((".json", ".json.gz"))


def redact_files(paths: list[Path]) -> dict[Path, bytes]:
    """Redacted bytes for each path, sharing one placeholder mapping.

    Archives are rewritten first so summaries citing their digests pick up new ones.
    """
    texts = {p: read(p) for p in paths}
    values = {p: json.loads(t) for p, t in texts.items() if is_json(p)}
    redactor = Redactor()
    redactor.learn_all(list(values.values()))
    output: dict[Path, bytes] = {}
    for path in sorted(paths, key=lambda p: p.suffix != ".gz"):
        if path not in values:
            output[path] = redactor.text(texts[path]).encode()
            continue
        original = values[path]
        redacted = redactor.redact(original)
        if redacted == original:  # Unchanged: keep the original bytes and digests.
            output[path] = path.read_bytes()
            continue
        if isinstance(original, dict) and "artifact_digest" in original:
            redacted = reseal(original, redacted)
            redactor.digests[original["artifact_digest"]] = redacted["artifact_digest"]
        text = dump(redacted, json_format(texts[path])).encode()
        output[path] = gzip.compress(text, 9, mtime=0) if path.suffix == ".gz" else text
        redactor.digests[hashlib.sha256(path.read_bytes()).hexdigest()] = hashlib.sha256(
            output[path]
        ).hexdigest()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="input output, or files")
    parser.add_argument("--in-place", action="store_true", help="rewrite every path given")
    args = parser.parse_args()
    if args.in_place:
        for path, data in redact_files(args.paths).items():
            if data != path.read_bytes():
                path.write_bytes(data)
                print(f"redacted {path}")
        return
    if len(args.paths) != 2:
        parser.error("expected input and output paths (or --in-place FILE...)")
    source, target = args.paths
    target.write_text(json.dumps(redact(json.loads(read(source))), indent=2) + "\n")


if __name__ == "__main__":
    main()
