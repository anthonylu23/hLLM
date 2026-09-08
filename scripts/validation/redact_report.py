"""Remove network identifiers from qualification JSON before committing a report."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
from pathlib import Path
from typing import Any

IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?(?![\w.])")
IPV6 = re.compile(r"(?:[0-9a-fA-F]{0,4}:){2,}[0-9a-fA-F:.]*(?:/\d{1,3})?")


def replace_address(match: re.Match[str]) -> str:
    value = match.group()
    try:
        address = ipaddress.ip_network(value, strict=False)
    except ValueError:
        return value
    suffix = f" prefix/{address.prefixlen}" if "/" in value else " address"
    return f"<IPv{address.version}{suffix}>"


def redact(value: Any) -> Any:
    if isinstance(value, str):
        value = re.sub(r"(?<=pong from )\S+", "peer", value)
        value = re.sub(r"[\w.-]+\.ts\.net\.?", "<tailnet hostname>", value)
        return IPV6.sub(replace_address, IPV4.sub(replace_address, value))
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(redact(json.loads(args.input.read_text())), indent=2) + "\n")


if __name__ == "__main__":
    main()
