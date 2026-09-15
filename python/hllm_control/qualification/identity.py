"""Bind controller and remote helper behavior, including generated wire mappings."""

from pathlib import Path

from hllm_control.profiling.models import digest
from hllm_control.serialization import sha256_file


def package_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    return digest({str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*.py"))})
