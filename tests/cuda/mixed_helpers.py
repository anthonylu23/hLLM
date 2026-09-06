from __future__ import annotations

import os
from pathlib import Path

from tests.process_helpers import BINARY, ROOT, Workers

CUDA_BINARY = Path(
    os.environ.get("HLLM_CUDA_WORKER", ROOT / "build/native/cuda/cpp/hllm-worker-cuda")
)


def mixed_workers(
    root: Path,
    *,
    mode: str = "pageable",
    device_limit: int = 128 * 1024 * 1024,
    host_limit: int | tuple[int, int] = 64 * 1024 * 1024,
) -> Workers:
    extra = ("--device-memory-limit-bytes", str(device_limit))
    if mode != "pageable":
        extra += ("--boundary-transfer-mode", mode, "--pinned-host-memory-limit-bytes", "8388608")
    return Workers(root, limit=host_limit, binaries=(BINARY, CUDA_BINARY), extra_args=((), extra))
