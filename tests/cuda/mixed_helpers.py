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
    cuda_first: bool = False,
    device_limit: int = 128 * 1024 * 1024,
    host_limit: int | tuple[int, int] = 64 * 1024 * 1024,
) -> Workers:
    extra = ("--device-memory-limit-bytes", str(device_limit))
    if mode != "pageable":
        host = host_limit if isinstance(host_limit, int) else host_limit[0 if cuda_first else 1]
        extra += (
            "--boundary-transfer-mode",
            mode,
            "--pinned-host-memory-limit-bytes",
            str(min(host, 8 * 1024 * 1024)),
        )
    return Workers(
        root,
        limit=host_limit,
        binaries=(CUDA_BINARY, BINARY) if cuda_first else (BINARY, CUDA_BINARY),
        extra_args=(extra, ()) if cuda_first else ((), extra),
    )
