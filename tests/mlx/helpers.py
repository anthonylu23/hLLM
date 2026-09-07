from __future__ import annotations

import os
from pathlib import Path

from tests.process_helpers import BINARY, ROOT, Workers

MLX_BINARY = Path(os.environ.get("HLLM_MLX_WORKER", ROOT / "build/native/mlx/cpp/hllm-worker-mlx"))


def mixed_workers(
    root: Path, limit: int | tuple[int, int] = 128 * 1024 * 1024, *, mlx_first: bool = False
) -> Workers:
    return Workers(
        root, limit=limit, binaries=(MLX_BINARY, BINARY) if mlx_first else (BINARY, MLX_BINARY)
    )
