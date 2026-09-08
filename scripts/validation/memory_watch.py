"""Sample memory of one explicitly selected qualification worker until it exits."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def cuda_memory_bytes(pid: int) -> int | None:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        for line in query.stdout.splitlines():
            process_id, size = line.split(",")
            if int(process_id) == pid:
                return int(size.strip()) * 1024**2
    except (OSError, subprocess.SubprocessError, ValueError):
        pass  # Missing process/unsupported counter/transient failure is unavailable, not zero.
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pid", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seconds", type=float, default=1800)
    args = parser.parse_args()
    start = time.monotonic()
    with args.output.open("w", buffering=1) as output:
        while time.monotonic() - start < args.seconds:
            ps = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(args.pid)], capture_output=True, text=True
            )
            if ps.returncode != 0 or not ps.stdout.strip():
                break
            sample = {"unix_time": time.time(), "pid": args.pid, "rss_bytes": int(ps.stdout) * 1024}
            status = Path(f"/proc/{args.pid}/status")
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        sample["rss_high_water_bytes"] = int(line.split()[1]) * 1024
            except FileNotFoundError:
                pass  # Non-Linux host, or the sampled process exited after ps.
            if args.cuda:
                sample["cuda_process_bytes"] = cuda_memory_bytes(args.pid)
            output.write(json.dumps(sample) + "\n")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
