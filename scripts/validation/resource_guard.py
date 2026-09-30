"""Run one owned process group with a time limit and system-memory stop conditions.

Standard library only: the same file can be copied to a remote host. Open the
evidence file before spawning; every exit path retires the owned process group.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import re
import signal
import subprocess
import time
from pathlib import Path


def system_memory() -> dict[str, int]:
    if platform.system() == "Darwin":
        vm = subprocess.check_output(["vm_stat"], text=True, timeout=5)
        match = re.search(r"page size of (\d+)", vm)
        if match is None:
            raise ValueError("vm_stat page size unavailable")
        pages = {k: int(v) for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.", vm, re.M)}
        page_size = int(match[1])
        return dict(
            available_bytes=(pages["Pages free"] + pages["Pages inactive"]) * page_size,
            swapout_bytes=pages["Swapouts"] * page_size,
            pressure=int(
                subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True, timeout=5
                )
            ),
        )
    if platform.system() != "Linux":
        raise ValueError("only Linux and macOS have supported memory guards")
    mem = {
        k: int(v) * 1024
        for k, v in re.findall(r"^(\w+):\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)
    }
    vm = {k: int(v) for k, v in re.findall(r"^(\w+) (\d+)", Path("/proc/vmstat").read_text(), re.M)}
    return dict(
        available_bytes=mem["MemAvailable"],
        swapout_bytes=vm["pswpout"] * os.sysconf("SC_PAGE_SIZE"),
    )


class MemoryGuard:
    def __init__(self, minimum_available: int, maximum_swapout: int, pressure_seconds: float):
        self.minimum_available = minimum_available
        self.maximum_swapout = maximum_swapout
        self.pressure_seconds = pressure_seconds
        self.recent: collections.deque[tuple[float, int]] = collections.deque()
        self.pressure_since: float | None = None

    def check(self, now: float, row: dict[str, int]) -> str | None:
        if row["available_bytes"] < self.minimum_available:
            return "available memory below configured floor"
        if row.get("pressure", 1) != 1:
            if self.pressure_since is None:
                self.pressure_since = now
            if now - self.pressure_since >= self.pressure_seconds:
                return "sustained non-normal Mac memory pressure"
        else:
            self.pressure_since = None
        self.recent.append((now, row["swapout_bytes"]))
        # Retain the observation bracketing the start of the 60-second window.
        while len(self.recent) > 1 and self.recent[1][0] <= now - 60:
            self.recent.popleft()
        if row["swapout_bytes"] - self.recent[0][1] > self.maximum_swapout:
            return "new swap-out exceeds configured 60-second limit"
        return None


def stop_group(child: subprocess.Popen) -> None:
    # Also retire descendants if the group leader has already exited.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1200)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--minimum-available-bytes", type=int, default=1024**3)
    parser.add_argument("--maximum-swapout-bytes", type=int, default=256 * 1024**2)
    parser.add_argument("--footprint-helper", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.seconds <= 7200:
        parser.error("provide a command and a time limit in (0, 7200] seconds")
    if args.minimum_available_bytes < 0 or args.maximum_swapout_bytes < 0:
        parser.error("memory thresholds must be nonnegative")
    if args.footprint_helper and not args.footprint_helper.is_file():
        parser.error("footprint helper does not exist")
    args.record.parent.mkdir(parents=True, exist_ok=True)
    guard = MemoryGuard(args.minimum_available_bytes, args.maximum_swapout_bytes, 16)
    reason: str | None = None

    def interrupted(signum, _frame):
        nonlocal reason
        reason = f"guard received signal {signum}"

    old_handlers = {
        s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    child = None
    start = time.monotonic()
    try:
        with args.record.open("x", buffering=1) as output:
            baseline = system_memory()
            reason = reason or guard.check(start, baseline)
            if baseline.get("pressure", 1) != 1:
                reason = reason or "Mac pressure is already non-normal"
            output.write(
                json.dumps(
                    {"event": "preflight", "unix_time": time.time(), "reason": reason, **baseline}
                )
                + "\n"
            )
            if reason:
                raise RuntimeError(reason)
            try:
                child = subprocess.Popen(command, start_new_session=True)
                output.write(
                    json.dumps(
                        {
                            "event": "start",
                            "unix_time": time.time(),
                            "pid": child.pid,
                            "seconds": args.seconds,
                            "command": command,
                        }
                    )
                    + "\n"
                )
                while child.poll() is None:
                    now = time.monotonic()
                    row = system_memory()
                    reason = reason or guard.check(now, row)
                    if now - start >= args.seconds:
                        reason = reason or "experiment time limit"
                    observation = {"unix_time": time.time(), "pid": child.pid, **row}
                    if args.footprint_helper:
                        measured = subprocess.run(
                            [str(args.footprint_helper), str(child.pid)],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if measured.returncode == 0:
                            observation["owned_process_memory"] = json.loads(measured.stdout)
                        elif child.poll() is None:
                            raise RuntimeError("owned process footprint sampling failed")
                    output.write(json.dumps(observation) + "\n")
                    if reason:
                        break
                    time.sleep(min(2, max(0, args.seconds - (time.monotonic() - start))))
            except BaseException as error:
                reason = reason or f"guard error: {type(error).__name__}: {error}"
                raise
            finally:
                if child is not None:
                    stop_group(child)
                output.write(
                    json.dumps(
                        {
                            "event": "exit",
                            "unix_time": time.time(),
                            "reason": reason,
                            "returncode": child.returncode if child else None,
                        }
                    )
                    + "\n"
                )
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return 124 if reason else child.returncode if child else 125


if __name__ == "__main__":
    raise SystemExit(main())
