"""Run the tiny two-worker CPU procedure and fault tests under one process-group guard."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from scripts.validation import resource_guard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("build/cpu-rehearsal"))
    args = parser.parse_args()
    for name in ("HLLM_CPU_WORKER", "HLLM_MEMORY_PROFILER"):
        path = Path(os.environ.get(name, ""))
        if not path.is_file():
            parser.error(f"set {name} to the built CPU executable")
        os.environ[name] = str(path.resolve())
    args.output_root.mkdir(parents=True, exist_ok=True)
    run = Path(
        tempfile.mkdtemp(prefix=datetime.now(UTC).strftime("%Y%m%d-%H%M%S-"), dir=args.output_root)
    ).resolve()
    print(f"CPU rehearsal evidence: {run}", flush=True)
    report: dict = dict(scope="cpu-rehearsal", accelerator_qualified=False)
    try:
        result = resource_guard.main(
            [
                "--seconds",
                "180",
                "--record",
                str(run / "guard.jsonl"),
                "--",
                sys.executable,
                "-m",
                "pytest",
                "tests/profiling/test_reload_cpu.py",
                "-q",
                "--basetemp",
                str(run / "tests"),
                "--junitxml",
                str(run / "junit.xml"),
            ]
        )
    except Exception as error:
        result = 125
        report["error"] = f"{type(error).__name__}: {error}"
        print(report["error"], file=sys.stderr)
    report["exit_code"] = result
    (run / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
