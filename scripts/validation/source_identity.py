"""Record a checkout's base revision, tracked source hashes, combined digest and binaries.

The source digest is SHA-256 over sorted `path\\0sha256\\n` lines of the git-tracked
files under SOURCE_PATHS, so it identifies the bytes that ran even with local edits.
Profiles take it as `--source-revision`; evidence records take it as
`--source-digest`. Prints the digest on the last line for launch scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_PATHS = (
    ".github",
    "CMakeLists.txt",
    "CMakePresets.json",
    "cpp",
    "proto",
    "pyproject.toml",
    "python",
    "scripts",
    "tests",
    "uv.lock",
)
DIGEST_METHOD = (
    "sha256 over sorted 'path\\0sha256\\n' lines of git-tracked files under the listed paths"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(root: Path, binaries: list[str]) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    files = sorted(f for f in git("ls-files", "--", *SOURCE_PATHS).splitlines() if f)
    hashes = {f: sha256(root / f) for f in files}
    digest = hashlib.sha256("".join(f"{f}\0{h}\n" for f, h in hashes.items()).encode()).hexdigest()
    return {
        "base_revision": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": [
            line for line in git("status", "--porcelain", "--", *SOURCE_PATHS).splitlines() if line
        ],
        "digest_method": DIGEST_METHOD,
        "source_files": hashes,
        "source_digest": digest,
        "binaries": {b: sha256(root / b) for b in binaries},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="git checkout")
    parser.add_argument("output", type=Path, help="identity JSON to write (not overwritten)")
    parser.add_argument("binaries", nargs="*", help="binaries to hash, relative to root")
    args = parser.parse_args(argv)
    record = source_identity(args.root.resolve(), args.binaries)
    with args.output.open("x") as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        record["base_revision"],
        f"dirty={len(record['dirty_files'])}",
        f"files={len(record['source_files'])}",
    )
    for binary, digest in record["binaries"].items():
        print(digest, binary)
    print(record["source_digest"])


if __name__ == "__main__":
    main()
