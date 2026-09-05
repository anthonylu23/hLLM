"""Generate importable Python Protobuf bindings from the canonical schemas."""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from grpc_tools import protoc

MODULE_IMPORT = re.compile(r"^(import [a-z_]+_pb2 as )", re.MULTILINE)


def _generate(root: Path, output: Path) -> None:
    proto = root / "proto"
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(str(path) for path in proto.glob("*.proto"))
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto}",
            f"--python_out={output}",
            f"--pyi_out={output}",
            f"--grpc_python_out={output}",
            *files,
        ]
    )
    if result != 0:
        raise SystemExit(result)

    for path in output.iterdir():
        if not path.is_file() or "_pb2" not in path.name or path.suffix not in {".py", ".pyi"}:
            continue
        source = path.read_text(encoding="utf-8")
        source = MODULE_IMPORT.sub(r"from . \1", source)
        path.write_text(source, encoding="utf-8")
    (output / "__init__.py").write_text(
        '"""Generated hLLM v1 Protobuf and gRPC bindings."""\n', encoding="utf-8"
    )


def _generated_files(path: Path) -> dict[str, bytes]:
    return {
        item.name: item.read_bytes()
        for item in path.iterdir()
        if item.is_file() and (item.name == "__init__.py" or "_pb2" in item.name)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if checked-in bindings differ")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "python" / "hllm_control" / "proto"

    with tempfile.TemporaryDirectory(prefix="hllm-proto-") as temporary:
        generated = Path(temporary)
        _generate(root, generated)
        if arguments.check:
            if not destination.is_dir() or _generated_files(destination) != _generated_files(
                generated
            ):
                raise SystemExit(
                    "generated Python bindings are stale; run `uv run python "
                    "scripts/generate_proto.py`"
                )
            return

        destination.mkdir(parents=True, exist_ok=True)
        for existing in destination.iterdir():
            if existing.is_file() and (existing.name == "__init__.py" or "_pb2" in existing.name):
                existing.unlink()
        for source in generated.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)


if __name__ == "__main__":
    main()
