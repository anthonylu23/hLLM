from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc


def test_protobuf_contracts_compile(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    proto = root / "proto"
    output = tmp_path / "generated"
    output.mkdir()
    files = sorted(str(path) for path in proto.glob("*.proto"))

    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"-I{proto}",
            f"--python_out={output}",
            f"--grpc_python_out={output}",
            *files,
        ]
    )

    assert result == 0
