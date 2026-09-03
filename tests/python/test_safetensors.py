from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
from conftest import write_safetensors
from hllm_control.models import DType
from hllm_control.prepare.safetensors import SafetensorsError, inspect_safetensors


def test_inspects_tensor_metadata_without_payload_decoding(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    write_safetensors(path, {"weight": (DType.BF16, (3, 4))})

    inspected = inspect_safetensors(path)

    assert inspected.tensors[0].dtype == DType.BF16
    assert inspected.tensors[0].shape == (3, 4)
    assert inspected.tensors[0].byte_length == 24


def test_rejects_inconsistent_tensor_length(tmp_path: Path) -> None:
    path = tmp_path / "broken.safetensors"
    header = json.dumps(
        {"weight": {"dtype": "F16", "shape": [2, 2], "data_offsets": [0, 6]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(6))

    with pytest.raises(SafetensorsError, match="expected 8"):
        inspect_safetensors(path)


def test_rejects_overlapping_tensors(tmp_path: Path) -> None:
    path = tmp_path / "broken.safetensors"
    header = json.dumps(
        {
            "one": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
            "two": {"dtype": "F16", "shape": [2], "data_offsets": [2, 6]},
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(6))

    with pytest.raises(SafetensorsError, match="overlaps"):
        inspect_safetensors(path)
