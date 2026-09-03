"""Strict, metadata-only Safetensors inspection."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hllm_control.models import DTYPE_BYTES, DType

MAX_HEADER_BYTES = 128 * 1024 * 1024


class SafetensorsError(ValueError):
    """Raised when a Safetensors container is malformed or unsupported."""


@dataclass(frozen=True)
class InspectedTensor:
    name: str
    dtype: DType
    shape: tuple[int, ...]
    data_offset: int
    byte_length: int


@dataclass(frozen=True)
class InspectedFile:
    path: Path
    size_bytes: int
    header_size_bytes: int
    tensors: tuple[InspectedTensor, ...]


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _parse_entry(name: str, raw: object) -> InspectedTensor:
    if not isinstance(raw, dict):
        raise SafetensorsError(f"tensor {name!r} metadata must be an object")
    entry = cast(dict[str, object], raw)
    try:
        raw_dtype = entry["dtype"]
        raw_shape = entry["shape"]
        raw_offsets = entry["data_offsets"]
    except (KeyError, TypeError, ValueError) as error:
        raise SafetensorsError(f"tensor {name!r} has invalid metadata") from error
    if not isinstance(raw_dtype, str):
        raise SafetensorsError(f"tensor {name!r} has an invalid dtype")
    try:
        dtype = DType(raw_dtype)
    except ValueError as error:
        raise SafetensorsError(f"tensor {name!r} has unsupported dtype {raw_dtype!r}") from error
    if not isinstance(raw_shape, list):
        raise SafetensorsError(f"tensor {name!r} has an invalid shape")
    shape_values = cast(list[object], raw_shape)
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in shape_values
    ):
        raise SafetensorsError(f"tensor {name!r} has an invalid shape")
    if not isinstance(raw_offsets, list):
        raise SafetensorsError(f"tensor {name!r} has invalid data offsets")
    offset_values = cast(list[object], raw_offsets)
    if len(offset_values) != 2:
        raise SafetensorsError(f"tensor {name!r} has invalid data offsets")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in offset_values):
        raise SafetensorsError(f"tensor {name!r} has invalid data offsets")
    start = cast(int, offset_values[0])
    end = cast(int, offset_values[1])
    if start < 0 or end < start:
        raise SafetensorsError(f"tensor {name!r} has an invalid data range")
    shape = tuple(cast(int, item) for item in shape_values)
    expected_bytes = _product(shape) * DTYPE_BYTES[dtype]
    if end - start != expected_bytes:
        raise SafetensorsError(
            f"tensor {name!r} byte length is {end - start}, expected {expected_bytes}"
        )
    return InspectedTensor(
        name=name,
        dtype=dtype,
        shape=shape,
        data_offset=start,
        byte_length=end - start,
    )


def inspect_safetensors(path: Path) -> InspectedFile:
    size_bytes = path.stat().st_size
    if size_bytes < 8:
        raise SafetensorsError(f"{path} is too short to be a Safetensors file")
    with path.open("rb") as stream:
        header_length_raw = stream.read(8)
        (header_length,) = struct.unpack("<Q", header_length_raw)
        if header_length == 0 or header_length > MAX_HEADER_BYTES:
            raise SafetensorsError(f"{path} has invalid header length {header_length}")
        if 8 + header_length > size_bytes:
            raise SafetensorsError(f"{path} header extends beyond the file")
        header_bytes = stream.read(header_length)
    try:
        parsed_header = cast(object, json.loads(header_bytes))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafetensorsError(f"{path} contains an invalid JSON header") from error
    if not isinstance(parsed_header, dict):
        raise SafetensorsError(f"{path} header must be a JSON object")
    header = cast(dict[str, object], parsed_header)

    tensors = tuple(
        sorted(
            (_parse_entry(name, raw) for name, raw in header.items() if name != "__metadata__"),
            key=lambda item: item.name,
        )
    )
    if not tensors:
        raise SafetensorsError(f"{path} does not contain tensors")

    data_size = size_bytes - 8 - header_length
    ranges = sorted(
        ((item.data_offset, item.data_offset + item.byte_length, item.name) for item in tensors),
        key=lambda item: item[0],
    )
    previous_end = 0
    for start, end, name in ranges:
        if start < previous_end:
            raise SafetensorsError(f"tensor {name!r} overlaps an earlier tensor")
        if end > data_size:
            raise SafetensorsError(f"tensor {name!r} extends beyond the data section")
        previous_end = end

    return InspectedFile(
        path=path,
        size_bytes=size_bytes,
        header_size_bytes=8 + header_length,
        tensors=tensors,
    )
