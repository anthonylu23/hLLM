"""Build deterministic model manifests without loading tensor payloads."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from hllm_control.models import (
    MANIFEST_SCHEMA_VERSION,
    ComponentMemory,
    ModelManifest,
    SourceDescriptor,
    TensorFile,
    TensorRecord,
    TensorRole,
)
from hllm_control.prepare.llama import ArchitectureError, LlamaArchitectureAdapter
from hllm_control.prepare.qwen3 import Qwen3ArchitectureAdapter
from hllm_control.prepare.safetensors import InspectedFile, InspectedTensor, inspect_safetensors
from hllm_control.serialization import canonical_json_bytes, sha256_file


class PreparationError(ValueError):
    """Raised when a source model cannot produce a safe, complete manifest."""


class HashMode(StrEnum):
    NONE = "none"
    METADATA = "metadata"
    FULL = "full"


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreparationError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise PreparationError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def _discover_tensor_files(model_path: Path) -> tuple[list[Path], dict[str, str], Path | None]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _load_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise PreparationError("Safetensors index must contain a non-empty weight_map")
        untyped_weight_map = cast(dict[object, object], weight_map)
        if any(
            not isinstance(name, str) or not isinstance(file, str)
            for name, file in untyped_weight_map.items()
        ):
            raise PreparationError("Safetensors weight_map keys and values must be strings")
        typed_weight_map = cast(dict[str, str], untyped_weight_map)
        relative_files = sorted(set(typed_weight_map.values()))
        paths = [model_path / item for item in relative_files]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise PreparationError(f"Safetensors index references missing files: {missing}")
        return paths, typed_weight_map, index_path

    paths = sorted(model_path.glob("*.safetensors"))
    if len(paths) != 1:
        raise PreparationError(
            "model directory must contain one Safetensors file or model.safetensors.index.json"
        )
    inspected = inspect_safetensors(paths[0])
    return paths, {tensor.name: paths[0].name for tensor in inspected.tensors}, None


def _metadata_digest(inspected: tuple[InspectedFile, ...]) -> str:
    payload = [
        {
            "file": item.path.name,
            "size": item.size_bytes,
            "header_size": item.header_size_bytes,
            "tensors": [
                {
                    "name": tensor.name,
                    "dtype": tensor.dtype.value,
                    "shape": tensor.shape,
                    "offset": tensor.data_offset,
                    "length": tensor.byte_length,
                }
                for tensor in item.tensors
            ],
        }
        for item in inspected
    ]
    return hashlib.sha256(canonical_json_bytes({"files": payload})).hexdigest()


def prepare_model(
    model_path: Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
    hash_mode: HashMode = HashMode.METADATA,
) -> ModelManifest:
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise PreparationError(f"model path is not a directory: {model_path}")
    config_path = model_path / "config.json"
    raw_config = _load_json_object(config_path)
    adapters = {"llama": LlamaArchitectureAdapter, "qwen3": Qwen3ArchitectureAdapter}
    model_type = raw_config.get("model_type")
    if not isinstance(model_type, str) or model_type not in adapters:
        raise PreparationError(f"unsupported model_type: {model_type!r}")
    adapter = adapters[model_type]()
    try:
        description = adapter.describe(raw_config)
    except ValueError as error:
        raise PreparationError(str(error)) from error

    paths, weight_map, index_path = _discover_tensor_files(model_path)
    inspected = tuple(inspect_safetensors(path) for path in paths)
    tensors_by_name: dict[str, tuple[InspectedFile, InspectedTensor]] = {}
    for container in inspected:
        for tensor in container.tensors:
            if tensor.name in tensors_by_name:
                raise PreparationError(f"tensor {tensor.name!r} occurs in multiple shards")
            tensors_by_name[tensor.name] = (container, tensor)
    if set(weight_map) != set(tensors_by_name):
        missing_from_files = sorted(set(weight_map) - set(tensors_by_name))
        missing_from_index = sorted(set(tensors_by_name) - set(weight_map))
        raise PreparationError(
            "Safetensors index mismatch: "
            f"missing_from_files={missing_from_files}, missing_from_index={missing_from_index}"
        )
    for name, expected_file in weight_map.items():
        actual_file = tensors_by_name[name][0].path.name
        if expected_file != actual_file:
            raise PreparationError(
                f"tensor {name!r} is indexed in {expected_file!r} but found in {actual_file!r}"
            )

    tensor_records: list[TensorRecord] = []
    component_bytes: defaultdict[tuple[TensorRole, int | None], int] = defaultdict(int)
    for name in sorted(tensors_by_name):
        container, tensor = tensors_by_name[name]
        try:
            ownership = adapter.classify_tensor(name, description.config)
            adapter.validate_tensor_shape(tensor.name, tensor.shape, description.config)
        except ArchitectureError as error:
            raise PreparationError(str(error)) from error
        record = TensorRecord(
            name=name,
            file=container.path.name,
            dtype=tensor.dtype,
            shape=tensor.shape,
            data_offset=tensor.data_offset,
            byte_length=tensor.byte_length,
            role=ownership.role,
            layer_index=ownership.layer_index,
            shared_weight_group=ownership.shared_weight_group,
        )
        tensor_records.append(record)
        component_bytes[(record.role, record.layer_index)] += record.byte_length

    actual_names = {record.name for record in tensor_records}
    expected_names = adapter.expected_tensor_names(description.config)
    if not expected_names.issubset(actual_names):
        raise PreparationError(
            f"model is missing required tensors: {sorted(expected_names - actual_names)}"
        )
    roles = {record.role for record in tensor_records}
    required_roles = {TensorRole.TOKEN_EMBEDDING, TensorRole.FINAL_NORM}
    if not description.config.tied_embeddings:
        required_roles.add(TensorRole.LM_HEAD)
    missing_roles = required_roles - roles
    if missing_roles:
        raise PreparationError(f"model is missing required tensor roles: {sorted(missing_roles)}")

    tensor_files = tuple(
        TensorFile(
            name=item.path.name,
            size_bytes=item.size_bytes,
            header_size_bytes=item.header_size_bytes,
            sha256=sha256_file(item.path) if hash_mode == HashMode.FULL else None,
        )
        for item in inspected
    )
    components = tuple(
        ComponentMemory(role=role, layer_index=layer_index, storage_bytes=size)
        for (role, layer_index), size in sorted(
            component_bytes.items(),
            key=lambda item: (item[0][0].value, -1 if item[0][1] is None else item[0][1]),
        )
    )
    config_sha256 = sha256_file(config_path)
    index_sha256 = sha256_file(index_path) if index_path else None
    tensor_metadata_sha256 = _metadata_digest(inspected) if hash_mode != HashMode.NONE else None

    source = SourceDescriptor(
        model_id=model_id or model_path.name,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        tensor_metadata_sha256=tensor_metadata_sha256,
    )
    total_storage_bytes = sum(item.byte_length for item in tensor_records)
    unsigned = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": source.model_dump(mode="json"),
        "architecture": description.architecture.model_dump(mode="json"),
        "config": description.config.model_dump(mode="json"),
        "tensor_files": [item.model_dump(mode="json") for item in tensor_files],
        "tensors": [item.model_dump(mode="json") for item in tensor_records],
        "components": [item.model_dump(mode="json") for item in components],
        "total_storage_bytes": total_storage_bytes,
    }
    manifest_digest = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return ModelManifest(
        manifest_id=f"manifest-{manifest_digest[:16]}",
        manifest_digest=manifest_digest,
        source=source,
        architecture=description.architecture,
        config=description.config,
        tensor_files=tensor_files,
        tensors=tuple(tensor_records),
        components=components,
        total_storage_bytes=total_storage_bytes,
    )
