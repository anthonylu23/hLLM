from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest
from hllm_control.models import DTYPE_BYTES, DType


def write_safetensors(
    path: Path,
    tensors: Mapping[str, tuple[DType, tuple[int, ...]]],
    *,
    sparse: bool = False,
) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    for name, (dtype, shape) in tensors.items():
        elements = 1
        for dimension in shape:
            elements *= dimension
        length = elements * DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype.value,
            "shape": list(shape),
            "data_offsets": [offset, offset + length],
        }
        offset += length
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    prefix = struct.pack("<Q", len(raw_header)) + raw_header
    if not sparse:
        path.write_bytes(prefix + bytes(offset))
        return
    with path.open("wb") as stream:
        stream.write(prefix)
        if offset:
            stream.seek(len(prefix) + offset - 1)
            stream.write(b"\0")


def tiny_config(*, tied: bool = False) -> dict[str, object]:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "vocab_size": 16,
        "max_position_embeddings": 128,
        "tie_word_embeddings": tied,
    }


def tiny_tensors(*, tied: bool = False) -> dict[str, tuple[DType, tuple[int, ...]]]:
    tensors: dict[str, tuple[DType, tuple[int, ...]]] = {
        "model.embed_tokens.weight": (DType.F16, (16, 8)),
        "model.norm.weight": (DType.F16, (8,)),
    }
    if not tied:
        tensors["lm_head.weight"] = (DType.F16, (16, 8))
    for layer in range(4):
        prefix = f"model.layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": (DType.F16, (8,)),
                f"{prefix}.post_attention_layernorm.weight": (DType.F16, (8,)),
                f"{prefix}.self_attn.q_proj.weight": (DType.F16, (8, 8)),
                f"{prefix}.self_attn.k_proj.weight": (DType.F16, (4, 8)),
                f"{prefix}.self_attn.v_proj.weight": (DType.F16, (4, 8)),
                f"{prefix}.self_attn.o_proj.weight": (DType.F16, (8, 8)),
                f"{prefix}.mlp.gate_proj.weight": (DType.F16, (16, 8)),
                f"{prefix}.mlp.up_proj.weight": (DType.F16, (16, 8)),
                f"{prefix}.mlp.down_proj.weight": (DType.F16, (8, 16)),
            }
        )
    return dict(sorted(tensors.items()))


@pytest.fixture
def tiny_model(tmp_path: Path) -> Path:
    model_path = tmp_path / "tiny-llama"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    tensors = tiny_tensors()
    names = list(tensors)
    midpoint = len(names) // 2
    shards = (
        ("model-00001-of-00002.safetensors", names[:midpoint]),
        ("model-00002-of-00002.safetensors", names[midpoint:]),
    )
    weight_map: dict[str, str] = {}
    for filename, shard_names in shards:
        write_safetensors(model_path / filename, {name: tensors[name] for name in shard_names})
        weight_map.update({name: filename for name in shard_names})
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return model_path
