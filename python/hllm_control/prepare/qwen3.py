"""Dense Qwen3 decoder semantics, including per-head query/key normalization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from hllm_control.models import ModelConfig
from hllm_control.prepare.llama import (
    LAYER_SUFFIXES,
    ArchitectureError,
    LlamaArchitectureAdapter,
    ModelDescription,
)


class Qwen3ArchitectureAdapter(LlamaArchitectureAdapter):
    architecture_id = "qwen3.v1"
    model_type = "qwen3"
    model_class = "Qwen3ForCausalLM"
    layer_suffixes = (*LAYER_SUFFIXES, "self_attn.q_norm.weight", "self_attn.k_norm.weight")

    def describe(self, raw: Mapping[str, Any]) -> ModelDescription:
        if raw.get("use_sliding_window", False) is not False:
            raise ArchitectureError("qwen3.v1 does not support sliding-window attention")
        description = super().describe(raw)
        if description.config.head_dim % 2:
            raise ArchitectureError("qwen3.v1 requires an even head_dim for RoPE")
        return ModelDescription(
            architecture=description.architecture.model_copy(
                update={
                    "feature_flags": tuple(
                        sorted((*description.architecture.feature_flags, "qk_norm"))
                    )
                }
            ),
            config=description.config,
        )

    def validate_tensor_shape(self, name: str, shape: tuple[int, ...], config: ModelConfig) -> None:
        if name.endswith(("self_attn.q_norm.weight", "self_attn.k_norm.weight")):
            expected = (config.head_dim,)
            if shape != expected:
                raise ArchitectureError(f"tensor {name!r} has shape {shape}, expected {expected}")
            return
        super().validate_tensor_shape(name, shape, config)
