"""Strict Llama-family configuration and tensor classification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hllm_control.models import ArchitectureDescriptor, ModelConfig, TensorRole

LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.")
LAYER_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)


class ArchitectureError(ValueError):
    """Raised when a model is not supported by an architecture adapter."""


@dataclass(frozen=True)
class TensorOwnership:
    role: TensorRole
    layer_index: int | None = None
    shared_weight_group: str | None = None


@dataclass(frozen=True)
class LlamaDescription:
    architecture: ArchitectureDescriptor
    config: ModelConfig


def _required_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArchitectureError(f"config field {key!r} must be a positive integer")
    return value


class LlamaArchitectureAdapter:
    architecture_id = "llama.v1"
    architecture_revision = "1"

    def describe(self, raw: Mapping[str, Any]) -> LlamaDescription:
        if raw.get("model_type") != "llama":
            raise ArchitectureError("only model_type='llama' is currently supported")
        architectures = raw.get("architectures", [])
        if architectures and "LlamaForCausalLM" not in architectures:
            raise ArchitectureError("config does not declare LlamaForCausalLM")

        hidden_size = _required_positive_int(raw, "hidden_size")
        num_attention_heads = _required_positive_int(raw, "num_attention_heads")
        explicit_head_dim = raw.get("head_dim")
        if explicit_head_dim is None:
            if hidden_size % num_attention_heads:
                raise ArchitectureError(
                    "hidden_size must divide evenly by num_attention_heads when head_dim is absent"
                )
            head_dim = hidden_size // num_attention_heads
        elif isinstance(explicit_head_dim, int) and not isinstance(explicit_head_dim, bool):
            head_dim = explicit_head_dim
        else:
            raise ArchitectureError("head_dim must be a positive integer")

        tied_embeddings = raw.get("tie_word_embeddings", False)
        if not isinstance(tied_embeddings, bool):
            raise ArchitectureError("tie_word_embeddings must be a boolean")

        model_config = ModelConfig(
            hidden_size=hidden_size,
            intermediate_size=_required_positive_int(raw, "intermediate_size"),
            num_layers=_required_positive_int(raw, "num_hidden_layers"),
            num_attention_heads=num_attention_heads,
            num_kv_heads=_required_positive_int(raw, "num_key_value_heads"),
            head_dim=head_dim,
            vocabulary_size=_required_positive_int(raw, "vocab_size"),
            maximum_sequence_length=_required_positive_int(raw, "max_position_embeddings"),
            tied_embeddings=tied_embeddings,
        )
        features = ["gqa" if model_config.num_kv_heads < num_attention_heads else "mha"]
        features.append("tied_embeddings" if tied_embeddings else "untied_embeddings")
        if explicit_head_dim is not None:
            features.append("explicit_head_dim")
        return LlamaDescription(
            architecture=ArchitectureDescriptor(
                architecture_id=self.architecture_id,
                architecture_revision=self.architecture_revision,
                feature_flags=tuple(sorted(features)),
            ),
            config=model_config,
        )

    def classify_tensor(self, name: str, config: ModelConfig) -> TensorOwnership:
        if name == "model.embed_tokens.weight":
            group = "token_embeddings" if config.tied_embeddings else None
            return TensorOwnership(TensorRole.TOKEN_EMBEDDING, shared_weight_group=group)
        match = LAYER_PATTERN.match(name)
        if match:
            layer_index = int(match.group(1))
            if layer_index >= config.num_layers:
                raise ArchitectureError(
                    f"tensor {name!r} refers to layer {layer_index}, but config has "
                    f"{config.num_layers} layers"
                )
            expected_prefix = f"model.layers.{layer_index}."
            suffix = name.removeprefix(expected_prefix)
            if suffix not in LAYER_SUFFIXES:
                raise ArchitectureError(f"unrecognized Llama layer tensor {name!r}")
            return TensorOwnership(TensorRole.TRANSFORMER_LAYER, layer_index=layer_index)
        if name == "model.norm.weight":
            return TensorOwnership(TensorRole.FINAL_NORM)
        if name == "lm_head.weight":
            group = "token_embeddings" if config.tied_embeddings else None
            return TensorOwnership(TensorRole.LM_HEAD, shared_weight_group=group)
        if name in {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}:
            return TensorOwnership(TensorRole.ARCHITECTURE_STATE)
        raise ArchitectureError(f"unrecognized Llama tensor {name!r}")

    def expected_tensor_names(self, config: ModelConfig) -> set[str]:
        names = {
            "model.embed_tokens.weight",
            "model.norm.weight",
            *(
                f"model.layers.{layer}.{suffix}"
                for layer in range(config.num_layers)
                for suffix in LAYER_SUFFIXES
            ),
        }
        if not config.tied_embeddings:
            names.add("lm_head.weight")
        return names

    def validate_tensor_shape(self, name: str, shape: tuple[int, ...], config: ModelConfig) -> None:
        attention_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_kv_heads * config.head_dim
        expected: tuple[int, ...] | None
        if name in {"model.embed_tokens.weight", "lm_head.weight"}:
            expected = (config.vocabulary_size, config.hidden_size)
        elif name == "model.norm.weight" or name.endswith("layernorm.weight"):
            expected = (config.hidden_size,)
        elif name.endswith("self_attn.q_proj.weight"):
            expected = (attention_width, config.hidden_size)
        elif name.endswith("self_attn.k_proj.weight") or name.endswith("self_attn.v_proj.weight"):
            expected = (kv_width, config.hidden_size)
        elif name.endswith("self_attn.o_proj.weight"):
            expected = (config.hidden_size, attention_width)
        elif name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
            expected = (config.intermediate_size, config.hidden_size)
        elif name.endswith("mlp.down_proj.weight"):
            expected = (config.hidden_size, config.intermediate_size)
        else:
            expected = None  # optional architecture state is backend-derived or checkpoint-specific
        if expected is not None and shape != expected:
            raise ArchitectureError(f"tensor {name!r} has shape {shape}, expected {expected}")
