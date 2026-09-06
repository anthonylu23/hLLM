"""Strict Llama-family configuration and tensor classification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

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
class ModelDescription:
    architecture: ArchitectureDescriptor
    config: ModelConfig


def _required_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ArchitectureError(f"config field {key!r} must be a positive integer")
    return value


def _positive_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ArchitectureError(f"config field {key!r} must be a positive number")
    return float(value)


def _boolean(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ArchitectureError(f"config field {key!r} must be a boolean")
    return value


def _eos_token_ids(config: Mapping[str, Any]) -> tuple[int, ...]:
    value = config.get("eos_token_id")
    if value is None:
        return ()
    values = cast(list[object], value) if isinstance(value, list) else [value]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in values):
        raise ArchitectureError(
            "config field 'eos_token_id' must be a non-negative integer or list"
        )
    return tuple(dict.fromkeys(cast(int, item) for item in values))


class LlamaArchitectureAdapter:
    architecture_id = "llama.v1"
    architecture_revision = "1"
    model_type = "llama"
    model_class = "LlamaForCausalLM"
    layer_suffixes = LAYER_SUFFIXES

    def describe(self, raw: Mapping[str, Any]) -> ModelDescription:
        if raw.get("model_type") != self.model_type:
            raise ArchitectureError(f"expected model_type={self.model_type!r}")
        architectures = raw.get("architectures", [])
        if architectures and self.model_class not in architectures:
            raise ArchitectureError(f"config does not declare {self.model_class}")

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

        tied_embeddings = _boolean(raw, "tie_word_embeddings", False)
        attention_bias = _boolean(raw, "attention_bias", False)
        mlp_bias = _boolean(raw, "mlp_bias", False)
        if attention_bias or mlp_bias:
            raise ArchitectureError(
                f"{self.architecture_id} does not yet support attention or MLP bias tensors"
            )
        hidden_activation = raw.get("hidden_act", "silu")
        if hidden_activation != "silu":
            raise ArchitectureError(f"{self.architecture_id} currently requires hidden_act='silu'")
        pretraining_tp = raw.get("pretraining_tp", 1)
        if pretraining_tp != 1:
            raise ArchitectureError(f"{self.architecture_id} currently requires pretraining_tp=1")
        partial_rotary_factor = raw.get("partial_rotary_factor", 1.0)
        if partial_rotary_factor != 1.0:
            raise ArchitectureError(
                f"{self.architecture_id} currently requires partial_rotary_factor=1.0"
            )
        if raw.get("sliding_window") is not None:
            raise ArchitectureError(
                f"{self.architecture_id} does not yet support sliding-window attention"
            )
        # Workers reject any rope_scaling at load time; refuse it here so a prepared
        # manifest and its plans describe only checkpoints the runtime can execute.
        if raw.get("rope_scaling") is not None:
            raise ArchitectureError(f"{self.architecture_id} currently supports unscaled RoPE only")

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
            rms_norm_eps=_positive_float(raw, "rms_norm_eps", 1e-6),
            rope_theta=_positive_float(raw, "rope_theta", 10_000.0),
            hidden_activation=hidden_activation,
            attention_bias=attention_bias,
            mlp_bias=mlp_bias,
            eos_token_ids=_eos_token_ids(raw),
        )
        features = ["gqa" if model_config.num_kv_heads < num_attention_heads else "mha"]
        features.append("tied_embeddings" if tied_embeddings else "untied_embeddings")
        if explicit_head_dim is not None:
            features.append("explicit_head_dim")
        return ModelDescription(
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
            if suffix not in self.layer_suffixes:
                raise ArchitectureError(
                    f"unrecognized {self.model_type.capitalize()} layer tensor {name!r}"
                )
            return TensorOwnership(TensorRole.TRANSFORMER_LAYER, layer_index=layer_index)
        if name == "model.norm.weight":
            return TensorOwnership(TensorRole.FINAL_NORM)
        if name == "lm_head.weight":
            group = "token_embeddings" if config.tied_embeddings else None
            return TensorOwnership(TensorRole.LM_HEAD, shared_weight_group=group)
        if name in {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}:
            return TensorOwnership(TensorRole.ARCHITECTURE_STATE)
        raise ArchitectureError(f"unrecognized {self.model_type.capitalize()} tensor {name!r}")

    def expected_tensor_names(self, config: ModelConfig) -> set[str]:
        names = {
            "model.embed_tokens.weight",
            "model.norm.weight",
            *(
                f"model.layers.{layer}.{suffix}"
                for layer in range(config.num_layers)
                for suffix in self.layer_suffixes
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
