from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_safetensors
from hllm_control.models import DType, TensorRole
from hllm_control.planner.config import load_links, load_workers, load_workload
from hllm_control.planner.planner import create_plan
from hllm_control.prepare.manifest import PreparationError, prepare_model
from hllm_control.wire import model_manifest_from_proto, model_manifest_to_proto

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests/fixtures/qwen3"


def qwen_config(*, full_size: bool = False) -> dict[str, object]:
    config = json.loads((FIXTURES / "config.json").read_text())
    if not full_size:
        config.update(
            hidden_size=6,
            intermediate_size=10,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=11,
            max_position_embeddings=32,
        )
    return config


def qwen_tensors(*, full_size: bool = False) -> dict[str, tuple[DType, tuple[int, ...]]]:
    hidden, intermediate, layers, heads, kv_heads, dim, vocab = (
        (2560, 9728, 36, 32, 8, 128, 151936) if full_size else (6, 10, 2, 4, 2, 4, 11)
    )
    shapes = {
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (heads * dim, hidden),
        "self_attn.k_proj.weight": (kv_heads * dim, hidden),
        "self_attn.v_proj.weight": (kv_heads * dim, hidden),
        "self_attn.o_proj.weight": (hidden, heads * dim),
        "self_attn.q_norm.weight": (dim,),
        "self_attn.k_norm.weight": (dim,),
        "mlp.gate_proj.weight": (intermediate, hidden),
        "mlp.up_proj.weight": (intermediate, hidden),
        "mlp.down_proj.weight": (hidden, intermediate),
    }
    tensors = {
        f"model.layers.{layer}.{suffix}": (DType.BF16, shape)
        for layer in range(layers)
        for suffix, shape in shapes.items()
    }
    tensors["model.embed_tokens.weight"] = (DType.BF16, (vocab, hidden))
    tensors["model.norm.weight"] = (DType.BF16, (hidden,))
    return tensors


def write_model(path: Path, config: dict[str, object]) -> None:
    (path / "config.json").write_text(json.dumps(config))
    write_safetensors(path / "model.safetensors", qwen_tensors())


def test_qwen_preparation_and_wire_roundtrip(tmp_path: Path) -> None:
    write_model(tmp_path, qwen_config())
    manifest = prepare_model(tmp_path)
    assert manifest.architecture.architecture_id == "qwen3.v1"
    assert "qk_norm" in manifest.architecture.feature_flags
    assert (
        manifest.config.num_attention_heads * manifest.config.head_dim
        != manifest.config.hidden_size
    )
    assert manifest.config.tied_embeddings
    assert not any(t.role == TensorRole.LM_HEAD for t in manifest.tensors)
    assert len(manifest.tensors) == 24
    assert model_manifest_from_proto(model_manifest_to_proto(manifest)) == manifest
    assert prepare_model(tmp_path) == manifest


@pytest.mark.parametrize("suffix", ["q_norm", "k_norm"])
@pytest.mark.parametrize("missing", [True, False])
def test_qwen_requires_per_head_norm_weights(tmp_path: Path, suffix: str, missing: bool) -> None:
    write_model(tmp_path, qwen_config())
    tensors = qwen_tensors()
    name = f"model.layers.0.self_attn.{suffix}.weight"
    if missing:
        del tensors[name]
    else:
        tensors[name] = (DType.BF16, (6,))  # hidden width is not the per-head width
    write_safetensors(tmp_path / "model.safetensors", tensors)
    with pytest.raises(PreparationError, match=r"missing required tensors|expected"):
        prepare_model(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_type", "qwen3_moe"),
        ("attention_bias", True),
        ("use_sliding_window", True),
        ("sliding_window", 16),
        ("rope_scaling", {"rope_type": "linear", "factor": 2}),
        ("architectures", ["LlamaForCausalLM"]),
        ("head_dim", 3),
    ],
)
def test_rejects_unsupported_qwen_semantics(tmp_path: Path, field: str, value: object) -> None:
    config = qwen_config()
    config[field] = value
    write_model(tmp_path, config)
    with pytest.raises(PreparationError):
        prepare_model(tmp_path)


def test_qwen4b_shape_fixture_matches_upstream_and_plans(tmp_path: Path) -> None:
    config = qwen_config(full_size=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = json.loads((FIXTURES / "model.safetensors.index.json").read_text())
    tensors = qwen_tensors(full_size=True)
    assert set(tensors) == set(index["weight_map"])
    metadata = json.loads((FIXTURES / "tensor-metadata.json").read_text())
    assert {
        name: {"dtype": dtype.value, "shape": list(shape)}
        for name, (dtype, shape) in tensors.items()
    } == metadata
    for shard in sorted(set(index["weight_map"].values())):
        write_safetensors(
            tmp_path / shard,
            {
                name: tensor
                for name, tensor in tensors.items()
                if index["weight_map"][name] == shard
            },
            sparse=True,
        )
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    manifest = prepare_model(
        tmp_path,
        model_id="Qwen/Qwen3-4B-Base",
        revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
    )
    assert len(manifest.tensors) == 398
    # The pinned index total_size is stale; actual shard headers sum to this value.
    assert manifest.total_storage_bytes == 8_044_936_192
    report = create_plan(
        manifest,
        load_workers(ROOT / "examples/profiles/workers-m3pro-3060ti.yaml"),
        load_links(ROOT / "examples/profiles/links-m3pro-3060ti.yaml"),
        load_workload(ROOT / "examples/workloads/interactive.yaml"),
    )
    assert len(report.candidates) == 70
    assert report.plan is not None
    assert report.plan.duplicated_tensor_groups == ("token_embeddings",)
    assert any(not candidate.feasible for candidate in report.candidates)
    # Both stages need the shared embedding. Q/K norms count once per owned layer.
    embedding_bytes = 151936 * 2560 * 2
    assert all(
        sum(stage.weight_bytes for stage in candidate.stages)
        == manifest.total_storage_bytes + embedding_bytes
        for candidate in report.candidates
    )
