from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import tiny_config, tiny_tensors, write_safetensors
from hllm_control.models import TensorRole
from hllm_control.prepare.manifest import HashMode, PreparationError, prepare_model


def test_prepares_deterministic_sharded_manifest(tiny_model: Path) -> None:
    first = prepare_model(
        tiny_model,
        model_id="test/tiny-llama",
        revision="0123456789abcdef",
    )
    second = prepare_model(
        tiny_model,
        model_id="test/tiny-llama",
        revision="0123456789abcdef",
    )

    assert first == second
    assert first.manifest_digest == second.manifest_digest
    assert first.config.num_layers == 4
    assert first.config.rms_norm_eps == 1e-5
    assert first.config.rope_theta == 500_000.0
    assert first.config.eos_token_ids == (2, 3)
    assert len(first.tensor_files) == 2
    assert first.total_storage_bytes == sum(tensor.byte_length for tensor in first.tensors)
    assert {item.layer_index for item in first.components if item.layer_index is not None} == {
        0,
        1,
        2,
        3,
    }


def test_full_hashing_records_shard_hashes(tiny_model: Path) -> None:
    manifest = prepare_model(tiny_model, hash_mode=HashMode.FULL)
    assert all(item.sha256 is not None and len(item.sha256) == 64 for item in manifest.tensor_files)


def test_prepares_tied_single_file_model(tmp_path: Path) -> None:
    model_path = tmp_path / "tied"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(tiny_config(tied=True)), encoding="utf-8")
    write_safetensors(model_path / "model.safetensors", tiny_tensors(tied=True))

    manifest = prepare_model(model_path)

    assert manifest.config.tied_embeddings
    assert not any(item.role == TensorRole.LM_HEAD for item in manifest.tensors)
    embedding = next(item for item in manifest.tensors if item.role == TensorRole.TOKEN_EMBEDDING)
    assert embedding.shared_weight_group == "token_embeddings"


def test_rejects_unclassified_tensor(tmp_path: Path) -> None:
    model_path = tmp_path / "unexpected"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    tensors = tiny_tensors()
    tensors["surprise.weight"] = tensors["model.norm.weight"]
    write_safetensors(model_path / "model.safetensors", tensors)

    with pytest.raises(PreparationError, match="unrecognized Llama tensor"):
        prepare_model(model_path)


def test_rejects_missing_required_layer_tensor(tmp_path: Path) -> None:
    model_path = tmp_path / "missing"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    tensors = tiny_tensors()
    del tensors["model.layers.2.self_attn.q_proj.weight"]
    write_safetensors(model_path / "model.safetensors", tensors)

    with pytest.raises(PreparationError, match="missing required tensors"):
        prepare_model(model_path)


def test_rejects_wrong_tensor_shape(tmp_path: Path) -> None:
    model_path = tmp_path / "wrong-shape"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(tiny_config()), encoding="utf-8")
    tensors = tiny_tensors()
    dtype, _ = tensors["model.layers.1.self_attn.k_proj.weight"]
    tensors["model.layers.1.self_attn.k_proj.weight"] = (dtype, (8, 8))
    write_safetensors(model_path / "model.safetensors", tensors)

    with pytest.raises(PreparationError, match="expected"):
        prepare_model(model_path)


def test_rejects_runtime_semantics_the_cpu_backend_cannot_execute(tmp_path: Path) -> None:
    model_path = tmp_path / "biased"
    model_path.mkdir()
    config = tiny_config()
    config["attention_bias"] = True
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_safetensors(model_path / "model.safetensors", tiny_tensors())

    with pytest.raises(PreparationError, match="bias tensors"):
        prepare_model(model_path)
