from __future__ import annotations

import json
from pathlib import Path

from conftest import tiny_config, tiny_tensors, write_safetensors
from hllm_control.models import (
    Backend,
    DType,
    MemoryBudget,
    MemoryDomain,
    PlannerSettings,
    WorkerProfile,
    WorkloadProfile,
)
from hllm_control.planner.config import load_links, load_workers, load_workload
from hllm_control.planner.planner import create_plan
from hllm_control.prepare.manifest import prepare_model
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st


def worker(worker_id: str, backend: Backend, capacity: int) -> WorkerProfile:
    domain = MemoryDomain.UNIFIED if backend == Backend.MLX else MemoryDomain.DEVICE
    return WorkerProfile(
        worker_id=worker_id,
        endpoint=f"{worker_id}:50051",
        backend=backend,
        primary_memory_domain=domain,
        supported_architectures=("llama.v1",),
        supported_execution_dtypes=(DType.F16,),
        memory_budgets=(
            MemoryBudget(
                domain=domain,
                capacity_bytes=capacity,
                runtime_reserve_bytes=0,
                safety_fraction=0.0,
            ),
        ),
        fixed_workspace_bytes=0,
        activation_buffer_count=1,
        allocator_allowance_fraction=0.0,
    )


def workload(total_cached_tokens: int = 64) -> WorkloadProfile:
    return WorkloadProfile(
        workload_id="test",
        prompt_tokens=8,
        output_tokens=8,
        concurrency=1,
        total_cached_tokens=total_cached_tokens,
    )


def test_enumerates_orders_and_all_splits(tiny_model: Path) -> None:
    manifest = prepare_model(tiny_model)
    workers = (worker("mac", Backend.MLX, 10_000_000), worker("cuda", Backend.CUDA, 10_000_000))

    report = create_plan(manifest, workers, (), workload())

    assert len(report.candidates) == 2 * (manifest.config.num_layers - 1)
    assert report.plan is not None
    assert report.selected_candidate_id is not None
    ranks = [item.rank for item in report.candidates if item.feasible]
    assert all(rank is not None for rank in ranks)
    assert sorted(rank for rank in ranks if rank is not None) == list(
        range(1, len(report.candidates) + 1)
    )
    for candidate in report.candidates:
        assert candidate.stages[0].layer_start == 0
        assert candidate.stages[0].layer_end == candidate.split_layer
        assert candidate.stages[1].layer_start == candidate.split_layer
        assert candidate.stages[1].layer_end == manifest.config.num_layers


def test_reports_no_feasible_plan(tiny_model: Path) -> None:
    manifest = prepare_model(tiny_model)
    workers = (worker("mac", Backend.MLX, 1), worker("cuda", Backend.CUDA, 1))

    report = create_plan(manifest, workers, (), workload())

    assert report.plan is None
    assert all(not item.feasible for item in report.candidates)
    assert all(item.rejection_reasons for item in report.candidates)


def test_tied_embeddings_are_explicitly_duplicated(tmp_path: Path) -> None:
    model_path = tmp_path / "tied"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps(tiny_config(tied=True)), encoding="utf-8")
    write_safetensors(model_path / "model.safetensors", tiny_tensors(tied=True))
    manifest = prepare_model(model_path)
    workers = (worker("mac", Backend.MLX, 10_000_000), worker("cuda", Backend.CUDA, 10_000_000))

    report = create_plan(manifest, workers, (), workload())

    assert report.plan is not None
    assert report.plan.duplicated_tensor_groups == ("token_embeddings",)


@given(st.integers(min_value=1, max_value=4096), st.integers(min_value=1, max_value=4096))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_kv_memory_is_monotonic(tiny_model: Path, smaller: int, increment: int) -> None:
    manifest = prepare_model(tiny_model)
    workers = (worker("mac", Backend.MLX, 100_000_000), worker("cuda", Backend.CUDA, 100_000_000))
    settings = PlannerSettings()

    before = create_plan(manifest, workers, (), workload(smaller), settings).candidates[0]
    after = create_plan(manifest, workers, (), workload(smaller + increment), settings).candidates[
        0
    ]

    assert after.stages[0].kv_cache_bytes >= before.stages[0].kv_cache_bytes
    assert after.stages[1].kv_cache_bytes >= before.stages[1].kv_cache_bytes


def test_checked_in_hardware_profiles_load() -> None:
    root = Path(__file__).parents[2]
    workers = load_workers(root / "examples/profiles/workers-m3pro-3060ti.yaml")
    links = load_links(root / "examples/profiles/links-m3pro-3060ti.yaml")
    loaded_workload = load_workload(root / "examples/workloads/interactive.yaml")

    assert workers[0].primary_memory_domain == MemoryDomain.UNIFIED
    assert workers[1].primary_memory_domain == MemoryDomain.DEVICE
    assert len(links) == 2
    assert loaded_workload.total_cached_tokens == 32768


def test_falcon3_shape_fixture_enumerates_42_realistic_candidates(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    model_path = tmp_path / "falcon3-shape"
    model_path.mkdir()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 3072,
        "intermediate_size": 9216,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 131072,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": False,
    }
    (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, tuple[DType, tuple[int, ...]]] = {
        "model.embed_tokens.weight": (DType.BF16, (131072, 3072)),
        "model.norm.weight": (DType.BF16, (3072,)),
        "lm_head.weight": (DType.BF16, (131072, 3072)),
    }
    for layer in range(22):
        prefix = f"model.layers.{layer}"
        tensors.update(
            {
                f"{prefix}.input_layernorm.weight": (DType.BF16, (3072,)),
                f"{prefix}.post_attention_layernorm.weight": (DType.BF16, (3072,)),
                f"{prefix}.self_attn.q_proj.weight": (DType.BF16, (3072, 3072)),
                f"{prefix}.self_attn.k_proj.weight": (DType.BF16, (1024, 3072)),
                f"{prefix}.self_attn.v_proj.weight": (DType.BF16, (1024, 3072)),
                f"{prefix}.self_attn.o_proj.weight": (DType.BF16, (3072, 3072)),
                f"{prefix}.mlp.gate_proj.weight": (DType.BF16, (9216, 3072)),
                f"{prefix}.mlp.up_proj.weight": (DType.BF16, (9216, 3072)),
                f"{prefix}.mlp.down_proj.weight": (DType.BF16, (3072, 9216)),
            }
        )
    write_safetensors(model_path / "model.safetensors", dict(sorted(tensors.items())), sparse=True)

    manifest = prepare_model(model_path, model_id="tiiuae/Falcon3-3B-Base-shape-fixture")
    report = create_plan(
        manifest,
        load_workers(root / "examples/profiles/workers-m3pro-3060ti.yaml"),
        load_links(root / "examples/profiles/links-m3pro-3060ti.yaml"),
        load_workload(root / "examples/workloads/interactive.yaml"),
    )

    assert manifest.config.num_layers == 22
    assert manifest.total_storage_bytes > 6_000_000_000
    assert len(report.candidates) == 42
    assert report.plan is not None
    assert any(not candidate.feasible for candidate in report.candidates)
