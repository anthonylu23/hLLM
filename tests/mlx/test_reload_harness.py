"""Tiny-model plumbing check for the overnight harness, not 4B qualification."""

import json
import sys

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DType, WorkloadProfile
from hllm_control.prepare.manifest import HashMode, prepare_model
from hllm_control.profiling.memory import MlxFitPolicy
from hllm_control.profiling.models import MemoryAmounts, digest
from hllm_control.profiling.runner import run_memory_profile
from hllm_control.proto import common_pb2
from hllm_control.serialization import sha256_file

from scripts.validation import reload_soak
from tests.mlx.helpers import MLX_BINARY
from tests.process_helpers import Workers, plan, wait_clean, write_model


def test_reload_harness_checks_fresh_fit_reference_and_cleanup(tmp_path, monkeypatch):
    write_model(tmp_path)
    manifest = prepare_model(tmp_path, hash_mode=HashMode.FULL)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json())
    deployment = plan(manifest, split=None).model_copy(update={"execution_dtype": DType.F16})
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(deployment.model_dump_json())
    ids = [1, 4, 2, 8, 3]
    with Workers(tmp_path) as reference:
        with DeploymentSession(
            manifest, plan(manifest, split=None), reference.endpoints
        ) as session:
            stream = session.generate(ids, maximum_new_tokens=4, stop_token_ids=[])
            try:
                expected = [e.token.token_id for e in stream if e.HasField("token")]
            finally:
                stream.close()
    files = {
        "config.json": manifest.source.config_sha256,
        **{s.name: s.sha256 for s in manifest.tensor_files},
    }
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(
            dict(
                checkpoint_files=files,
                checkpoint_digest=digest(files),
                producer=dict(
                    dtype="f16",
                    attention="eager",
                    tf32=False,
                    note="synthetic harness smoke; tokens from native CPU, not a full-model oracle",
                ),
                long_generation=dict(token_ids=ids, generated_ids=expected),
            )
        )
    )
    memory_path = tmp_path / "memory.json"
    run_memory_profile(
        binary=MLX_BINARY.with_name("hllm-profile-memory-mlx"),
        root=tmp_path,
        manifest=manifest,
        plan=deployment,
        stage_index=0,
        workload=WorkloadProfile(
            workload_id="reload-smoke",
            prompt_tokens=5,
            output_tokens=4,
            total_cached_tokens=9,
            kv_dtype=DType.F16,
        ),
        backend=Backend.MLX,
        capacity=MemoryAmounts(unified=32 * 1024**2),
        output=memory_path,
        source_revision="test",
        concurrent_load="test suite",
        measured_cycles=1,
        mlx_fit_policy=MlxFitPolicy.FOOTPRINT,
    )
    with Workers(tmp_path, limit=32 * 1024**2, binaries=(MLX_BINARY, MLX_BINARY)) as workers:
        identity = workers.controls[0].GetQualificationState(common_pb2.Empty(), timeout=5)
        config = tmp_path / "workers.json"
        config.write_text(
            json.dumps(
                {
                    "cpu-a": dict(
                        endpoint=workers.endpoints["cpu-a"],
                        memory_profile=str(memory_path),
                        binary_digest=sha256_file(MLX_BINARY),
                        runtime_driver_version=identity.driver_version,
                        mlx_fit_policy="footprint-v1",
                    )
                }
            )
        )
        output = tmp_path / "soak.json"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "reload-soak",
                "--manifest",
                str(manifest_path),
                "--plan",
                str(plan_path),
                "--reference",
                str(reference_path),
                "--workers",
                str(config),
                "--output",
                str(output),
                "--cycles",
                "2",
                "--requests",
                "1",
                "--idle-seconds",
                "0",
            ],
        )
        reload_soak.main()
        result = json.loads(output.read_text())
        assert result["completed"] and len(result["cycles"]) == 2
        for cycle in result["cycles"]:
            fit = cycle["preflight"]["cpu-a"]["assessment"]
            assert fit["status"] == "safe" and fit["policy"] == "mlx-footprint-max-v1"
            assert cycle["requests"][0]["exact_reference_match"]
        wait_clean(workers, loaded=False)
        # Reject a fresh availability/headroom failure before publishing a model.
        settings = json.loads(config.read_text())
        settings["cpu-a"]["host_headroom_bytes"] = 2**60
        config.write_text(json.dumps(settings))
        rejected = tmp_path / "rejected.json"
        monkeypatch.setattr(
            sys, "argv", [str(rejected) if v == str(output) else v for v in sys.argv]
        )
        with pytest.raises(RuntimeError, match="fresh physical-fit gate rejected"):
            reload_soak.main()
        failure = json.loads(rejected.read_text())
        assert not failure["completed"]
        assert failure["cycles"][0]["preflight"]["cpu-a"]["assessment"]["status"] == "unsafe"
        wait_clean(workers, loaded=False)
