"""Two-worker CPU rehearsal: procedure evidence, not an independent model oracle."""

import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

import grpc
import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DType, WorkloadProfile
from hllm_control.prepare.manifest import HashMode, prepare_model
from hllm_control.profiling.models import MemoryAmounts, make_artifact
from hllm_control.profiling.runner import run_memory_profile
from hllm_control.proto import common_pb2
from hllm_control.serialization import sha256_file

from scripts.validation import reload_soak
from tests.process_helpers import BINARY, Workers, plan, wait_clean, write_model

PROFILER = Path(os.environ.get("HLLM_MEMORY_PROFILER", BINARY.with_name("hllm-profile-memory-cpu")))
CAP = 32 * 1024**2


@pytest.fixture
def rehearsal(tmp_path, monkeypatch):
    write_model(tmp_path)
    manifest = prepare_model(tmp_path, hash_mode=HashMode.FULL)
    deployment = plan(manifest)
    ids = [1, 4, 2, 8, 3]
    with Workers(tmp_path, limit=CAP) as workers:
        with DeploymentSession(manifest, plan(manifest, split=None), workers.endpoints) as session:
            expected = [
                e.token.token_id
                for e in session.generate(ids, maximum_new_tokens=4, stop_token_ids=[])
                if e.HasField("token")
            ]
        wait_clean(workers, loaded=False)
        files = {
            "config.json": manifest.source.config_sha256,
            **{s.name: s.sha256 for s in manifest.tensor_files},
        }
        reference = dict(
            checkpoint_files=files,
            checkpoint_digest=reload_soak.digest(files),
            producer=dict(dtype="f32", attention="eager", tf32=False, purpose="cpu-rehearsal"),
            long_generation=dict(token_ids=ids, generated_ids=expected),
        )
        settings = {}
        profiles = []
        for i, name in enumerate(workers.endpoints):
            memory_path = tmp_path / f"{name}.memory.json"
            profile = run_memory_profile(
                binary=PROFILER,
                root=tmp_path,
                manifest=manifest,
                plan=deployment,
                stage_index=i,
                workload=WorkloadProfile(
                    workload_id="cpu-rehearsal",
                    prompt_tokens=5,
                    output_tokens=4,
                    total_cached_tokens=9,
                    kv_dtype=DType.F32,
                ),
                backend=Backend.CPU,
                capacity=MemoryAmounts(host=CAP),
                output=memory_path,
                source_revision="cpu-rehearsal",
                concurrent_load="serial tiny CPU tests",
                measured_cycles=1,
                timeout_seconds=30,
            )
            profiles.append(profile)
            q = workers.controls[i].GetQualificationState(common_pb2.Empty(), timeout=5)
            settings[name] = dict(
                endpoint=workers.endpoints[name],
                memory_profile=str(memory_path),
                binary_digest=sha256_file(BINARY),
                runtime_driver_version=q.driver_version,
            )
        for name, value in (
            ("manifest", manifest.model_dump(mode="json")),
            ("plan", deployment.model_dump(mode="json")),
            ("reference", reference),
            ("workers", settings),
        ):
            (tmp_path / f"{name}.json").write_text(json.dumps(value))
        args = ["reload-soak", "--cpu-rehearsal"]
        for name in ("manifest", "plan", "reference", "workers", "output"):
            args.extend(["--" + name, str(tmp_path / f"{name}.json")])
        args.extend(["--cycles", "2", "--requests", "2", "--idle-seconds", "0", "--timeout", "5"])
        monkeypatch.setattr(sys, "argv", args)
        yield tmp_path, workers, profiles


def test_cpu_rehearsal_reload_cancel_unload(rehearsal, monkeypatch):
    root, workers, _ = rehearsal
    args = sys.argv.copy()
    monkeypatch.setattr(sys, "argv", [v for v in args if v != "--cpu-rehearsal"])
    with pytest.raises(ValueError, match="selected uniform precision"):
        reload_soak.main()
    monkeypatch.setattr(sys, "argv", args)
    reference = json.loads((root / "reference.json").read_text())
    manifest = prepare_model(root, hash_mode=HashMode.FULL)
    with pytest.raises(ValueError, match="reference precision/purpose mismatch"):
        reload_soak.reference_tokens(reference, manifest)
    reload_soak.main()
    report = json.loads((root / "output.json").read_text())
    assert report["completed"] and report["scope"] == "cpu-rehearsal"
    assert len(report["cycles"]) == 2
    for cycle in report["cycles"]:
        assert all(r["exact_reference_match"] for r in cycle["requests"])
        assert len(cycle["requests"]) == 2
        assert set(cycle["preflight"]) == {"cpu-a", "cpu-b"}
        assert all(r["assessment"]["status"] == "safe" for r in cycle["preflight"].values())
        assert all(r["assessment"]["policy"] == "cpu-rss-v1" for r in cycle["preflight"].values())
        assert "after_cancel" in cycle and "after_unload_idle" in cycle
        assert cycle["cancellation"]["exact_prefix_match"]
        assert cycle["cancellation"]["generated_ids"] == cycle["cancellation"]["expected_prefix"]
        assert len(cycle["cancellation"]["generated_ids"]) == 3
    wait_clean(workers, loaded=False)
    observations = [
        json.loads(r) for r in (root / "output.observations.jsonl").read_text().splitlines()
    ]
    assert {"after-load", "cancel-cleanup", "unloaded", "unloaded-idle"} <= {
        r["phase"] for r in observations
    }


@pytest.mark.parametrize("fault", ["stale", "future", "binary", "capacity", "workload"])
def test_cpu_rehearsal_rejects_bad_evidence_before_loading(rehearsal, monkeypatch, fault):
    root, workers, profiles = rehearsal
    profile = profiles[0]
    if fault in ("stale", "future"):
        conditions = profile.conditions.model_copy(
            update={
                "measured_at": datetime.now(UTC) + timedelta(hours=-25 if fault == "stale" else 1)
            }
        )
        profile = make_artifact(profile.key, conditions, profile.measurement)
    elif fault == "capacity":
        measurement = profile.measurement.model_copy(
            update={"admission_capacity": MemoryAmounts(host=CAP + 1)}
        )
        profile = make_artifact(profile.key, profile.conditions, measurement)
    elif fault == "workload":
        w = profile.key.workload.model_copy(update={"prompt_tokens": 6, "total_cached_tokens": 10})
        key = profile.key.model_copy(
            update={"workload": w, "workload_digest": reload_soak.digest(w)}
        )
        profile = make_artifact(key, profile.conditions, profile.measurement)
    else:
        config = json.loads((root / "workers.json").read_text())
        config["cpu-a"]["binary_digest"] = "0" * 64
        (root / "workers.json").write_text(json.dumps(config))
    (root / "cpu-a.memory.json").write_text(profile.model_dump_json())
    enter = Mock(side_effect=AssertionError("rejected evidence must not start a deployment"))
    monkeypatch.setattr(DeploymentSession, "__enter__", enter)
    with pytest.raises(
        ValueError,
        match={
            "stale": "stale",
            "future": "future",
            "binary": "binary/environment",
            "capacity": "admission cap",
            "workload": "scope",
        }[fault],
    ):
        reload_soak.main()
    enter.assert_not_called()
    wait_clean(workers, loaded=False)
    if fault != "workload":
        report = json.loads((root / "output.json").read_text())
        assert not report["completed"]
        # Rejected before any stage load: no cycle reached its after-load snapshot.
        assert report["cycles"] and all("after_load" not in c for c in report["cycles"])
        observations = (root / "output.observations.jsonl").read_text().splitlines()
        assert "after-load" not in {json.loads(r)["phase"] for r in observations}


def test_cpu_rehearsal_refuses_busy_workers_without_disturbing_deployment(rehearsal, monkeypatch):
    root, workers, _ = rehearsal
    manifest = prepare_model(root, hash_mode=HashMode.FULL)
    reference = json.loads((root / "reference.json").read_text())["long_generation"]
    with DeploymentSession(manifest, plan(manifest), workers.endpoints) as existing:
        before = existing.memory_reports()
        assert all(r.loaded_weight_bytes > 0 for r in before)
        enter = Mock(side_effect=AssertionError("busy workers must not be redeployed"))
        fit = Mock(
            side_effect=AssertionError("busy workers must be refused before profiling gates")
        )
        monkeypatch.setattr(DeploymentSession, "__enter__", enter)
        monkeypatch.setattr(reload_soak, "fresh_fit", fit)
        with pytest.raises(RuntimeError, match="workers are busy"):
            reload_soak.main()
        enter.assert_not_called()
        fit.assert_not_called()
        report = json.loads((root / "output.json").read_text())
        assert not report["completed"] and not report["cycles"]
        assert "workers are busy" in report["error"]
        assert all("memory" in r for r in report["failure_cleanup"].values())
        assert [r.loaded_weight_bytes for r in existing.memory_reports()] == [
            r.loaded_weight_bytes for r in before
        ]
        # A refusal must not unload, cancel or replace somebody else's deployment.
        stream = existing.generate(reference["token_ids"], maximum_new_tokens=4, stop_token_ids=[])
        try:
            assert [e.token.token_id for e in stream if e.HasField("token")] == reference[
                "generated_ids"
            ]
        finally:
            stream.close()
    wait_clean(workers, loaded=False)


@pytest.mark.parametrize("fault", ["tokens", "worker-loss"])
def test_cpu_rehearsal_preserves_failure_and_cleans_surviving_workers(
    rehearsal, monkeypatch, fault
):
    root, workers, _ = rehearsal
    if fault == "tokens":
        reference = json.loads((root / "reference.json").read_text())
        reference["long_generation"]["generated_ids"][0] ^= 1
        (root / "reference.json").write_text(json.dumps(reference))
    else:
        generate = DeploymentSession.generate

        def interrupted(self, *args, **kwargs):
            stream = generate(self, *args, **kwargs)
            try:
                for event in stream:
                    yield event
                    if event.HasField("token"):
                        workers.processes[1].terminate()
                        workers.processes[1].wait(timeout=5)
                        # Observe a real failed RPC, after one token was recorded.
                        workers.controls[1].GetMemoryReport(common_pb2.Empty(), timeout=0.5)
            finally:
                stream.close()

        monkeypatch.setattr(DeploymentSession, "generate", interrupted)
    with pytest.raises((RuntimeError, grpc.RpcError)):
        reload_soak.main()
    report = json.loads((root / "output.json").read_text())
    assert not report["completed"] and report["error"]
    request = report["cycles"][0]["requests"][0]
    assert request["generated_ids"] and not request["exact_reference_match"]
    alive = [0, 1] if fault == "tokens" else [0]
    for i in alive:
        r = workers.controls[i].GetMemoryReport(common_pb2.Empty(), timeout=2)
        assert r.active_requests == r.reserved_cache_bytes == r.reserved_workspace_bytes == 0
        assert r.loaded_weight_bytes == 0
    if fault == "worker-loss":
        assert len(request["generated_ids"]) == 1 and request["error"]
        assert "error" in report["failure_cleanup"]["cpu-b"]
