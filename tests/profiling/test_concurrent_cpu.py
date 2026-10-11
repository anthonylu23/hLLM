"""Two-worker CPU rehearsal of the concurrency-two harness: procedure evidence only."""

import json
import os
import sys
from pathlib import Path

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DType, WorkloadProfile
from hllm_control.prepare.manifest import HashMode, prepare_model
from hllm_control.profiling.models import MemoryAmounts
from hllm_control.profiling.runner import run_memory_profile
from hllm_control.proto import common_pb2
from hllm_control.qualification.concurrent import read_evidence
from hllm_control.serialization import sha256_file

from scripts.validation import concurrent_soak, reload_soak, resource_guard
from tests.process_helpers import BINARY, Workers, plan, wait_clean, write_model

PROFILER = Path(os.environ.get("HLLM_MEMORY_PROFILER", BINARY.with_name("hllm-profile-memory-cpu")))
CAP = 32 * 1024**2
OBSERVING = (
    "--max-active-requests",
    "2",
    "--max-cached-tokens",
    "48",
    "--request-observations",
    "on",
)


def prepare(tmp_path: Path, workers: Workers) -> list[str]:
    manifest = prepare_model(tmp_path, hash_mode=HashMode.FULL)
    deployment = plan(manifest)
    ids = [1, 4, 2, 8, 3]
    with DeploymentSession(manifest, plan(manifest, split=None), workers.endpoints) as session:
        expected = [
            e.token.token_id
            for e in session.generate(ids, maximum_new_tokens=16, stop_token_ids=[])
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
    for i, name in enumerate(workers.endpoints):
        memory_path = tmp_path / f"{name}.memory.json"
        run_memory_profile(
            binary=PROFILER,
            root=tmp_path,
            manifest=manifest,
            plan=deployment,
            stage_index=i,
            workload=WorkloadProfile(
                workload_id="cpu-rehearsal",
                prompt_tokens=5,
                output_tokens=16,
                total_cached_tokens=21,
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
        q = workers.controls[i].GetQualificationState(common_pb2.Empty(), timeout=5)
        # A genuine guard record (start plus a normal exit) for each worker.
        guard = tmp_path / f"{name}.guard.jsonl"
        assert (
            resource_guard.main(
                ["--seconds", "5", "--record", str(guard), "--", sys.executable, "-c", "pass"]
            )
            == 0
        )
        settings[name] = dict(
            endpoint=workers.endpoints[name],
            memory_profile=str(memory_path),
            binary_digest=sha256_file(BINARY),
            runtime_driver_version=q.driver_version,
            guard_record=str(guard),
        )
    (tmp_path / "baseline.json").write_text(json.dumps({"scope": "cpu-rehearsal-baseline"}))
    for name, value in (
        ("manifest", manifest.model_dump(mode="json")),
        ("plan", deployment.model_dump(mode="json")),
        ("reference", reference),
        ("workers", settings),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    args = ["--cpu-rehearsal"]
    for name in ("manifest", "plan", "reference", "workers"):
        args.extend(["--" + name, str(tmp_path / f"{name}.json")])
    args.extend(["--baseline-report", str(tmp_path / "baseline.json")])
    args.extend(["--output", str(tmp_path / "evidence.json")])
    args.extend(["--source-digest", "cpu-rehearsal", "--transport", "loopback"])
    args.extend(["--guard-description", "5-second guard around a no-op; workers unguarded"])
    # Sixteen tokens keep each tiny request alive long enough for the 5 ms sampler to
    # catch both reservations and for the survivor to outlive the cancelled victim.
    args.extend(["--cycles", "2", "--pairs", "2", "--idle-seconds", "0", "--timeout", "10"])
    args.extend(["--sample-interval", "0.005", "--cancel-after", "8"])
    return args


def test_cpu_rehearsal_produces_accepted_concurrency_evidence(tmp_path: Path) -> None:
    write_model(tmp_path)
    with Workers(tmp_path, limit=CAP, extra_args=(OBSERVING, OBSERVING)) as workers:
        args = prepare(tmp_path, workers)
        concurrent_soak.main(args)
        evidence = read_evidence(tmp_path / "evidence.json")
        assert evidence.results.acceptance == "accepted", evidence.results.acceptance_reasons
        assert evidence.results.failure_reason is None
        kinds = [(r.cycle, r.kind) for r in evidence.results.rounds]
        assert kinds == [(c, k) for c in range(2) for k in ("serial", "paired", "paired", "cancel")]
        assert all(c.exact for c in evidence.results.comparisons)
        victims = [c for c in evidence.results.comparisons if c.role == "cancel-victim"]
        assert len(victims) == 2 and all(len(v.expected_ids) == 8 for v in victims)
        overlapping = {(o.worker_id, o.request_ids) for o in evidence.results.overlap}
        for round_ in evidence.results.rounds:
            if round_.kind != "serial":
                for worker in ("cpu-a", "cpu-b"):
                    assert (worker, round_.request_ids) in overlapping
        for worker in evidence.observations.workers:
            assert max(s.peak_concurrent_requests for s in worker.snapshots) == 2
            assert len({s.process_id for s in worker.snapshots if s.process_id}) == 1
        assert all(
            p.policy == "cpu-rss-v1" and p.status == "safe" for p in evidence.results.physical
        )
        assert all(g.outcome == "normal" for g in evidence.results.guards)
        assert all(c.verified for c in evidence.results.cleanup)
        assert all(c.request_observations for c in evidence.configuration.workers)
        rows = [
            json.loads(r)
            for r in (tmp_path / "evidence.observations.jsonl").read_text().splitlines()
        ]
        assert any(
            r.get("phase", "").startswith("paired-0-0") and len(r.get("requests", [])) == 2
            for r in rows
        )
        wait_clean(workers, loaded=False)
        # The record is immutable and a second run cannot replace it.
        with pytest.raises(FileExistsError):
            concurrent_soak.main(args)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--max-active-requests", "1", "--request-observations", "on"), "fewer than two"),
        (
            (
                "--max-active-requests",
                "2",
            ),
            "start the worker with --request-observations on",
        ),
    ],
)
def test_harness_refuses_workers_that_cannot_prove_overlap(
    tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    write_model(tmp_path)
    with Workers(tmp_path, limit=CAP, extra_args=(arguments, arguments)) as workers:
        args = prepare(tmp_path, workers)
        with pytest.raises(ValueError, match=message):
            concurrent_soak.main(args)
        assert not (tmp_path / "evidence.json").exists()
        wait_clean(workers, loaded=False)
