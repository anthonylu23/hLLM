"""Two-worker CPU rehearsal of the concurrency-two harness: procedure evidence only."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import Backend, DType, WorkloadProfile
from hllm_control.prepare.manifest import HashMode, prepare_model
from hllm_control.profiling.models import MemoryAmounts
from hllm_control.profiling.runner import run_memory_profile
from hllm_control.proto import common_pb2
from hllm_control.qualification.concurrent import read_evidence
from hllm_control.serialization import sha256_file

from scripts.validation import concurrent_soak, reload_soak
from scripts.validation.worker_pid import worker_pid
from tests.process_helpers import BINARY, ROOT, Workers, plan, wait_clean, write_model

PROFILER = Path(os.environ.get("HLLM_MEMORY_PROFILER", BINARY.with_name("hllm-profile-memory-cpu")))
CAP = 32 * 1024**2
PROMPT = [1, 4, 2, 8, 3]
# The cancellation round is a race between the controller's cancel RPCs and the
# victim's natural completion on every stage: a tiny worker decodes a token in about a
# millisecond, so the reference must be long relative to --cancel-after for the cancel
# to land first even on a loaded CI runner. 128 tokens with a cancel after three leaves
# 125 tokens of headroom; the sixteen-token reference used before left eight.
TOKENS = 128
OBSERVING = (
    "--max-active-requests",
    "2",
    "--max-cached-tokens",
    str(2 * (len(PROMPT) + TOKENS)),
    "--request-observations",
    "on",
)


def guards(tmp_path: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Each worker runs under its own resource guard, as in the two-host experiment."""
    return tuple(
        (
            sys.executable,
            str(ROOT / "scripts/validation/resource_guard.py"),
            "--seconds",
            "240",
            "--record",
            str(tmp_path / f"{name}.guard.jsonl"),
            "--",
        )
        for name in ("cpu-a", "cpu-b")
    )  # type: ignore[return-value]


def prepare(tmp_path: Path, workers: Workers, chunk: int = 0) -> list[str]:
    manifest = prepare_model(tmp_path, hash_mode=HashMode.FULL)
    deployment = plan(manifest)
    ids = PROMPT
    with DeploymentSession(manifest, plan(manifest, split=None), workers.endpoints) as session:
        expected = [
            e.token.token_id
            for e in session.generate(ids, maximum_new_tokens=TOKENS, stop_token_ids=[])
            if e.HasField("token")
        ]
    assert len(expected) == TOKENS
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
                prompt_tokens=len(PROMPT),
                output_tokens=TOKENS,
                total_cached_tokens=len(PROMPT) + TOKENS,
                kv_dtype=DType.F32,
                prefill_chunk_tokens=chunk,
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
        settings[name] = dict(
            endpoint=workers.endpoints[name],
            memory_profile=str(memory_path),
            binary_digest=sha256_file(BINARY),
            runtime_driver_version=q.driver_version,
            guard_record=str(tmp_path / f"{name}.guard.jsonl"),
        )
    for name, value in (
        ("manifest", manifest.model_dump(mode="json")),
        ("plan", deployment.model_dump(mode="json")),
        ("reference", reference),
        ("workers", settings),
        # The reload soak's worker format has no guard or sampler paths.
        (
            "baseline-workers",
            {k: {f: v[f] for f in v if f != "guard_record"} for k, v in settings.items()},
        ),
    ):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    # The passing single-request baseline is a real reload-soak report on these
    # binaries, this plan and this reference; the harness binds all three.
    reload_soak.main(
        [
            "--cpu-rehearsal",
            *[
                a
                for n in ("manifest", "plan", "reference")
                for a in ("--" + n, str(tmp_path / f"{n}.json"))
            ],
            "--workers",
            str(tmp_path / "baseline-workers.json"),
            "--output",
            str(tmp_path / "baseline.json"),
            "--cycles",
            "1",
            "--requests",
            "1",
            "--idle-seconds",
            "0",
            "--timeout",
            "10",
        ]
    )
    wait_clean(workers, loaded=False)
    args = ["--cpu-rehearsal"]
    for name in ("manifest", "plan", "reference", "workers"):
        args.extend(["--" + name, str(tmp_path / f"{name}.json")])
    args.extend(["--baseline-report", str(tmp_path / "baseline.json")])
    args.extend(["--output", str(tmp_path / "evidence.json")])
    args.extend(["--source-digest", "cpu-rehearsal", "--transport", "loopback"])
    args.extend(["--guard-description", "resource_guard.py around each worker, 240 s"])
    # Each 128-token request lives for well over a hundred milliseconds, so the 20 ms
    # sampler catches both reservations and the survivor outlives the cancelled victim.
    args.extend(["--cycles", "2", "--pairs", "2", "--idle-seconds", "0", "--timeout", "10"])
    args.extend(["--sample-interval", "0.02", "--cancel-after", "3"])
    return args


# Chunk 2 sends the five-token prompt in three prefill passes; profiles, baseline and
# requests all use it.
@pytest.mark.parametrize("chunk", [0, 2])
def test_cpu_rehearsal_produces_accepted_concurrency_evidence(tmp_path: Path, chunk: int) -> None:
    write_model(tmp_path)
    with Workers(
        tmp_path, limit=CAP, extra_args=(OBSERVING, OBSERVING), wrappers=guards(tmp_path)
    ) as workers:
        args = prepare(tmp_path, workers, chunk)
        assert json.loads((tmp_path / "baseline.json").read_text())["prefill_chunk_tokens"] == chunk
        concurrent_soak.main(args)
        evidence = read_evidence(tmp_path / "evidence.json")
        assert evidence.results.acceptance == "accepted", evidence.results.acceptance_reasons
        assert all(c.prefill_chunk_tokens == chunk for c in evidence.configuration.workers)
        assert evidence.results.failure_reason is None
        assert evidence.preconditions.baseline.scope == "cpu-rehearsal"
        assert evidence.preconditions.baseline.worker_binary_digests == tuple(
            (w.worker_id, w.worker_binary_digest) for w in evidence.identity.workers
        )
        assert [(g.worker_id, g.cycle) for g in evidence.preconditions.gates] == [
            (w, c) for c in range(2) for w in ("cpu-a", "cpu-b")
        ]
        kinds = [(r.cycle, r.kind) for r in evidence.results.rounds]
        assert kinds == [(c, k) for c in range(2) for k in ("serial", "paired", "paired", "cancel")]
        assert all(c.exact for c in evidence.results.comparisons)
        victims = [c for c in evidence.results.comparisons if c.role == "cancel-victim"]
        assert len(victims) == 2 and all(len(v.expected_ids) == 3 for v in victims)
        survivors = [c for c in evidence.results.comparisons if c.role == "cancel-survivor"]
        assert all(len(s.expected_ids) == TOKENS for s in survivors)
        overlapping = {(o.worker_id, o.request_ids) for o in evidence.results.overlap}
        for round_ in evidence.results.rounds:
            if round_.kind != "serial":
                for worker in ("cpu-a", "cpu-b"):
                    assert (worker, round_.request_ids) in overlapping
        for worker in evidence.observations.workers:
            assert max(s.peak_concurrent_requests for s in worker.snapshots) == 2
            assert len({s.process_id for s in worker.snapshots if s.process_id}) == 1
        assert [(p.worker_id, p.cycle) for p in evidence.results.physical] == [
            (w, c) for w in ("cpu-a", "cpu-b") for c in range(2)
        ]
        assert all(
            p.policy == "cpu-rss-v1" and p.status == "safe" for p in evidence.results.physical
        )
        # The guards wrapped the worker processes themselves and covered the whole run.
        for g, w in zip(evidence.results.guards, evidence.identity.workers, strict=True):
            assert g.outcome == "normal" and not g.reasons, g
            assert g.process_id == w.process_id and g.exited_unix_time is None
            # The still-growing guard record was read once; its snapshot reproduces the hash.
            snapshot = tmp_path / f"evidence.guard-{w.worker_id}.jsonl"
            assert sha256_file(snapshot) == g.record_sha256
            assert (
                (tmp_path / f"{w.worker_id}.guard.jsonl")
                .read_bytes()
                .startswith(snapshot.read_bytes())
            )
            # Launchers bind samplers to the PID the worker reports, not a command line.
            assert worker_pid(w.endpoint, expect_worker_id=w.worker_id) == w.process_id
        with pytest.raises(ValueError, match="is worker 'cpu-a', not 'cpu-b'"):
            worker_pid(workers.endpoints["cpu-a"], expect_worker_id="cpu-b")
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
        # The workers were reached, so the refusal leaves a note naming its stage.
        note = json.loads((tmp_path / "evidence.failure.json").read_text())
        assert note["stage"] == "worker identity" and message in note["run_failure"]
        wait_clean(workers, loaded=False)


def test_harness_refuses_workers_with_different_allowances(tmp_path: Path) -> None:
    write_model(tmp_path)
    with Workers(tmp_path, limit=CAP, extra_args=(OBSERVING, OBSERVING)) as workers:
        args = prepare(tmp_path, workers)
        specifications = json.loads((tmp_path / "workers.json").read_text())
        specifications["cpu-b"]["host_headroom_bytes"] = 2 * 1024**3
        (tmp_path / "workers.json").write_text(json.dumps(specifications))
        with pytest.raises(ValueError, match="must share extra overhead"):
            concurrent_soak.main(args)
        assert not (tmp_path / "evidence.json").exists()
        assert (tmp_path / "evidence.failure.json").is_file()
        wait_clean(workers, loaded=False)


def test_harness_refuses_an_unbound_baseline_before_loading(tmp_path: Path, monkeypatch) -> None:
    write_model(tmp_path)
    with Workers(tmp_path, limit=CAP, extra_args=(OBSERVING, OBSERVING)) as workers:
        args = prepare(tmp_path, workers)
        enter = Mock(side_effect=AssertionError("an unbound baseline must not start a deployment"))
        monkeypatch.setattr(DeploymentSession, "__enter__", enter)
        baseline = tmp_path / "baseline.json"
        report = json.loads(baseline.read_text())
        for mutate, message in (
            (lambda: baseline.write_text("not a JSON report; no passing baseline"), "not a usable"),
            (
                lambda: baseline.write_text(json.dumps({**report, "completed": False})),
                "exact outputs",
            ),
            (
                lambda: baseline.write_text(
                    json.dumps({**report, "plan": {**report["plan"], "plan_digest": "other"}})
                ),
                "different plan",
            ),
            (
                lambda: baseline.write_text(json.dumps({**report, "prefill_chunk_tokens": 2})),
                "with prefill chunk 2",
            ),
        ):
            mutate()
            with pytest.raises(ValueError, match=message):
                concurrent_soak.main(args)
            assert not (tmp_path / "evidence.json").exists()
        enter.assert_not_called()
        wait_clean(workers, loaded=False)


def test_failure_at_the_first_gate_preserves_partial_evidence(tmp_path: Path, monkeypatch) -> None:
    write_model(tmp_path)
    with Workers(
        tmp_path, limit=CAP, extra_args=(OBSERVING, OBSERVING), wrappers=guards(tmp_path)
    ) as workers:
        args = prepare(tmp_path, workers)
        monkeypatch.setattr(
            concurrent_soak,
            "fresh_fit",
            Mock(side_effect=RuntimeError("simulated worker lost at gate")),
        )
        with pytest.raises(RuntimeError, match="simulated worker lost at gate"):
            concurrent_soak.main(args)
        assert not (tmp_path / "evidence.failure.json").exists()
        evidence = read_evidence(tmp_path / "evidence.json")
        assert evidence.results.acceptance == "failed"
        assert evidence.results.failure_reason == "RuntimeError: simulated worker lost at gate"
        assert not evidence.preconditions.gates and not evidence.results.physical
        assert not evidence.results.rounds
        assert any("no fit gate recorded" in r for r in evidence.results.acceptance_reasons)
        wait_clean(workers, loaded=False)
