"""Opt-in real HTTP concurrency qualification against explicitly selected workers.

The run binds to a supplied deployment plan and the same worker specifications as
`reload_soak.py` and `concurrent_soak.py`. Before every level's load it checks that
each worker admits the level (active requests and aggregate cached tokens) and applies
the fresh single-request physical-fit gate to each worker's same-day profile; any
refusal stops the run before loading. A level passes only if every stage, on its own,
was observed holding that many active requests.

Run memory_watch.py beside each worker for physical measurements. Native allocator
and reservation snapshots in this report are separate from those process samples.
Batch-dependent seeded sampling is a known limit; compare sampled runs only within
one level.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import socket
import sys
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from itertools import pairwise
from pathlib import Path

import grpc
import httpx
import uvicorn
from google.protobuf.json_format import MessageToDict
from hllm_control.controller import DeploymentSession
from hllm_control.models import DeploymentPlan, ModelManifest
from hllm_control.profiling.models import MemoryMeasurement, ProfileArtifact
from hllm_control.proto import common_pb2, control_pb2_grpc, execution_pb2
from hllm_control.serving.app import create_app
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.serving.tokenizer import TextTokenizer

if __package__ in (None, ""):  # `python scripts/validation/serving_soak.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.validation.checkpoint_run import snapshots
from scripts.validation.reload_soak import Worker, fresh_fit

# Called with a level's concurrency before that level loads; returns what it checked
# and raises to refuse the level.
Gate = Callable[[int], dict]


class RecordingRuntime(ServingRuntime):
    """Observe native events without changing their scheduling or pacing."""

    def __init__(
        self, session: DeploymentSession, concurrency: int, *, prefill_chunk_tokens: int = 0
    ) -> None:
        super().__init__(session, concurrency, prefill_chunk_tokens=prefill_chunk_tokens)
        self.records: dict[str, dict] = {}

    async def generate(
        self,
        identifier: str,
        tokens: Sequence[int],
        maximum: int,
        stops: Sequence[int],
        timeout: float,
        sampling: execution_pb2.SamplingOptions | None = None,
    ) -> AsyncGenerator[int]:
        record = {"started": time.monotonic(), "tokens": [], "token_times": [], "probabilities": []}
        self.records[identifier] = record
        iterator = super().generate(identifier, tokens, maximum, stops, timeout, sampling)
        try:
            async for token in iterator:
                record["tokens"].append(token)
                record["token_times"].append(time.monotonic())
                if identifier in self.latest_tokens:
                    record["probabilities"].append(MessageToDict(self.latest_tokens[identifier]))
                yield token
        finally:
            await iterator.aclose()
            record["retired"] = time.monotonic()


async def snapshot(runtime: ServingRuntime) -> list[dict]:
    result = []
    for control in runtime.controls:
        memory, metrics = await asyncio.gather(
            control.GetMemoryReport(common_pb2.Empty(), timeout=5),
            control.GetMetrics(common_pb2.Empty(), timeout=5),
        )
        result.append({"memory": MessageToDict(memory), "metrics": MessageToDict(metrics)})
    return result


async def clean(runtime: ServingRuntime) -> list[dict]:
    until = time.monotonic() + 10
    while True:
        reports = await snapshot(runtime)
        if all(
            not any(
                int(r["memory"].get(k, 0))
                for k in ("activeRequests", "reservedCacheBytes", "reservedWorkspaceBytes")
            )
            for r in reports
        ):
            if runtime.cleanup_failed:
                raise RuntimeError("HTTP runtime reported unconfirmed cleanup")
            return reports
        if time.monotonic() >= until:
            raise RuntimeError("sequence reservations did not retire")
        await asyncio.sleep(0.05)


def validate_serving_profile(
    a: ProfileArtifact,
    manifest: ModelManifest,
    plan: DeploymentPlan,
    worker_id: str,
    reservation_tokens: int,
) -> None:
    """A single-request profile may gate an HTTP level only for this stage and precision.

    Every HTTP request must fit the profiled per-request reservation. The profile is
    the concurrency-one baseline; the serving peaks of higher levels are what the
    report records.
    """
    k = a.key
    if (
        k.manifest_digest != manifest.manifest_digest
        or k.assignment != next(s for s in plan.stages if s.worker_id == worker_id)
        or k.execution_dtype != plan.execution_dtype
        or k.weight_dtype != plan.weight_dtype
        or k.workload.kv_dtype != plan.execution_dtype
        or k.workload.activation_dtype != plan.activation_dtype
        or k.workload.concurrency != 1
        or k.workload.total_cached_tokens < reservation_tokens
        or not isinstance(a.measurement, MemoryMeasurement)
        or not a.measurement.completed
    ):
        raise ValueError(f"{worker_id}: memory profile does not cover this serving workload")


def worker_gate(
    plan: DeploymentPlan,
    manifest: ModelManifest,
    workers: dict[str, Worker],
    reservation_tokens: int,
) -> Gate:
    """Capacity and fresh physical-fit checks against each worker's own profile."""
    profiles = {
        k: ProfileArtifact.model_validate_json(w.memory_profile.read_text())
        for k, w in workers.items()
    }
    for k, a in profiles.items():
        validate_serving_profile(a, manifest, plan, k, reservation_tokens)

    def gate(concurrency: int) -> dict:
        result: dict = {}
        for k, w in workers.items():
            with grpc.insecure_channel(w.endpoint) as channel:
                control = control_pb2_grpc.WorkerControlStub(channel)
                c = control.GetCapabilities(common_pb2.Empty(), timeout=5).worker
                active = int(c.maximum_active_requests or 1)
                cached = int(c.maximum_cached_tokens)
                if active < concurrency:
                    raise RuntimeError(f"{k}: --max-active-requests {active} < {concurrency}")
                if cached and cached < concurrency * reservation_tokens:
                    raise RuntimeError(
                        f"{k}: --max-cached-tokens {cached} cannot admit {concurrency} "
                        f"reservations of {reservation_tokens} tokens"
                    )
                fit = fresh_fit(profiles[k], w, control)
            result[k] = dict(maximum_active_requests=active, maximum_cached_tokens=cached, **fit)
            if fit["assessment"]["status"] != "safe":
                raise RuntimeError(
                    f"fresh physical-fit gate rejected {k} before level {concurrency}"
                )
        return result

    return gate


async def qualify(
    manifest: ModelManifest,
    tokenizer: TextTokenizer,
    plan: DeploymentPlan,
    endpoints: dict[str, str],
    prompts: list[str],
    *,
    rounds: int,
    output_tokens: int,
    levels: list[int],
    timeout: float,
    output: Path,
    gate: Gate | None,
    prefill_chunk_tokens: int = 0,
    stagger_seconds: float = 0.01,
    observation_interval: float = 0.1,
    sampling: dict | None = None,
    minimum_duration_seconds: float = 0,
) -> dict:
    if (
        rounds < 1
        or output_tokens < 4
        or not levels
        or levels[0] != 1
        or levels != sorted(set(levels))
        or levels[-1] > 64
        or not 0 <= stagger_seconds <= 1
        or not 0.001 <= observation_interval <= 1
        or not 0 <= minimum_duration_seconds <= 7200
    ):
        raise ValueError("invalid qualification workload or observation settings")
    if sampling is not None and (
        not isinstance(sampling, dict)
        or set(sampling) - {"temperature", "top_p", "top_k", "seed", "logprobs"}
    ):
        raise ValueError("sampling options must contain only supported sampling fields")
    names = [s.worker_id for s in plan.stages]
    if plan.manifest_digest != manifest.manifest_digest or set(endpoints) != set(names):
        raise ValueError("plan must match the manifest and name exactly the selected workers")
    report = {
        "completed": False,
        "started_unix_time": time.time(),
        "manifest_digest": manifest.manifest_digest,
        "plan": plan.model_dump(mode="json"),
        "physical_fit_gate": gate is not None,
        "rounds": rounds,
        "output_tokens": output_tokens,
        "prefill_chunk_tokens": prefill_chunk_tokens,
        "stagger_seconds": stagger_seconds,
        "observation_interval": observation_interval,
        "sampling": sampling or {},
        "minimum_duration_seconds_at_max_concurrency": minimum_duration_seconds,
        "prompts": prompts,
        "levels": [],
    }
    reference: dict[str, list[int]] = {}

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch(exist_ok=False)
    try:
        for concurrency in levels:
            level = {"concurrency": concurrency, "requests": [], "rounds": [], "observations": []}
            report["levels"].append(level)
            if gate is not None:
                level["preflight"] = await asyncio.to_thread(gate, concurrency)
                save()
            runtime = RecordingRuntime(
                DeploymentSession(manifest, plan, endpoints),
                concurrency,
                prefill_chunk_tokens=prefill_chunk_tokens,
            )
            app = create_app(
                runtime,
                tokenizer,
                model_name="qualification",
                maximum_queued=concurrency,
                timeout=timeout,
            )
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            observing = None
            try:
                while not server.started:
                    if serving.done():
                        await serving
                        raise RuntimeError("HTTP startup failed")
                    await asyncio.sleep(0.01)
                level["qualification"] = [
                    MessageToDict(await c.GetQualificationState(common_pb2.Empty(), timeout=5))
                    for c in runtime.controls
                ]

                async def observe(level: dict = level, runtime: ServingRuntime = runtime) -> None:
                    while True:
                        level["observations"].append(
                            {"unix_time": time.time(), "workers": await snapshot(runtime)}
                        )
                        await asyncio.sleep(observation_interval)

                observing = asyncio.create_task(observe())
                async with httpx.AsyncClient(
                    base_url=f"http://127.0.0.1:{port}", timeout=timeout + 15
                ) as client:

                    async def request(
                        prompt: str,
                        *,
                        cancel: bool = False,
                        delay: float = 0,
                        level: dict = level,
                        runtime: RecordingRuntime = runtime,
                    ) -> dict:
                        await asyncio.sleep(delay)
                        started = time.monotonic()
                        fragments = []
                        identifier = None
                        done = False
                        probability_tokens = 0
                        async with client.stream(
                            "POST",
                            "/v1/completions",
                            json={
                                "model": "qualification",
                                "prompt": prompt,
                                "max_tokens": output_tokens,
                                "stop_token_ids": [],
                                "stream": True,
                                **(sampling or {}),
                            },
                        ) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line.startswith("data: "):
                                    continue
                                if line == "data: [DONE]":
                                    done = True
                                    break
                                event = json.loads(line[6:])
                                if "error" in event:
                                    raise RuntimeError(event["error"])
                                identifier = event["id"]
                                if sampling is not None and sampling.get("logprobs") is not None:
                                    metadata = event["choices"][0].get("logprobs")
                                    if metadata is not None:
                                        probability_tokens += len(metadata["tokens"])
                                        if any(
                                            not math.isfinite(p) or p > 0
                                            for p in metadata["token_logprobs"]
                                        ):
                                            raise RuntimeError("invalid token log probability")
                                        if any(
                                            len(top) > sampling["logprobs"]
                                            for top in metadata["top_logprobs"]
                                        ):
                                            raise RuntimeError("unbounded probability metadata")
                                if event["choices"][0]["text"]:
                                    fragments.append(time.monotonic())
                                    if cancel and len(fragments) == 2:
                                        break
                        if identifier is None or (not cancel and not done):
                            raise RuntimeError("HTTP response did not complete")
                        record = runtime.records[identifier]
                        times = record["token_times"]
                        result = {
                            "id": identifier,
                            "prompt": prompt,
                            "cancelled": cancel,
                            "elapsed_seconds": time.monotonic() - started,
                            "first_text_seconds": fragments[0] - started if fragments else None,
                            "first_token_seconds": times[0] - started if times else None,
                            "before_native_seconds": record["started"] - started,
                            "inter_token_seconds": [b - a for a, b in pairwise(times)],
                            "token_ids": list(record["tokens"]),
                            "probability_tokens": probability_tokens,
                            "token_probabilities": record["probabilities"],
                        }
                        # Keep the failed trace as well as successful requests so a
                        # numerical or lifecycle mismatch remains diagnosable.
                        level["requests"].append(result)
                        # Cancellation can race with additional native tokens; compare every
                        # observed token to the corresponding single-request prefix.
                        expected = reference.get(prompt)
                        if expected is not None and result["token_ids"] != expected[: len(times)]:
                            mismatch = next(
                                (
                                    i
                                    for i, token in enumerate(result["token_ids"])
                                    if i >= len(expected) or token != expected[i]
                                ),
                                min(len(expected), len(times)),
                            )
                            report["token_mismatch"] = {
                                "request_id": identifier,
                                "index": mismatch,
                                "expected": expected,
                                "actual": result["token_ids"],
                            }
                            raise RuntimeError("concurrent token IDs differ from baseline")
                        if (
                            not cancel
                            and sampling is not None
                            and sampling.get("logprobs") is not None
                            and probability_tokens != output_tokens
                        ):
                            raise RuntimeError("missing HTTP probability metadata")
                        if not cancel and len(times) != output_tokens:
                            raise RuntimeError("wrong output token count")
                        return result

                    for prompt in prompts:
                        result = await request(prompt)
                        reference.setdefault(prompt, result["token_ids"])
                    level["warm_memory"] = await clean(runtime)
                    cycle = 0
                    soak_started = time.monotonic()
                    minimum_duration = minimum_duration_seconds if concurrency == levels[-1] else 0
                    while cycle < rounds or time.monotonic() - soak_started < minimum_duration:
                        # Include one queued request, stagger arrivals, and isolate one cancel.
                        await asyncio.gather(
                            *(
                                request(
                                    prompts[i % len(prompts)],
                                    cancel=i == 0,
                                    delay=i * stagger_seconds,
                                )
                                for i in range(concurrency + 1)
                            )
                        )
                        level["rounds"].append({"index": cycle, "workers": await clean(runtime)})
                        cycle += 1
                        save()
                    level["soak_seconds"] = time.monotonic() - soak_started
                    await request(prompts[0])  # Post-soak recovery.
                    level["final_memory"] = await clean(runtime)
                # Every stage must hold the level on its own; one busy stage does not
                # show that the other stage overlapped requests.
                peaks = {
                    name: max(
                        (
                            int(sample["workers"][index]["memory"].get("activeRequests", 0))
                            for sample in level["observations"]
                        ),
                        default=0,
                    )
                    for index, name in enumerate(names)
                }
                level["peak_active_requests"] = peaks
                short = [name for name, peak in peaks.items() if peak < concurrency]
                if short:
                    raise RuntimeError(
                        f"requested native concurrency was not observed on {', '.join(short)}"
                    )
            finally:
                if observing is not None:
                    observing.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await observing
                server.should_exit = True
                await serving
                listener.close()
            # The server lifespan closes its channels; use fresh channels for unload checks.
            level["after_unload"] = await asyncio.to_thread(snapshots, names, endpoints)
            if any(
                any(
                    int(w["memory"].get(k, 0))
                    for k in (
                        "loadedWeightBytes",
                        "activeRequests",
                        "reservedCacheBytes",
                        "reservedWorkspaceBytes",
                    )
                )
                for w in level["after_unload"]
            ):
                raise RuntimeError("unload left live reservations")
            save()
        report["completed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["finished_unix_time"] = time.time()
        save()
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tokenizer_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan", type=Path, required=True, help="deployment plan to serve")
    parser.add_argument(
        "--workers",
        type=Path,
        required=True,
        help="worker specifications keyed by plan worker ID, as for reload_soak.py",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--sampling-json", type=Path)
    parser.add_argument("--minimum-duration-seconds", type=float, default=0)
    args = parser.parse_args(argv)
    if (
        args.rounds < 1
        or args.output_tokens < 4
        or args.levels != sorted(set(args.levels))
        or (not args.levels or args.levels[0] != 1 or args.levels[-1] > 64)
    ):
        parser.error(
            "positive rounds, >=4 output tokens and increasing levels starting at 1 required"
        )
    manifest = ModelManifest.model_validate_json(args.manifest.read_text())
    plan = DeploymentPlan.model_validate_json(args.plan.read_text())
    workers = {k: Worker.model_validate(v) for k, v in json.loads(args.workers.read_text()).items()}
    if set(workers) != {s.worker_id for s in plan.stages}:
        parser.error("worker names must exactly match plan stages")
    tokenizer = TextTokenizer(args.tokenizer_root, manifest.config.vocabulary_size)
    prompts = ["The capital of France is", "Explain how a computer processes a request. " * 16]
    reservation = max(len(tokenizer.encode(p)) for p in prompts) + args.output_tokens
    result = asyncio.run(
        qualify(
            manifest,
            tokenizer,
            plan,
            {k: w.endpoint for k, w in workers.items()},
            prompts,
            rounds=args.rounds,
            output_tokens=args.output_tokens,
            levels=args.levels,
            timeout=args.timeout,
            output=args.output,
            gate=worker_gate(plan, manifest, workers, reservation),
            prefill_chunk_tokens=args.prefill_chunk_tokens,
            sampling=json.loads(args.sampling_json.read_text()) if args.sampling_json else None,
            minimum_duration_seconds=args.minimum_duration_seconds,
        )
    )
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "report_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
