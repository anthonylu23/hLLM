"""Opt-in real HTTP concurrency qualification against explicitly selected workers.

Run memory_watch.py beside each worker for physical measurements. Native allocator
and reservation snapshots in this report are separate from those process samples.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import math
import socket
import time
from collections.abc import AsyncGenerator, Sequence
from itertools import pairwise
from pathlib import Path

import httpx
import uvicorn
from google.protobuf.json_format import MessageToDict
from hllm_control.controller import DeploymentSession
from hllm_control.models import DType, ModelManifest
from hllm_control.proto import common_pb2, execution_pb2
from hllm_control.serving.app import create_app
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.serving.tokenizer import TextTokenizer

from scripts.validation.checkpoint_run import make_plan


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


async def qualify(
    manifest: ModelManifest,
    tokenizer: TextTokenizer,
    endpoints: dict[str, str],
    names: list[str],
    prompts: list[str],
    *,
    rounds: int,
    output_tokens: int,
    levels: list[int],
    timeout: float,
    output: Path,
    split: int,
    dtype: DType = DType.F16,
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
    plan = make_plan(manifest, names, dtype, split)
    report = {
        "completed": False,
        "started_unix_time": time.time(),
        "manifest_digest": manifest.manifest_digest,
        "plan": plan.model_dump(mode="json"),
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
            level = {"concurrency": concurrency, "requests": [], "rounds": [], "observations": []}
            report["levels"].append(level)
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
                observed = max(
                    int(w["memory"].get("activeRequests", 0))
                    for sample in level["observations"]
                    for w in sample["workers"]
                )
                level["peak_active_requests"] = observed
                if observed < concurrency:
                    raise RuntimeError("requested native concurrency was not observed")
            finally:
                if observing is not None:
                    observing.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await observing
                server.should_exit = True
                await serving
                listener.close()
            # The server lifespan closes its channels; use fresh channels for unload checks.
            from scripts.validation.checkpoint_run import snapshots

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("tokenizer_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", nargs=2, choices=("mlx", "cuda"), required=True)
    parser.add_argument("--mlx-endpoint", required=True)
    parser.add_argument("--cuda-endpoint", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--split", type=int, default=14)
    parser.add_argument("--prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--sampling-json", type=Path)
    parser.add_argument("--minimum-duration-seconds", type=float, default=0)
    args = parser.parse_args()
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
    tokenizer = TextTokenizer(args.tokenizer_root, manifest.config.vocabulary_size)
    prompts = ["The capital of France is", "Explain how a computer processes a request. " * 16]
    result = asyncio.run(
        qualify(
            manifest,
            tokenizer,
            {"mlx": args.mlx_endpoint, "cuda": args.cuda_endpoint},
            args.workers,
            prompts,
            rounds=args.rounds,
            output_tokens=args.output_tokens,
            levels=args.levels,
            timeout=args.timeout,
            output=args.output,
            split=args.split,
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
