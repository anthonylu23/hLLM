"""Mixed CPU/accelerator concurrent soak, opt-in through CTest."""

import asyncio
import os
from pathlib import Path

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.serving.runtime import ServingRuntime

from tests.process_helpers import BINARY, Workers, plan, tokens, wait_clean, write_model

ACCELERATOR = os.environ.get("HLLM_ACCELERATOR_WORKER")
pytestmark = pytest.mark.skipif(not ACCELERATOR, reason="accelerator worker not configured")


@pytest.mark.parametrize("reverse", [False, True])
def test_concurrent_mixed_soak_and_cancellation(tmp_path: Path, reverse: bool) -> None:
    assert ACCELERATOR is not None
    manifest = write_model(tmp_path)
    with Workers(tmp_path) as reference:
        with DeploymentSession(manifest, plan(manifest), reference.endpoints) as baseline:
            expected = tokens(baseline, count=96)
    arguments = (
        "--max-active-requests",
        "4",
        "--max-cached-tokens",
        "512",
        "--max-decode-batch",
        "4",
    )
    accelerator_arguments = arguments
    if os.environ["HLLM_ACCELERATOR_BACKEND"] == "cuda":
        accelerator_arguments += ("--device-memory-limit-bytes", str(128 * 1024**2))
    with Workers(
        tmp_path,
        limit=128 * 1024**2,
        binaries=(BINARY, Path(ACCELERATOR)),
        extra_args=(arguments, accelerator_arguments),
    ) as workers:

        async def run() -> None:
            runtime = ServingRuntime(
                DeploymentSession(manifest, plan(manifest, reverse=reverse), workers.endpoints), 4
            )
            await runtime.start()
            peak_active = 0
            finished = False

            async def observe() -> None:
                from hllm_control.proto import common_pb2

                nonlocal peak_active
                while not finished:
                    reports = await asyncio.gather(
                        *[
                            c.GetMemoryReport(common_pb2.Empty(), timeout=2)
                            for c in runtime.controls
                        ]
                    )
                    peak_active = max(peak_active, *(r.active_requests for r in reports))
                    await asyncio.sleep(0.005)

            observer = asyncio.create_task(observe())
            try:
                for cycle in range(5):

                    async def request(index: int, run: int = cycle) -> list[int]:
                        stream = runtime.generate(f"soak-{run}-{index}", [1, 4, 2], 96, [], 30)
                        result = []
                        try:
                            async for token in stream:
                                result.append(token)
                                if index == 0 and len(result) == 2:
                                    break
                        finally:
                            await stream.aclose()
                        return result

                    results = await asyncio.gather(*(request(i) for i in range(4)))
                    assert results[0] == expected[:2]
                    assert results[1:] == [expected] * 3
                    wait_clean(workers)
                    assert not runtime.cleanup_failed
                assert peak_active >= 2
            finally:
                finished = True
                await observer
                await runtime.close()

        asyncio.run(run())
        wait_clean(workers, loaded=False)


@pytest.mark.parametrize("reverse", [False, True])
def test_sampled_requests_reproduce_under_concurrency(tmp_path: Path, reverse: bool) -> None:
    # Exact equality is qualified for this tiny fixture, not arbitrary checkpoints:
    # batch-dependent rounding can change seeded choices (see the M6 sweep report).
    from hllm_control.proto import execution_pb2

    assert ACCELERATOR is not None
    manifest = write_model(tmp_path)
    arguments = (
        "--max-active-requests",
        "4",
        "--max-cached-tokens",
        "512",
        "--max-decode-batch",
        "4",
    )
    accelerator_arguments = arguments
    if os.environ["HLLM_ACCELERATOR_BACKEND"] == "cuda":
        accelerator_arguments += ("--device-memory-limit-bytes", str(128 * 1024**2))
    with Workers(
        tmp_path,
        limit=128 * 1024**2,
        binaries=(BINARY, Path(ACCELERATOR)),
        extra_args=(arguments, accelerator_arguments),
    ) as workers:

        async def run() -> None:
            runtime = ServingRuntime(
                DeploymentSession(manifest, plan(manifest, reverse=reverse), workers.endpoints), 4
            )
            await runtime.start()
            try:

                async def request(identifier: str, seed: int) -> list[int]:
                    return [
                        token
                        async for token in runtime.generate(
                            identifier,
                            [1, 4, 2],
                            32,
                            [],
                            30,
                            execution_pb2.SamplingOptions(
                                temperature=1, top_p=0.95, top_k=8, seed=seed
                            ),
                        )
                    ]

                first = await request("reference-42", 42)
                second = await request("reference-123", 123)
                assert first != second
                results = await asyncio.gather(
                    *(request(f"sampled-{i}", seed) for i, seed in enumerate([42, 123, 42, 123]))
                )
                assert results == [first, second, first, second]
                wait_clean(workers)
            finally:
                await runtime.close()

        asyncio.run(run())
        wait_clean(workers, loaded=False)
