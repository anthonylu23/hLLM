"""Chunk acknowledgments preserve token positions, sampling state and cleanup."""

import asyncio
from pathlib import Path

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.proto import execution_pb2
from hllm_control.serving.runtime import ServingRuntime

from tests.process_helpers import Workers, plan, wait_clean, write_model


@pytest.mark.parametrize("split", [None, 1])
@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_chunked_prefill_matches_greedy_and_seeded_reference(
    tmp_path: Path,
    split: int | None,
    chunk: int,
) -> None:
    manifest = write_model(tmp_path)
    arguments = ("--max-active-requests", "4")
    with Workers(tmp_path, extra_args=(arguments, arguments)) as workers:

        async def run() -> None:
            expected = []
            prompt = [1, 4, 2, 5, 3, 1, 4]
            for chunk_size in [0, chunk]:
                runtime = ServingRuntime(
                    DeploymentSession(manifest, plan(manifest, split=split), workers.endpoints),
                    4,
                    prefill_chunk_tokens=chunk_size,
                )
                await runtime.start()
                try:

                    async def request(
                        identifier: str, sampled: bool, runtime: ServingRuntime = runtime
                    ) -> list[int]:
                        sampling = (
                            execution_pb2.SamplingOptions(
                                temperature=1,
                                top_p=0.95,
                                top_k=8,
                                seed=42,
                            )
                            if sampled
                            else None
                        )
                        return [
                            token
                            async for token in runtime.generate(
                                identifier,
                                prompt,
                                32,
                                [],
                                30,
                                sampling,
                            )
                        ]

                    actual = await asyncio.gather(request("greedy", False), request("seeded", True))
                    if not chunk_size:
                        expected = actual
                    else:
                        assert actual == expected
                    wait_clean(workers)
                finally:
                    await runtime.close()

        asyncio.run(run())
        wait_clean(workers, loaded=False)
