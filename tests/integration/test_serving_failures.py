"""Real HTTP lifecycle checks with owned workers and controlled client pacing."""

import asyncio
import contextlib
import json
import socket
from pathlib import Path

import httpx
import pytest
import uvicorn
from hllm_control.controller import DeploymentSession
from hllm_control.serving.app import create_app
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.serving.tokenizer import TextTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers

from tests.process_helpers import Workers, plan, wait_clean, write_model


@pytest.mark.parametrize("failure", ["shutdown", "worker_loss", "queued_disconnect"])
def test_active_and_queued_requests_retire(tmp_path: Path, failure: str) -> None:
    manifest = write_model(tmp_path)
    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(11)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    with Workers(tmp_path) as workers:

        async def run() -> None:
            first_token = asyncio.Event()
            proceed = asyncio.Event()

            class GatedRuntime(ServingRuntime):
                async def generate(
                    self, identifier, tokens, maximum, stops, timeout, sampling=None
                ):
                    source = super().generate(identifier, tokens, maximum, stops, timeout, sampling)
                    try:
                        async for token in source:
                            if not first_token.is_set():
                                if failure == "worker_loss":
                                    # Kill while native generation has only just delivered its
                                    # first token, before any deliberate HTTP pacing.
                                    workers.processes[1].kill()
                                first_token.set()
                                await proceed.wait()
                            await asyncio.sleep(0.005)
                            yield token
                    finally:
                        await source.aclose()

            runtime = GatedRuntime(
                DeploymentSession(manifest, plan(manifest), workers.endpoints), 1
            )
            app = create_app(
                runtime,
                TextTokenizer(tmp_path, 11),
                model_name="tiny",
                maximum_queued=1,
                timeout=10,
            )
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="critical", timeout_graceful_shutdown=1)
            )
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            active = queued = gate_waiter = None
            try:
                async with asyncio.timeout(20):
                    while not server.started:
                        if serving.done():
                            await serving
                        await asyncio.sleep(0.01)
                    async with httpx.AsyncClient(
                        base_url=f"http://127.0.0.1:{listener.getsockname()[1]}", timeout=15
                    ) as client:
                        payload = {
                            "model": "tiny",
                            "prompt": "t1 t2 t3",
                            "max_tokens": 400,
                            "stream": True,
                        }

                        async def consume():
                            async with client.stream("POST", "/v1/completions", json=payload) as r:
                                assert r.status_code == 200, await r.aread()
                                return r.status_code, [line async for line in r.aiter_lines()]

                        active = asyncio.create_task(consume())
                        gate_waiter = asyncio.create_task(first_token.wait())
                        done, _ = await asyncio.wait(
                            [active, gate_waiter],
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if active in done:
                            await active
                            raise AssertionError("request ended before test gate")
                        queued = asyncio.create_task(
                            client.post("/v1/completions", json={**payload, "stream": False})
                        )
                        while "hllm_queued_requests 1" not in (await client.get("/metrics")).text:
                            await asyncio.sleep(0.01)
                        if failure == "shutdown":
                            server.should_exit = True
                            await serving
                            await asyncio.gather(active, queued, return_exceptions=True)
                            assert not runtime.calls
                            wait_clean(workers, loaded=False)
                        elif failure == "worker_loss":
                            proceed.set()
                            status, lines = await active
                            assert status == 200
                            events = [
                                json.loads(line[6:])
                                for line in lines
                                if line.startswith("data: ") and line != "data: [DONE]"
                            ]
                            assert any("error" in event for event in events)
                            assert "data: [DONE]" in lines
                            assert (await queued).status_code == 503
                            assert runtime.cleanup_failed
                            assert (await client.get("/health")).status_code == 503
                            assert not runtime.calls
                            # The surviving stage still has to retire its own sequence.
                            from hllm_control.proto import common_pb2

                            report = workers.controls[0].GetMemoryReport(common_pb2.Empty())
                            assert report.active_requests == 0
                            assert report.reserved_cache_bytes == 0
                            assert report.reserved_workspace_bytes == 0
                        else:
                            queued.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await queued
                            while (
                                "hllm_queued_requests 0" not in (await client.get("/metrics")).text
                            ):
                                await asyncio.sleep(0.01)
                            active.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await active
                            while (
                                "hllm_active_requests 0" not in (await client.get("/metrics")).text
                            ):
                                await asyncio.sleep(0.01)
                            wait_clean(workers)
                            assert not runtime.cleanup_failed
                            assert (
                                await client.post(
                                    "/v1/completions",
                                    json={**payload, "max_tokens": 4, "stream": False},
                                )
                            ).status_code == 200
            finally:
                proceed.set()
                for task in (active, queued, gate_waiter):
                    if task is not None and not task.done():
                        task.cancel()
                for task in (active, queued, gate_waiter):
                    if task is not None:
                        await asyncio.gather(task, return_exceptions=True)
                server.should_exit = True
                await asyncio.wait_for(serving, 15)
                listener.close()

        asyncio.run(run())
