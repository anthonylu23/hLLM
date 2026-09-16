"""HTTP -> tokenizer -> two native stages -> text, plus concurrent lifecycle soak."""

import asyncio
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from hllm_control.controller import DeploymentSession
from hllm_control.serving.app import create_app
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.serving.tokenizer import TextTokenizer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from tests.process_helpers import Workers, plan, wait_clean, write_model


def test_http_completion_chat_stream_and_concurrent_soak(tmp_path: Path) -> None:
    manifest = write_model(tmp_path / "model")
    native = Tokenizer(
        models.WordLevel(
            {f"t{i}": i for i in range(manifest.config.vocabulary_size)}, unk_token="t0"
        )
    )
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    native.save(str(tmp_path / "model/tokenizer.json"))
    (tmp_path / "model/tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{% for message in messages %}{{ message.content }} {% endfor %}",
            }
        )
    )
    tokenizer = TextTokenizer(tmp_path / "model", manifest.config.vocabulary_size)
    args = ("--max-active-requests", "4", "--max-cached-tokens", "512")
    with Workers(tmp_path / "model", extra_args=(args, args)) as workers:
        deployment = plan(manifest)
        with DeploymentSession(manifest, deployment, workers.endpoints) as baseline:
            ids = [
                e.token.token_id
                for e in baseline.generate([1, 2, 3], maximum_new_tokens=16, stop_token_ids=[])
                if e.HasField("token")
            ]
        expected = native.decode(ids)
        session = DeploymentSession(manifest, deployment, workers.endpoints)
        app = create_app(ServingRuntime(session, 4), tokenizer, model_name="tiny", timeout=10)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
            assert client.get("/v1/deployments").json()["data"][0]["maximum_active_requests"] == 4
            payload = {
                "model": "tiny",
                "prompt": "t1 t2 t3",
                "max_tokens": 16,
                "stop_token_ids": [],
            }
            result = client.post("/v1/completions", json=payload)
            assert result.status_code == 200, result.text
            assert result.json()["choices"][0]["text"] == expected
            assert result.json()["usage"] == {
                "prompt_tokens": 3,
                "completion_tokens": 16,
                "total_tokens": 19,
            }
            chat = client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "t1 t2 t3"}],
                    "max_tokens": 16,
                    "stop_token_ids": [],
                },
            )
            assert chat.status_code == 200, chat.text
            assert chat.json()["choices"][0]["message"]["content"] == expected
            stream = client.post(
                "/v1/completions",
                json={**payload, "stream": True, "stream_options": {"include_usage": True}},
            )
            assert stream.status_code == 200, stream.text
            lines = [line[6:] for line in stream.text.splitlines() if line.startswith("data: ")]
            assert lines[-1] == "[DONE]"
            chunks = [json.loads(line) for line in lines[:-1]]
            assert "".join(c["choices"][0]["text"] for c in chunks if c["choices"]) == expected
            assert chunks[-1]["usage"]["completion_tokens"] == 16
            assert (
                client.post("/v1/completions", json={**payload, "temperature": -0.8}).status_code
                == 400
            )
            assert (
                client.post("/v1/completions", json={**payload, "model": "missing"}).status_code
                == 404
            )
            assert (
                client.post("/v1/completions", json={**payload, "max_tokens": 512}).status_code
                == 400
            )
            for _round in range(5):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(
                        pool.map(lambda _: client.post("/v1/completions", json=payload), range(8))
                    )
                for response in results:
                    assert response.status_code == 200, response.text
                    assert response.json()["choices"][0]["text"] == expected
                wait_clean(workers)
            metrics = client.get("/metrics").text
            assert "hllm_completed 43" in metrics
            assert "hllm_active_requests 0" in metrics
            sampled = {**payload, "temperature": 0.8, "top_p": 0.9, "top_k": 5, "seed": 42}
            probabilities = client.post("/v1/completions", json={**payload, "logprobs": 3})
            assert probabilities.status_code == 200, probabilities.text
            choice = probabilities.json()["choices"][0]
            assert choice["text"] == expected
            assert len(choice["logprobs"]["tokens"]) == 16
            assert all(value <= 0 for value in choice["logprobs"]["token_logprobs"])
            assert all(len(top) <= 3 for top in choice["logprobs"]["top_logprobs"])
            chat_probs = client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny",
                    "messages": [{"role": "user", "content": "t1 t2 t3"}],
                    "max_tokens": 4,
                    "stream": True,
                    "logprobs": True,
                    "top_logprobs": 2,
                },
            )
            assert chat_probs.status_code == 200, chat_probs.text
            chunks = [
                json.loads(line[6:])
                for line in chat_probs.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            content = [
                entry
                for event in chunks
                for entry in (event["choices"][0].get("logprobs") or {}).get("content", [])
            ]
            assert len(content) == 4
            assert all(len(entry["top_logprobs"]) == 2 for entry in content)
            expected_sample = client.post("/v1/completions", json=sampled).json()
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(
                    pool.map(lambda _: client.post("/v1/completions", json=sampled), range(8))
                )
            assert all(response.status_code == 200 for response in responses)
            assert all(
                response.json()["choices"] == expected_sample["choices"] for response in responses
            )
            assert (
                client.post(
                    "/v1/completions",
                    json={
                        **payload,
                        "temperature": 1,
                        "top_k": 1,
                    },
                ).json()["choices"][0]["text"]
                == expected
            )
        wait_clean(workers, loaded=False)


def test_module_cli_registers_serve() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "hllm_control.cli", "serve", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "tokenizer-root" in result.stdout


def test_native_concurrency_cancellation_and_recovery(tmp_path: Path) -> None:
    manifest = write_model(tmp_path / "model")
    args = ("--max-active-requests", "4")
    with Workers(tmp_path / "model", extra_args=(args, args)) as workers:

        async def run() -> None:
            runtime = ServingRuntime(
                DeploymentSession(manifest, plan(manifest), workers.endpoints), 4
            )
            await runtime.start()
            try:
                baseline = [t async for t in runtime.generate("baseline", [1, 2, 3], 128, [], 10)]

                async def request(index: int) -> list[int]:
                    stream = runtime.generate(f"request-{index}", [1, 2, 3], 128, [], 10)
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
                assert results[0] == baseline[:2]
                assert all(r == baseline for r in results[1:])
                assert not runtime.cleanup_failed
                wait_clean(workers)
                assert [
                    t async for t in runtime.generate("recovery", [1, 2, 3], 128, [], 10)
                ] == baseline
            finally:
                await runtime.close()

        asyncio.run(run())
        wait_clean(workers, loaded=False)


def test_socket_disconnect_overload_and_recovery(tmp_path: Path) -> None:
    import socket
    import threading
    import time
    from collections.abc import AsyncGenerator, Sequence

    import httpx
    import uvicorn

    manifest = write_model(tmp_path / "model")
    native = Tokenizer(models.WordLevel({f"t{i}": i for i in range(11)}, unk_token="t0"))
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    native.save(str(tmp_path / "model/tokenizer.json"))

    from hllm_control.proto import execution_pb2

    class PacedRuntime(ServingRuntime):
        # Deliberately keep the HTTP request open while its native stream is live.
        # This tests HTTP admission/disconnect behavior without timing assertions
        # about how fast the tiny fixture executes on a particular host.
        async def generate(
            self,
            identifier: str,
            tokens: Sequence[int],
            maximum: int,
            stops: Sequence[int],
            timeout: float,
            sampling: execution_pb2.SamplingOptions | None = None,
        ) -> AsyncGenerator[int]:
            source = super().generate(identifier, tokens, maximum, stops, timeout, sampling)
            try:
                async for token in source:
                    await asyncio.sleep(0.02)
                    yield token
            finally:
                await source.aclose()

    with Workers(tmp_path / "model") as workers:
        runtime = PacedRuntime(DeploymentSession(manifest, plan(manifest), workers.endpoints), 1)
        app = create_app(
            runtime,
            TextTokenizer(tmp_path / "model", 11),
            model_name="tiny",
            maximum_queued=0,
            timeout=2,
        )
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            url = f"http://127.0.0.1:{sock.getsockname()[1]}"
            server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
            thread.start()
            try:
                until = time.monotonic() + 10
                while not server.started:
                    assert thread.is_alive() and time.monotonic() < until
                    time.sleep(0.01)
                payload = {"model": "tiny", "prompt": "t1 t2 t3", "max_tokens": 128, "stream": True}
                with httpx.Client(base_url=url, timeout=10) as client:
                    with client.stream("POST", "/v1/completions", json=payload) as response:
                        assert response.status_code == 200
                        assert next(response.iter_lines()).startswith("data: ")
                        rejected = client.post("/v1/completions", json={**payload, "stream": False})
                        assert rejected.status_code == 429
                    until = time.monotonic() + 10
                    while "hllm_active_requests 0" not in client.get("/metrics").text:
                        assert time.monotonic() < until
                        time.sleep(0.01)
                    wait_clean(workers)
                    assert not runtime.cleanup_failed
                    recovered = client.post(
                        "/v1/completions", json={**payload, "max_tokens": 4, "stream": False}
                    )
                    assert recovered.status_code == 200, recovered.text
                    deadline = client.post("/v1/completions", json={**payload, "stream": False})
                    assert deadline.status_code == 504, deadline.text
                    wait_clean(workers)
                    oversized = client.post(
                        "/v1/completions",
                        content=b"x" * (1024 * 1024 + 1),
                        headers={"Content-Type": "application/json"},
                    )
                    assert oversized.status_code == 413
            finally:
                server.should_exit = True
                thread.join(timeout=15)
                assert not thread.is_alive()
        wait_clean(workers, loaded=False)
