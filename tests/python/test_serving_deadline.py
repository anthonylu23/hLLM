"""The serving deadline must end a response with an error, even behind a slow client."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hllm_control.serving.app import create_app
from hllm_control.serving.runtime import ServingRuntime

from tests.python.test_serving_text import tokenizer


def endless_runtime() -> MagicMock:
    runtime = MagicMock(spec=ServingRuntime)
    runtime.concurrency = 1
    runtime.cleanup_failed = False
    runtime.supports_sampling = runtime.supports_logprobs = True
    runtime.session = SimpleNamespace(
        manifest=SimpleNamespace(config=SimpleNamespace(eos_token_ids=[], vocabulary_size=4)),
        profile_bundle=None,
    )
    runtime.latest_tokens = {}
    runtime.metrics = AsyncMock(return_value="")

    async def generate(*_args, **_kwargs):
        while True:
            await asyncio.sleep(0.005)
            yield 1

    runtime.generate = generate
    return runtime


async def call(
    app,
    method: str,
    path: str,
    body: bytes,
    send_delay: float | None,
    disconnect_after: float | None = None,
):
    """Drive one HTTP request through the raw ASGI interface.

    `send_delay` is how long each response body write blocks; None blocks forever,
    a client that stopped reading. `disconnect_after` makes the client go away.
    """
    messages: list[dict] = []
    delivered = False
    begun = asyncio.get_running_loop().time()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 80),
    }

    async def receive():
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        loop = asyncio.get_running_loop()
        if disconnect_after is not None and loop.time() - begun >= disconnect_after:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()  # The client stays connected.

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            if send_delay is None:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(send_delay)
        messages.append(message)

    await app(scope, receive, send)
    return messages


def metrics(app) -> dict[str, float]:
    async def read():
        messages = await call(app, "GET", "/metrics", b"", 0)
        return b"".join(m.get("body", b"") for m in messages).decode()

    text = asyncio.run(read())
    return {
        line.split()[0]: float(line.split()[1])
        for line in text.splitlines()
        if line.startswith("hllm_")
    }


def stream_request() -> bytes:
    return json.dumps(
        {"model": "tiny", "prompt": "hello", "max_tokens": 100000, "stream": True}
    ).encode()


@pytest.mark.parametrize("send_delay", [0.0, 0.15])
def test_stream_deadline_reports_timeout_even_behind_a_slow_client(
    tmp_path: Path, send_delay: float
) -> None:
    app = create_app(endless_runtime(), tokenizer(tmp_path), model_name="tiny", timeout=0.3)
    messages = asyncio.run(
        asyncio.wait_for(call(app, "POST", "/v1/completions", stream_request(), send_delay), 10)
    )
    body = b"".join(m.get("body", b"") for m in messages).decode()
    events = [line[len("data: ") :] for line in body.split("\n\n") if line]
    assert events[-1] == "[DONE]"
    assert json.loads(events[-2])["error"]["type"] == "timeout_error"
    assert messages[-1]["type"] == "http.response.body"
    assert not messages[-1].get("more_body", False)
    counted = metrics(app)
    assert counted["hllm_failed"] == 1
    assert counted["hllm_cancelled"] == 0
    assert counted["hllm_active_requests"] == 0


def test_stream_deadline_ends_a_response_the_client_never_reads(tmp_path: Path) -> None:
    app = create_app(endless_runtime(), tokenizer(tmp_path), model_name="tiny", timeout=0.3)
    with patch("hllm_control.serving.app.STREAM_DRAIN_SECONDS", 0.2):
        asyncio.run(
            asyncio.wait_for(call(app, "POST", "/v1/completions", stream_request(), None), 10)
        )
    counted = metrics(app)
    assert counted["hllm_failed"] == 1
    assert counted["hllm_cancelled"] == 0
    assert counted["hllm_active_requests"] == 0


def test_nonstream_client_disconnect_cancels_generation(tmp_path: Path) -> None:
    app = create_app(endless_runtime(), tokenizer(tmp_path), model_name="tiny", timeout=30)
    body = json.dumps({"model": "tiny", "prompt": "hello", "max_tokens": 100000}).encode()

    async def run() -> None:
        request = asyncio.create_task(call(app, "POST", "/v1/completions", body, 0, 0.1))
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, 10)

    asyncio.run(run())
    counted = metrics(app)
    assert counted["hllm_cancelled"] == 1
    assert counted["hllm_failed"] == 0
    assert counted["hllm_active_requests"] == 0
