"""Streaming must release emitted response records while generation continues."""

import asyncio
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request
from fastapi.routing import APIRoute
from hllm_control.proto import execution_pb2
from hllm_control.serving.app import create_app
from hllm_control.serving.logprobs import token_logprobs
from hllm_control.serving.models import CompletionRequest
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.serving.tokenizer import TextOutput

from tests.python.test_serving_text import tokenizer


@pytest.mark.parametrize("stream", [False, True])
def test_emitted_response_history_is_only_retained_for_nonstream(
    tmp_path: Path, stream: bool
) -> None:
    class Record(dict):
        pass

    class Fragment(str):
        pass

    records = []
    fragments = []
    original_push = TextOutput.push

    def record(*args):
        result = Record(token_logprobs(*args))
        records.append(weakref.ref(result))
        return result

    def push(output, token):
        result = Fragment(original_push(output, token))
        fragments.append(weakref.ref(result))
        return result

    async def run():
        runtime = MagicMock(spec=ServingRuntime)
        runtime.concurrency = 1
        runtime.cleanup_failed = False
        runtime.supports_sampling = runtime.supports_logprobs = True
        runtime.session = SimpleNamespace(
            manifest=SimpleNamespace(config=SimpleNamespace(eos_token_ids=[], vocabulary_size=4)),
            profile_bundle=None,
        )
        runtime.latest_tokens = {}

        async def generate(identifier, *_args, **_kwargs):
            for _ in range(512):
                runtime.latest_tokens[identifier] = execution_pb2.TokenEvent(
                    token_id=1, logprob=-0.5
                )
                yield 1
                if stream:
                    assert sum(ref() is not None for ref in records) <= 2
                    assert sum(ref() is not None for ref in fragments) <= 2

        runtime.generate = generate
        app = create_app(runtime, tokenizer(tmp_path), model_name="tiny")
        endpoint = next(
            route.endpoint
            for route in app.routes
            if isinstance(route, APIRoute) and route.path == "/v1/completions"
        )
        response = await endpoint(
            CompletionRequest(
                model="tiny", prompt="hello", max_tokens=512, stream=stream, logprobs=0
            ),
            Request({"type": "http"}),
        )
        if stream:
            chunks = [chunk async for chunk in response.iterator]
            assert chunks[-1] == "data: [DONE]\n\n"
        else:
            import json

            choice = json.loads(response.body)["choices"][0]
            assert len(choice["logprobs"]["tokens"]) == 512
            assert choice["text"] == " ".join(["hello"] * 512)
        assert len(records) == len(fragments) == 512
        assert not any(ref() is not None for ref in records + fragments)

    with (
        patch("hllm_control.serving.app.token_logprobs", record),
        patch.object(TextOutput, "push", push),
    ):
        asyncio.run(run())
