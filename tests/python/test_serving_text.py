import json
from pathlib import Path

import pytest
from hllm_control.serving.models import ChatRequest, CompletionRequest
from hllm_control.serving.tokenizer import TextOutput, TextTokenizer
from pydantic import ValidationError
from tokenizers import Tokenizer, decoders, models, pre_tokenizers


def tokenizer(root: Path) -> TextTokenizer:
    native = Tokenizer(
        models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2, "STOP": 3}, unk_token="[UNK]")
    )
    native.pre_tokenizer = pre_tokenizers.Whitespace()
    native.decoder = decoders.WordPiece(prefix="##", cleanup=False)
    native.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": "{% for message in messages %}{{ message.content }} {% endfor %}",
            }
        )
    )
    return TextTokenizer(root, 4)


def test_chat_and_stream_stop_matches_across_token_boundaries(tmp_path: Path) -> None:
    text = tokenizer(tmp_path)
    assert text.chat([{"role": "user", "content": "hello world"}]) == [1, 2]
    output = TextOutput(text, ["hello world"])
    assert output.push(1) == ""
    assert output.push(2) == ""
    assert output.stopped
    assert output.finish() == ""
    output = TextOutput(text, ["world STOP"])
    parts = [output.push(token) for token in [1, 2]]
    parts.append(output.finish())
    assert "".join(parts) == "hello world"


@pytest.mark.parametrize(
    "parameters",
    [
        {"temperature": -0.7},
        {"top_p": 0},
        {"top_k": -5},
        {"n": 2},
        {"seed": -1},
        {"logprobs": 6},
        {"stop": ""},
        {"stop": ["a"] * 5},
        {"stream_options": {"include_usage": True}},
        {"max_tokens": True},
    ],
)
def test_unsupported_generation_parameters_are_rejected(parameters: dict) -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(model="tiny", prompt="hello", **parameters)


def test_tools_are_rejected_and_greedy_numeric_defaults_accepted() -> None:
    assert CompletionRequest(model="tiny", prompt="hello", temperature=0.0, top_p=1.0).n == 1
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {"model": "tiny", "messages": [{"role": "tool", "content": "x"}]}
        )


def test_incomplete_utf8_is_held_until_character_finishes(tmp_path: Path) -> None:
    native = Tokenizer(models.BPE(vocab={"<0xC3>": 0, "<0xA9>": 1}, merges=[], byte_fallback=True))
    native.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    native.save(str(tmp_path / "tokenizer.json"))
    text = TextTokenizer(tmp_path, 2)
    output = TextOutput(text, [])
    assert output.push(0) == ""
    assert output.push(1) == "é"
    assert output.finish() == ""


def test_admission_queue_is_bounded_fifo_and_cancel_safe() -> None:
    import asyncio
    import time
    from typing import Any

    from hllm_control.serving.app import Admission, ApiError

    class Connected:
        async def is_disconnected(self) -> bool:
            return False

    request: Any = Connected()

    async def run() -> None:
        queue = Admission(1, 2)
        deadline = time.monotonic() + 5
        await queue.acquire(request, deadline)
        first = asyncio.create_task(queue.acquire(request, deadline))
        second = asyncio.create_task(queue.acquire(request, deadline))
        await asyncio.sleep(0.02)
        with pytest.raises(ApiError, match="full"):
            await queue.acquire(request, deadline)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert len(queue.waiters) == 1
        queue.active -= 1
        # The free active slot plus waiting capacity stays bounded while the
        # oldest waiter is asleep; a burst of arrivals cannot bypass the cap.
        third = asyncio.create_task(queue.acquire(request, deadline))
        fourth = asyncio.create_task(queue.acquire(request, deadline))
        await asyncio.sleep(0)
        with pytest.raises(ApiError, match="full"):
            await queue.acquire(request, deadline)
        await asyncio.wait_for(second, 1)
        assert not third.done() and not fourth.done()
        queue.active -= 1
        await asyncio.wait_for(third, 1)
        assert not fourth.done()
        queue.active -= 1
        await asyncio.wait_for(fourth, 1)
        assert not queue.waiters and queue.active == 1
        queue.active -= 1
        with pytest.raises(ApiError, match="deadline"):
            await queue.acquire(request, time.monotonic() - 1)
        assert not queue.waiters and queue.active == 0

    asyncio.run(run())
