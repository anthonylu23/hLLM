"""Bounded metadata for consumed native tokens, before text stop filtering."""

from typing import Any

from hllm_control.proto import execution_pb2
from hllm_control.serving.tokenizer import TextTokenizer


def token_logprobs(
    event: execution_pb2.TokenEvent,
    tokenizer: TextTokenizer,
    offset: int,
) -> dict[str, Any]:
    def decode(identifier: int) -> str:
        return tokenizer.native.decode([identifier], skip_special_tokens=False)

    return {
        "token": decode(event.token_id),
        "logprob": event.logprob,
        # A standalone token can be only part of a UTF-8 sequence. Do not invent
        # byte data from its replacement-character display string.
        "bytes": None,
        "text_offset": offset,
        "top_logprobs": [
            {"token": decode(candidate.token_id), "logprob": candidate.logprob, "bytes": None}
            for candidate in event.top_logprobs
        ],
    }


def format_logprobs(records: list[dict[str, Any]], *, chat: bool) -> dict[str, Any]:
    if chat:
        return {
            "content": [
                {k: v for k, v in record.items() if k != "text_offset"} for record in records
            ]
        }
    return {
        "tokens": [record["token"] for record in records],
        "token_logprobs": [record["logprob"] for record in records],
        "top_logprobs": [
            {entry["token"]: entry["logprob"] for entry in record["top_logprobs"]}
            for record in records
        ],
        "text_offset": [record["text_offset"] for record in records],
    }
