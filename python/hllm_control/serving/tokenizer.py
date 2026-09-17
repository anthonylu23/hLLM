"""Local tokenizer assets and text-only sandboxed chat templates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from jinja2 import Template, TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream  # pyright: ignore[reportUnknownVariableType]


class _Decoder(Protocol):
    def step(self, tokenizer: Tokenizer, id: int) -> str | None: ...


def _tojson(value: object, **kwargs: Any) -> str:
    return json.dumps(value, ensure_ascii=False, **kwargs)


def _raise_template(message: str) -> None:
    raise TemplateError(message)


class TextTokenizer:
    def __init__(self, root: Path, vocabulary_size: int) -> None:
        self.native = Tokenizer.from_file(str(root / "tokenizer.json"))
        if any(index >= vocabulary_size for index in self.native.get_vocab().values()):
            raise ValueError("tokenizer IDs exceed the model vocabulary")
        self.native.no_truncation()
        self.native.no_padding()
        config_path = root / "tokenizer_config.json"
        config: dict[str, Any] = json.loads(config_path.read_text()) if config_path.exists() else {}
        template_path = root / "chat_template.jinja"
        source = (
            template_path.read_text() if template_path.exists() else config.get("chat_template")
        )
        if isinstance(source, list):
            source = next(
                (
                    t["template"]
                    for t in cast(list[dict[str, Any]], source)
                    if t.get("name") == "default"
                ),
                None,
            )
        if source is not None and not isinstance(source, str):
            raise ValueError("a single default chat template is required")
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        environment.globals["raise_exception"] = _raise_template  # pyright: ignore[reportUnknownMemberType]
        environment.filters["tojson"] = _tojson  # pyright: ignore[reportUnknownMemberType]
        self.template: Template | None = environment.from_string(source) if source else None
        self.special_tokens: dict[str, str] = {
            key: value["content"] if isinstance(value, dict) else value
            for key, value in config.items()
            if key.endswith("_token") and isinstance(value, (str, dict))
        }

    def encode(self, text: str) -> list[int]:
        return self.native.encode(text, add_special_tokens=True).ids

    def chat(self, messages: list[dict[str, str]]) -> list[int]:
        if self.template is None:
            raise ValueError("this tokenizer has no chat template; use /v1/completions")
        text = self.template.render(
            messages=messages, add_generation_prompt=True, **self.special_tokens
        )
        return self.native.encode(text, add_special_tokens=False).ids


class TextOutput:
    """Incremental UTF-8 decoding and stop matching without exposing stop prefixes."""

    def __init__(self, tokenizer: TextTokenizer, stops: Sequence[str]) -> None:
        self.tokenizer = tokenizer
        self.stream = cast(_Decoder, DecodeStream(skip_special_tokens=True))
        self.stops = tuple(stops)
        self.ids: list[int] = []
        self.decoded = ""
        self.pending = ""
        self.stopped = False

    def _release(self, text: str, *, final: bool = False) -> str:
        self.pending += text
        matches = [self.pending.find(stop) for stop in self.stops if stop in self.pending]
        if matches:
            output = self.pending[: min(matches)]
            self.pending = ""
            self.stopped = True
            return output
        retained = 0
        if not final:
            for stop in self.stops:
                for length in range(1, min(len(stop), len(self.pending) + 1)):
                    if self.pending.endswith(stop[:length]):
                        retained = max(retained, length)
        split = len(self.pending) - retained
        output, self.pending = self.pending[:split], self.pending[split:]
        return output

    def push(self, token: int) -> str:
        self.ids.append(token)
        piece = self.stream.step(self.tokenizer.native, token) or ""
        self.decoded += piece
        return self._release(piece)

    def finish(self) -> str:
        if self.stopped:
            return ""
        full = self.tokenizer.native.decode(self.ids, skip_special_tokens=True)
        if not full.startswith(self.decoded):
            raise ValueError("tokenizer decoder does not preserve streamed text")
        return self._release(full[len(self.decoded) :], final=True)
