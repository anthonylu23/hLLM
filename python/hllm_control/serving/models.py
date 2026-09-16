"""The supported completion subset is explicit; unknown fields are errors."""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StreamOptions(ApiModel):
    include_usage: bool = False


class GenerationOptions(ApiModel):
    model: str
    max_tokens: int = Field(default=32, ge=1, le=1_000_000)
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    top_p: float = Field(default=1, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    n: Literal[1] = 1
    stop: str | list[str] | None = None
    stop_token_ids: list[Annotated[int, Field(ge=0)]] | None = Field(default=None, max_length=32)

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        stops = self.stop_strings()
        if len(stops) > 4 or any(not s or len(s) > 256 for s in stops):
            raise ValueError("stop accepts up to four nonempty strings of at most 256 characters")
        return self

    @property
    def logprob_limit(self) -> int | None:
        return None

    def stop_strings(self) -> list[str]:
        return [self.stop] if isinstance(self.stop, str) else self.stop or []


class CompletionRequest(GenerationOptions):
    logprobs: int | None = Field(default=None, ge=0, le=5)

    @property
    def logprob_limit(self) -> int | None:
        return self.logprobs

    prompt: str = Field(min_length=1, max_length=1_000_000)


class ChatMessage(ApiModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=1_000_000)


class ChatRequest(GenerationOptions):
    logprobs: bool = False
    top_logprobs: int = Field(default=0, ge=0, le=5)

    @property
    def logprob_limit(self) -> int | None:
        return self.top_logprobs if self.logprobs else None

    @model_validator(mode="after")
    def validate_logprobs(self) -> Self:
        if self.top_logprobs and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        return self

    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
