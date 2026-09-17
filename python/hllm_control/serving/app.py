"""A bounded OpenAI-compatible HTTP interface for one persistent deployment."""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import anyio
import grpc
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from jinja2 import TemplateError
from starlette.types import Receive, Scope, Send

from hllm_control.proto import execution_pb2
from hllm_control.serving.body_limit import BodyLimit
from hllm_control.serving.logprobs import format_logprobs, token_logprobs
from hllm_control.serving.models import ChatRequest, CompletionRequest, GenerationOptions
from hllm_control.serving.runtime import ServingRuntime
from hllm_control.serving.tokenizer import TextOutput, TextTokenizer

# Nested handlers are registered through FastAPI decorators.
# pyright: reportUnusedFunction=false


class ManagedStream(StreamingResponse):
    def __init__(self, content: AsyncGenerator[str], release: Callable[[], None]) -> None:
        super().__init__(
            content,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
        self.iterator = content
        self.release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.iterator.aclose()
                finally:
                    self.release()


class ApiError(Exception):
    def __init__(
        self, message: str, status: int = 400, kind: str = "invalid_request_error"
    ) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind

    def body(self) -> dict[str, object]:
        return {"error": {"message": str(self), "type": self.kind, "param": None, "code": None}}


def native_error(error: grpc.RpcError) -> ApiError:
    code = error.code() if isinstance(error, grpc.aio.AioRpcError) else None
    if code == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return ApiError("native admission capacity exceeded", 429, "rate_limit_error")
    if code == grpc.StatusCode.DEADLINE_EXCEEDED:
        return ApiError("generation deadline exceeded", 504, "timeout_error")
    return ApiError("native worker failed or became unavailable", 503, "server_error")


@dataclass
class Metrics:
    requests: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    rejected: int = 0
    generated_tokens: int = 0
    ttft_seconds: float = 0
    ttft_count: int = 0
    itl_seconds: float = 0
    itl_count: int = 0
    elapsed_seconds: float = 0


class Admission:
    """Event-loop-owned FIFO admission; queued requests do not reserve native KV."""

    def __init__(self, maximum: int, queued: int) -> None:
        self.maximum = maximum
        self.queued = queued
        self.active = 0
        self.waiters: deque[object] = deque()

    async def acquire(self, request: Request, deadline: float) -> None:
        if self.active + len(self.waiters) >= self.maximum + self.queued:
            raise ApiError("serving queue is full", 429, "rate_limit_error")
        ticket = object()
        self.waiters.append(ticket)
        try:
            while self.waiters[0] is not ticket or self.active >= self.maximum:
                if time.monotonic() >= deadline:
                    raise ApiError("deadline expired while queued", 504, "timeout_error")
                if await request.is_disconnected():
                    raise ApiError("client disconnected", 499, "cancelled_error")
                await asyncio.sleep(0.01)
            if time.monotonic() >= deadline:
                raise ApiError("deadline expired while queued", 504, "timeout_error")
            self.active += 1
        finally:
            self.waiters.remove(ticket)


def create_app(
    runtime: ServingRuntime,
    tokenizer: TextTokenizer,
    *,
    model_name: str | None = None,
    maximum_queued: int = 16,
    timeout: float = 60,
) -> FastAPI:
    if (
        not 1 <= runtime.concurrency <= 64
        or not 0 <= maximum_queued <= 1024
        or not 0 < timeout <= 3600
    ):
        raise ValueError("invalid serving limits")
    name = model_name or runtime.session.manifest.source.model_id
    admission = Admission(runtime.concurrency, maximum_queued)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="hLLM Runtime", lifespan=lifespan)
    app.add_middleware(BodyLimit)

    @app.exception_handler(ApiError)
    async def api_error(_request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(error.body(), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        messages = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in error.errors()]
        return JSONResponse(ApiError("; ".join(messages)).body(), status_code=400)

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": name, "object": "model", "created": 0, "owned_by": "hllm"}],
        }

    @app.get("/v1/deployments")
    async def deployments() -> dict[str, object]:
        return {
            "data": [
                {
                    "id": runtime.session.plan.plan_id,
                    "model": name,
                    "stages": len(runtime.session.plan.stages),
                    "maximum_active_requests": runtime.concurrency,
                    "sampling": (
                        ["greedy", "temperature", "top_p", "top_k"]
                        if runtime.supports_sampling
                        else ["greedy"]
                    ),
                    "prefill_chunk_tokens": runtime.prefill_chunk_tokens,
                }
            ]
        }

    @app.get("/health")
    async def health() -> JSONResponse:
        healthy = await runtime.healthy()
        return JSONResponse(
            {"status": "ok" if healthy else "unavailable"}, status_code=200 if healthy else 503
        )

    @app.get("/metrics")
    async def metric_response() -> PlainTextResponse:
        values = {
            **vars(metrics),
            "active_requests": admission.active,
            "queued_requests": len(admission.waiters),
        }
        text = "\n".join(f"hllm_{key} {value}" for key, value in values.items()) + "\n"
        return PlainTextResponse(
            text + await runtime.metrics(), media_type="text/plain; version=0.0.4"
        )

    async def complete(
        options: GenerationOptions, tokens: list[int], request: Request, *, chat: bool
    ) -> JSONResponse | StreamingResponse:
        if runtime.cleanup_failed:
            raise ApiError(
                "worker cleanup is unconfirmed; restart the deployment", 503, "server_error"
            )
        if options.model != name:
            raise ApiError("requested model is not deployed", 404)
        stops = options.stop_token_ids
        if stops is None:
            stops = list(runtime.session.manifest.config.eos_token_ids)
        try:
            runtime.validate(tokens, options.max_tokens, stops)
        except ValueError as error:
            raise ApiError(str(error)) from error
        sampling = None
        if (
            options.temperature != 0
            or options.top_k
            or options.top_p != 1
            or options.seed is not None
            or options.logprob_limit is not None
        ):
            if options.logprob_limit is not None and not runtime.supports_logprobs:
                raise ApiError("worker does not support log probabilities")
            if not runtime.supports_sampling:
                raise ApiError("worker does not support sampling")
            if options.top_k > runtime.session.manifest.config.vocabulary_size:
                raise ApiError("top_k exceeds vocabulary")
            if runtime.session.profile_bundle is not None:
                raise ApiError("sampling is not qualified by this measured profile")
            sampling = execution_pb2.SamplingOptions(
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                seed=options.seed,
                return_logprobs=options.logprob_limit is not None,
                top_logprobs=options.logprob_limit or 0,
            )
        started = time.monotonic()
        deadline = started + timeout
        metrics.requests += 1
        try:
            await admission.acquire(request, deadline)
        except ApiError:
            metrics.rejected += 1
            raise
        if runtime.cleanup_failed:
            admission.active -= 1
            raise ApiError(
                "worker cleanup is unconfirmed; restart the deployment", 503, "server_error"
            )
        slot_owned = True

        def release_slot() -> None:
            nonlocal slot_owned
            if slot_owned:
                admission.active -= 1
                slot_owned = False

        identifier = ("chatcmpl-" if chat else "cmpl-") + uuid4().hex
        created = int(time.time())
        common = {"id": identifier, "created": created, "model": name}
        usage = {"prompt_tokens": len(tokens), "completion_tokens": 0, "total_tokens": len(tokens)}
        output = TextOutput(tokenizer, options.stop_strings())
        finish_reason = "length"
        pieces: list[str] = []
        logprob_records: list[dict[str, Any]] = []

        def chunk(
            text: str, finish: str | None = None, details: list[dict[str, Any]] | None = None
        ) -> dict[str, object]:
            choice: dict[str, object] = {"index": 0, "finish_reason": finish}
            if chat:
                choice["delta"] = {"content": text} if text else {}
            else:
                choice.update(text=text, logprobs=None)
            if details is not None:
                choice["logprobs"] = format_logprobs(details, chat=chat)
            return {
                **common,
                "object": "chat.completion.chunk" if chat else "text_completion",
                "choices": [choice],
            }

        async def events() -> AsyncGenerator[dict[str, object]]:
            nonlocal finish_reason
            remaining = deadline - time.monotonic()
            native = runtime.generate(
                identifier,
                tokens,
                options.max_tokens,
                stops,
                max(0.001, remaining),
                **({"sampling": sampling} if sampling is not None else {}),
            )
            previous_token: float | None = None
            succeeded = False
            task = asyncio.current_task()

            async def watch_disconnect() -> None:
                while True:
                    if await request.is_disconnected():
                        if task is not None:
                            task.cancel()
                        return
                    await asyncio.sleep(0.02)

            monitor = asyncio.create_task(watch_disconnect()) if not options.stream else None
            try:
                async with asyncio.timeout(max(0.001, remaining)):
                    if chat and options.stream:
                        yield {
                            **common,
                            "object": "chat.completion.chunk",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    async for token in native:
                        now = time.monotonic()
                        if previous_token is None:
                            metrics.ttft_seconds += now - started
                            metrics.ttft_count += 1
                        else:
                            metrics.itl_seconds += now - previous_token
                            metrics.itl_count += 1
                        previous_token = now
                        usage["completion_tokens"] += 1
                        usage["total_tokens"] += 1
                        metrics.generated_tokens += 1
                        details = None
                        if options.logprob_limit is not None:
                            metadata = runtime.latest_tokens.get(identifier)
                            if metadata is None:
                                raise RuntimeError(
                                    "native token omitted requested log probabilities"
                                )
                            entry = token_logprobs(metadata, tokenizer, len(output.decoded))
                            if not options.stream:
                                logprob_records.append(entry)
                            details = [entry]
                        if token in stops:
                            finish_reason = "stop"
                            if details is not None:
                                yield chunk("", details=details)
                            continue  # Drain native terminal acknowledgment before success.
                        text = output.push(token)
                        if text and not options.stream:
                            pieces.append(text)
                        if text or details is not None:
                            yield chunk(text, details=details)
                        if output.stopped:
                            finish_reason = "stop"
                            break
                    text = output.finish()
                    if text:
                        if not options.stream:
                            pieces.append(text)
                        yield chunk(text)
                    if output.stopped:
                        finish_reason = "stop"
                succeeded = True
                metrics.completed += 1
                yield chunk("", finish_reason)
                if options.stream_options and options.stream_options.include_usage:
                    yield {
                        **common,
                        "object": "chat.completion.chunk" if chat else "text_completion",
                        "choices": [],
                        "usage": usage,
                    }
            except TimeoutError as error:
                metrics.failed += 1
                raise ApiError("generation deadline exceeded", 504, "timeout_error") from error
            except grpc.RpcError as error:
                metrics.failed += 1
                raise native_error(error) from error
            except (asyncio.CancelledError, GeneratorExit):
                if not succeeded:
                    metrics.cancelled += 1
                raise
            except Exception as error:
                metrics.failed += 1
                raise ApiError("generation failed", 500, "server_error") from error
            finally:
                if monitor is not None:
                    monitor.cancel()
                with anyio.CancelScope(shield=True):
                    try:
                        await native.aclose()
                    finally:
                        release_slot()
                        metrics.elapsed_seconds += time.monotonic() - started

        if options.stream:

            async def stream() -> AsyncGenerator[str]:
                iterator = events()
                try:
                    async for event in iterator:
                        yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
                except ApiError as error:
                    yield "data: " + json.dumps(error.body()) + "\n\n"
                finally:
                    await iterator.aclose()
                yield "data: [DONE]\n\n"

            return ManagedStream(stream(), release_slot)
        async for _event in events():
            pass
        choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason}
        if chat:
            choice["message"] = {"role": "assistant", "content": "".join(pieces)}
        else:
            choice.update(text="".join(pieces), logprobs=None)
        if options.logprob_limit is not None:
            choice["logprobs"] = format_logprobs(logprob_records, chat=chat)
        return JSONResponse(
            {
                **common,
                "object": "chat.completion" if chat else "text_completion",
                "choices": [choice],
                "usage": usage,
            }
        )

    @app.post("/v1/completions", response_model=None)
    async def completions(
        options: CompletionRequest, request: Request
    ) -> JSONResponse | StreamingResponse:
        return await complete(options, tokenizer.encode(options.prompt), request, chat=False)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        options: ChatRequest, request: Request
    ) -> JSONResponse | StreamingResponse:
        try:
            tokens = tokenizer.chat([message.model_dump() for message in options.messages])
        except (ValueError, TemplateError) as error:
            raise ApiError(str(error)) from error
        return await complete(options, tokens, request, chat=True)

    return app
