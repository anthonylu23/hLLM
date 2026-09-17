"""Async token RPCs over a deployment loaded once for the server lifetime."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Sequence
from typing import Any

import anyio
import grpc
from grpc import aio

from hllm_control.controller import DeploymentSession
from hllm_control.models import MAXIMUM_RPC_BYTES, maximum_boundary_tokens
from hllm_control.proto import (
    common_pb2,
    control_pb2,
    control_pb2_grpc,
    execution_pb2,
    execution_pb2_grpc,
)

# Generated grpc factories have no typed async overload.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false


class ServingRuntime:
    def __init__(
        self, session: DeploymentSession, concurrency: int, *, prefill_chunk_tokens: int = 0
    ) -> None:
        self.session = session
        self.concurrency = concurrency
        if not 0 <= prefill_chunk_tokens <= 65536:
            raise ValueError("invalid prefill chunk size")
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.supports_sampling = False
        self.supports_logprobs = False
        self.latest_tokens: dict[str, execution_pb2.TokenEvent] = {}
        self.channels: list[aio.Channel] = []
        self.controls: list[Any] = []
        self.calls: dict[str, Any] = {}
        self.cleanup_failed = False

    async def start(self) -> None:
        if (
            self.session.profile_bundle
            and self.concurrency != self.session.profile_bundle.workload.concurrency
        ):
            raise ValueError("serving concurrency differs from measured workload")
        if self.prefill_chunk_tokens and self.session.profile_bundle:
            raise ValueError("chunked prefill requires new measured profiles")
        self.channels = [
            aio.insecure_channel(
                self.session.endpoints[s.worker_id],
                options=[
                    ("grpc.max_receive_message_length", MAXIMUM_RPC_BYTES),
                    ("grpc.max_send_message_length", MAXIMUM_RPC_BYTES),
                ],
            )
            for s in self.session.plan.stages
        ]
        self.controls = [control_pb2_grpc.WorkerControlStub(c) for c in self.channels]
        try:
            sampling_support = []
            logprob_support = []
            for control in self.controls:
                capabilities = await control.GetCapabilities(common_pb2.Empty(), timeout=5)
                sampling_support.append(capabilities.worker.supports_sampling)
                logprob_support.append(capabilities.worker.supports_logprobs)
                if self.prefill_chunk_tokens and not capabilities.worker.supports_chunked_prefill:
                    raise ValueError("worker does not support chunked prefill")
                if self.concurrency > (capabilities.worker.maximum_active_requests or 1):
                    raise ValueError("worker --max-active-requests is below serving concurrency")
            self.supports_sampling = all(sampling_support)
            self.supports_logprobs = all(logprob_support)
            await asyncio.to_thread(self.session.__enter__)
        except BaseException:
            for channel in self.channels:
                await channel.close()
            self.session.close()
            raise

    async def close(self) -> None:
        for call in tuple(self.calls.values()):
            call.cancel()
        for channel in self.channels:
            await channel.close()
        await asyncio.to_thread(self.session.close)

    def validate(self, tokens: Sequence[int], maximum: int, stops: Sequence[int]) -> None:
        config = self.session.manifest.config
        if not tokens or len(tokens) + maximum > config.maximum_sequence_length:
            raise ValueError("prompt and output exceed model context capacity or prompt is empty")
        if any(token < 0 or token >= config.vocabulary_size for token in (*tokens, *stops)):
            raise ValueError("token ID exceeds vocabulary")
        if len(self.session.plan.stages) == 2 and min(
            len(tokens), self.prefill_chunk_tokens or len(tokens)
        ) > maximum_boundary_tokens(config.hidden_size, self.session.plan.activation_dtype):
            raise ValueError("prompt exceeds the stage boundary transport limit")
        bundle = self.session.profile_bundle
        if bundle and (len(tokens), maximum) != (
            bundle.workload.prompt_tokens,
            bundle.workload.output_tokens,
        ):
            raise ValueError("request differs from measured workload")

    async def generate(
        self,
        identifier: str,
        tokens: Sequence[int],
        maximum: int,
        stops: Sequence[int],
        timeout: float,
        sampling: execution_pb2.SamplingOptions | None = None,
    ) -> AsyncGenerator[int]:
        if maximum <= 0 or not 0 < timeout <= 3600:
            raise ValueError("output length and timeout must be positive and bounded")
        if identifier in self.calls:
            raise ValueError("request ID is already active")
        self.validate(tokens, maximum, stops)
        if sampling is not None and not self.supports_sampling:
            raise ValueError("worker does not support sampling")
        if sampling is not None and self.session.profile_bundle is not None:
            raise ValueError("sampling is not qualified by this measured profile")
        call = execution_pb2_grpc.GenerationStub(self.channels[0]).Generate(
            execution_pb2.GenerationRequest(
                sampling=sampling,
                prefill_chunk_tokens=self.prefill_chunk_tokens,
                deployment_id=self.session.plan.plan_id,
                deployment_version=self.session.plan.deployment_version,
                request_id=identifier,
                token_ids=tokens,
                maximum_new_tokens=maximum,
                stop_token_ids=stops,
                deadline_unix_ms=int((time.time() + timeout) * 1000),
            ),
            timeout=timeout + 2,
        )
        self.calls[identifier] = call
        completed = False
        try:
            async for event in call:
                if event.request_id != identifier:
                    raise RuntimeError("generation event request ID mismatch")
                if event.HasField("token"):
                    if event.token.HasField("logprob"):
                        self.latest_tokens[identifier] = event.token
                    yield event.token.token_id
                elif event.HasField("terminal"):
                    if event.terminal.state != execution_pb2.TERMINAL_STATE_COMPLETED:
                        raise RuntimeError("native generation failed")
                    completed = True
            if not completed:
                raise RuntimeError("generation ended without completion")
        finally:
            # Starlette uses level cancellation: every await in an unshielded
            # finally can be cancelled again. Retire before allowing slot reuse.
            with anyio.CancelScope(shield=True):
                try:
                    call.cancel()
                    if not completed:
                        await self.retire(identifier)
                finally:
                    self.calls.pop(identifier, None)
                    self.latest_tokens.pop(identifier, None)

    async def retire(self, identifier: str) -> None:
        request = control_pb2.CancelRequestMessage(
            plan_id=self.session.plan.plan_id,
            deployment_version=self.session.plan.deployment_version,
            request_id=identifier,
            reason="HTTP request ended",
        )
        try:
            await asyncio.gather(*[c.CancelRequest(request, timeout=2) for c in self.controls])
            until = time.monotonic() + 5
            while True:
                reports = await asyncio.gather(
                    *[c.GetMemoryReport(common_pb2.Empty(), timeout=2) for c in self.controls]
                )
                if all(identifier not in r.active_request_ids for r in reports):
                    return
                if time.monotonic() >= until:
                    break
                await asyncio.sleep(0.01)
        except grpc.RpcError:
            pass
        self.cleanup_failed = True

    async def healthy(self) -> bool:
        try:
            reports = await asyncio.gather(
                *[c.GetMemoryReport(common_pb2.Empty(), timeout=2) for c in self.controls]
            )
            return (
                not self.cleanup_failed
                and bool(reports)
                and all(r.loaded_weight_bytes > 0 for r in reports)
            )
        except grpc.RpcError:
            return False

    async def metrics(self) -> str:
        rows: list[str] = []
        for index, control in enumerate(self.controls):
            try:
                memory = await control.GetMemoryReport(common_pb2.Empty(), timeout=2)
                metrics = await control.GetMetrics(common_pb2.Empty(), timeout=2)
            except grpc.RpcError:
                rows.append(f'hllm_worker_up{{stage="{index}"}} 0')
                continue
            rows.append(f'hllm_worker_up{{stage="{index}"}} 1')
            for field in ("active_requests", "reserved_cache_bytes", "reserved_workspace_bytes"):
                rows.append(f'hllm_worker_{field}{{stage="{index}"}} {getattr(memory, field)}')
            for field in (
                "queued_prefills",
                "queued_decodes",
                "executing_microbatches",
                "completed_steps",
                "queue_wait_ms",
                "decode_batches",
                "largest_decode_batch",
            ):
                rows.append(f'hllm_worker_{field}{{stage="{index}"}} {getattr(metrics, field)}')
        return "\n".join(rows) + "\n"
