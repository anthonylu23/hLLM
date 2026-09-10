"""Deployment setup and token events; activations and decode stay in native workers."""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping, Sequence
from uuid import uuid4

import grpc

from hllm_control.models import (
    MAXIMUM_RPC_BYTES,
    DeploymentPlan,
    ModelManifest,
    PlanningMode,
    maximum_boundary_tokens,
)
from hllm_control.planner.activation import validate_activation
from hllm_control.planner.measured import ProfileBundle
from hllm_control.proto import (
    common_pb2,
    control_pb2,
    control_pb2_grpc,
    execution_pb2,
    execution_pb2_grpc,
)
from hllm_control.wire import deployment_plan_to_proto, model_manifest_to_proto

# grpcio's generated service factories are untyped. Keep that boundary here.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

_TRANSPORT_DEADLINE_GRACE_SECONDS = 2.0


class DeploymentSession:
    """Own a short-lived deployment on already running workers.

    A session loads stages, streams requests, then cancels requests and unloads on exit.
    Workers must use the local checkpoint matching the prepared manifest.
    """

    def __init__(
        self,
        manifest: ModelManifest,
        plan: DeploymentPlan,
        endpoints: Mapping[str, str],
        *,
        profile_bundle: ProfileBundle | None = None,
    ) -> None:
        if plan.manifest_digest != manifest.manifest_digest:
            raise ValueError("plan and manifest do not match")
        if any(stage.worker_id not in endpoints for stage in plan.stages):
            raise ValueError("missing worker endpoint")
        if plan.planning_mode == PlanningMode.MEASURED and profile_bundle is None:
            raise ValueError("measured activation requires its profile bundle")
        self.profile_bundle = profile_bundle
        self.manifest = manifest
        self.plan = plan
        self.endpoints = dict(endpoints)
        self._channels = [
            grpc.insecure_channel(
                endpoints[stage.worker_id],
                options=[
                    ("grpc.max_receive_message_length", MAXIMUM_RPC_BYTES),
                    ("grpc.max_send_message_length", MAXIMUM_RPC_BYTES),
                ],
            )
            for stage in plan.stages
        ]
        self._controls = [control_pb2_grpc.WorkerControlStub(ch) for ch in self._channels]
        self._loaded: list[int] = []
        self._requests: set[str] = set()

    def __enter__(self) -> DeploymentSession:
        wire_plan = deployment_plan_to_proto(self.plan)
        wire_manifest = model_manifest_to_proto(self.manifest)
        endpoints = [
            control_pb2.StageEndpoint(
                stage_index=stage.stage_index,
                worker_id=stage.worker_id,
                endpoint=self.endpoints[stage.worker_id],
            )
            for stage in self.plan.stages
        ]
        try:
            if self.profile_bundle is not None:
                validate_activation(self.manifest, self.plan, self.profile_bundle, self._controls)
            # Downstream is ready before the driver can accept generation.
            for index in reversed(range(len(self._controls))):
                try:
                    response = self._controls[index].LoadStage(
                        control_pb2.LoadStageRequest(
                            plan=wire_plan,
                            manifest=wire_manifest,
                            stage_index=index,
                            stage_endpoints=endpoints,
                        ),
                        timeout=60,
                    )
                except grpc.RpcError:
                    # A lost response can conceal a successful load. Retire this
                    # plan identity as well as the stages already acknowledged.
                    self._loaded.append(index)
                    raise
                if not response.accepted:
                    raise RuntimeError(f"worker {index} rejected load: {response.detail}")
                self._loaded.append(index)
        except BaseException:
            self.close()
            raise
        return self

    def generate(
        self,
        token_ids: Sequence[int],
        *,
        maximum_new_tokens: int = 32,
        stop_token_ids: Sequence[int] | None = None,
        timeout: float = 60.0,
        request_id: str | None = None,
        capture_timing: bool = False,
    ) -> Generator[execution_pb2.GenerationEvent, None, None]:
        if not token_ids or maximum_new_tokens <= 0 or not 0 < timeout <= 3600:
            raise ValueError("prompt, output length and timeout must be positive and bounded")
        if len(token_ids) + maximum_new_tokens > self.manifest.config.maximum_sequence_length:
            raise ValueError("prompt and output exceed model context capacity")
        if len(self.plan.stages) == 2 and len(token_ids) > maximum_boundary_tokens(
            self.manifest.config.hidden_size, self.plan.activation_dtype
        ):
            raise ValueError("prompt exceeds the stage boundary transport limit")
        if self.profile_bundle is not None and (len(token_ids), maximum_new_tokens) != (
            self.profile_bundle.workload.prompt_tokens,
            self.profile_bundle.workload.output_tokens,
        ):
            raise ValueError("request differs from measured workload")
        stops = self.manifest.config.eos_token_ids if stop_token_ids is None else stop_token_ids
        if any(
            token < 0 or token >= self.manifest.config.vocabulary_size
            for token in (*token_ids, *stops)
        ):
            raise ValueError("token ID exceeds vocabulary")
        identifier = request_id or str(uuid4())
        if identifier in self._requests:
            raise ValueError("request ID is already active in this session")
        self._requests.add(identifier)
        call = execution_pb2_grpc.GenerationStub(self._channels[0]).Generate(
            execution_pb2.GenerationRequest(
                capture_timing=capture_timing,
                deployment_id=self.plan.plan_id,
                deployment_version=self.plan.deployment_version,
                request_id=identifier,
                token_ids=token_ids,
                maximum_new_tokens=maximum_new_tokens,
                stop_token_ids=stops,
                deadline_unix_ms=int((time.time() + timeout) * 1000),
            ),
            # The worker enforces the application deadline itself; the transport
            # deadline trails it slightly so its DEADLINE_EXCEEDED status (rather
            # than a locally synthesized one) is what the caller observes.
            timeout=timeout + _TRANSPORT_DEADLINE_GRACE_SECONDS,
        )
        completed = False
        try:
            for event in call:
                if event.request_id != identifier:
                    raise RuntimeError("generation event request ID mismatch")
                if event.HasField("terminal"):
                    if event.terminal.state != execution_pb2.TERMINAL_STATE_COMPLETED:
                        raise RuntimeError(f"generation failed: {event.terminal}")
                    completed = True
                yield event
            if not completed:
                raise RuntimeError("generation stream ended without a terminal event")
        finally:
            call.cancel()
            if not completed:
                self.cancel(identifier)
            self._requests.discard(identifier)

    def cancel(self, request_id: str) -> None:
        request = control_pb2.CancelRequestMessage(
            plan_id=self.plan.plan_id,
            deployment_version=self.plan.deployment_version,
            request_id=request_id,
            reason="controller cleanup",
        )
        for control in self._controls:
            try:
                control.CancelRequest(request, timeout=2)
            except grpc.RpcError:
                pass  # A lost worker cannot acknowledge cleanup; native deadlines bound its lease.

    def close(self) -> None:
        for request in tuple(self._requests):
            self.cancel(request)
        unload = control_pb2.UnloadStageRequest(
            plan_id=self.plan.plan_id, deployment_version=self.plan.deployment_version
        )
        for index in reversed(self._loaded):
            # Cancellation is asynchronous: allow native compute/stream cleanup to finish.
            until = time.monotonic() + 5
            while True:
                try:
                    self._controls[index].UnloadStage(unload, timeout=2)
                    break
                except grpc.RpcError as error:
                    if (
                        error.code() != grpc.StatusCode.FAILED_PRECONDITION
                        or time.monotonic() >= until
                    ):
                        break
                    time.sleep(0.02)
        self._loaded.clear()
        for channel in self._channels:
            channel.close()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def memory_reports(self) -> list[control_pb2.MemoryReport]:
        return [
            control.GetMemoryReport(common_pb2.Empty(), timeout=5) for control in self._controls
        ]
