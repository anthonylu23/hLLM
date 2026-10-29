"""Serving binds prefill chunking to the measured workload and the boundary limit."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hllm_control.controller import DeploymentSession
from hllm_control.models import DType, WorkloadProfile
from hllm_control.serving.runtime import ServingRuntime


def session(*, chunk: int | None, stages: int = 0) -> MagicMock:
    value = MagicMock(spec=DeploymentSession)
    value.profile_bundle = (
        None
        if chunk is None
        else SimpleNamespace(
            workload=WorkloadProfile.model_construct(
                prompt_tokens=4096, output_tokens=8, concurrency=1, prefill_chunk_tokens=chunk
            )
        )
    )
    value.plan = SimpleNamespace(stages=(None,) * stages, activation_dtype=DType.F16)
    value.manifest = SimpleNamespace(
        config=SimpleNamespace(hidden_size=2560, vocabulary_size=16, maximum_sequence_length=32768)
    )
    return value


def test_measured_bundle_requires_the_same_chunk_size() -> None:
    with pytest.raises(ValueError, match="chunk size 128 differs from the measured workload's 0"):
        asyncio.run(ServingRuntime(session(chunk=0), 1, prefill_chunk_tokens=128).start())
    with pytest.raises(ValueError, match="chunk size 0 differs from the measured workload's 128"):
        asyncio.run(ServingRuntime(session(chunk=128), 1).start())
    # Matching sizes (or no measured bundle) pass the identity check.
    for chunk, bundle in ((0, 0), (128, 128), (128, None)):
        current = session(chunk=bundle)
        asyncio.run(ServingRuntime(current, 1, prefill_chunk_tokens=chunk).start())
        current.__enter__.assert_called_once()


def test_boundary_limit_applies_to_each_chunk() -> None:
    prompt = [1] * 4096  # 4096 x 2560 x 2 bytes exceeds one 8 MiB boundary message.
    with pytest.raises(ValueError, match="boundary transport limit"):
        ServingRuntime(session(chunk=None, stages=2), 1).validate(prompt, 8, ())
    ServingRuntime(session(chunk=None, stages=2), 1, prefill_chunk_tokens=512).validate(
        prompt, 8, ()
    )
    # A single-stage deployment never ships prefill across a boundary.
    ServingRuntime(session(chunk=None, stages=1), 1).validate(prompt, 8, ())


def test_session_sends_the_prefill_chunk_and_bounds_each_chunk() -> None:
    from unittest.mock import patch

    from hllm_control.proto import execution_pb2

    current = DeploymentSession.__new__(DeploymentSession)
    current.manifest = session(chunk=None).manifest
    plan = SimpleNamespace(
        stages=(None, None), activation_dtype=DType.F16, plan_id="p", deployment_version=1
    )
    current.plan = plan  # pyright: ignore[reportAttributeAccessIssue] -- the fields generate() reads
    current._channels = [None]  # pyright: ignore[reportAttributeAccessIssue]
    current._requests = set()
    sent = []

    class Call:
        def __init__(self, request) -> None:
            sent.append(request)
            self.events = iter(
                [
                    execution_pb2.GenerationEvent(
                        request_id=request.request_id,
                        terminal=execution_pb2.TerminalEvent(
                            state=execution_pb2.TERMINAL_STATE_COMPLETED
                        ),
                    )
                ]
            )

        def __iter__(self):
            return self.events

        def cancel(self) -> None:
            pass

    prompt = [1] * 4096  # 4096 x 2560 x 2 bytes exceeds one 8 MiB boundary message.

    def run(**kwargs) -> None:
        current.profile_bundle = kwargs.pop("bundle", None)
        list(current.generate(prompt, maximum_new_tokens=8, stop_token_ids=[], **kwargs))

    with patch("hllm_control.controller.execution_pb2_grpc.GenerationStub") as stub:
        stub.return_value.Generate.side_effect = lambda request, timeout: Call(request)
        run(prefill_chunk_tokens=512)
        assert [r.prefill_chunk_tokens for r in sent] == [512]
        with pytest.raises(ValueError, match="boundary transport limit"):
            run()
        with pytest.raises(ValueError, match="invalid prefill chunk size"):
            run(prefill_chunk_tokens=-1)
        # A measured deployment serves only the chunk size it was measured at.
        with pytest.raises(ValueError, match="differs from measured workload"):
            run(prefill_chunk_tokens=512, bundle=session(chunk=256).profile_bundle)
        run(prefill_chunk_tokens=256, bundle=session(chunk=256).profile_bundle)
        assert [r.prefill_chunk_tokens for r in sent] == [512, 256]
