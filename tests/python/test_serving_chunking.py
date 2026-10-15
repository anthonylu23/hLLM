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
