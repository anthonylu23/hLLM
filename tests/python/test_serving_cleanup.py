"""Cancellation must not interrupt native retirement and permit early slot reuse."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
from hllm_control.serving.runtime import ServingRuntime


def test_level_cancellation_shields_native_retirement() -> None:
    class Call:
        def cancel(self) -> None:
            pass

        def __aiter__(self):
            return self

        async def __anext__(self):
            await anyio.sleep_forever()

    class Runtime(ServingRuntime):
        def validate(self, tokens, maximum, stops):
            pass

        async def retire(self, identifier):
            await anyio.sleep(0.02)
            retired.append(identifier)

    retired = []

    async def run() -> None:
        session = MagicMock(
            profile_bundle=None, plan=SimpleNamespace(plan_id="p", deployment_version=1)
        )
        runtime = Runtime(session, 1)
        runtime.channels = [MagicMock()]
        with patch("hllm_control.serving.runtime.execution_pb2_grpc.GenerationStub") as stub:
            stub.return_value.Generate.return_value = Call()
            with anyio.move_on_after(0.01):
                async for _token in runtime.generate("cancelled", [1], 3, [], 1):
                    pass
        assert retired == ["cancelled"]
        assert not runtime.calls

    asyncio.run(run())
