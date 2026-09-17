"""Bound JSON bodies before parsing; replay the body and preserve disconnect events."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyLimit:
    def __init__(self, app: ASGIApp, maximum_bytes: int = 1024 * 1024) -> None:
        self.app = app
        self.maximum = maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        parts: list[bytes] = []
        length = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            length += len(body)
            if length > self.maximum:
                response = JSONResponse(
                    {
                        "error": {
                            "message": "request body exceeds 1 MiB",
                            "type": "invalid_request_error",
                            "param": None,
                            "code": None,
                        }
                    },
                    status_code=413,
                )
                await response(scope, receive, send)
                return
            parts.append(body)
            if not message.get("more_body", False):
                break
        pending = True

        async def replay() -> Message:
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": b"".join(parts), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
