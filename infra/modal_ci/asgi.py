"""Minimal ASGI front door for the controller: stdlib only, no web framework to configure or patch.

`POST /github` hands the raw body and headers to `Core.handle_webhook`, which authenticates them;
nothing here parses or trusts the payload. `GET /health` is liveness only and reveals nothing.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from infra.modal_ci.core import MAX_BODY_BYTES, Core

log = logging.getLogger("modal_ci")

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]


async def _respond(send: Send, status: int, body: str) -> None:
    await send(
        {"type": "http.response.start", "status": status, "headers": [(b"content-type", b"text/plain; charset=utf-8")]}
    )
    await send({"type": "http.response.body", "body": body.encode()})


class _TooLarge(Exception):
    pass


class _Disconnected(Exception):
    pass


async def _read_body(receive: Receive) -> bytes:
    """The request body; raises once it exceeds the size limit (stop reading rather than buffer it)."""
    chunks: list[bytes] = []
    size = 0
    while True:
        message = await receive()
        if message["type"] != "http.request":
            raise _Disconnected
        chunks.append(message.get("body", b""))
        size += len(chunks[-1])
        if size > MAX_BODY_BYTES:
            raise _TooLarge
        if not message.get("more_body"):
            return b"".join(chunks)


def make_app(core: Core) -> Callable[[Scope, Receive, Send], Awaitable[None]]:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return

        method, path = scope["method"], scope["path"]
        if path == "/health" and method == "GET":
            await _respond(send, 200, "ok")
        elif path == "/github" and method == "POST":
            try:
                body = await _read_body(receive)
            except _TooLarge:
                await _respond(send, 413, "too large")
                return
            except _Disconnected:
                return  # nobody is listening for a response
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
            try:
                resp = await asyncio.to_thread(core.handle_webhook, headers, body)
            except Exception:
                log.exception("unhandled error in webhook handler")
                await _respond(send, 500, "error")
                return
            await _respond(send, resp.status, resp.body)
        elif path in ("/health", "/github"):
            await _respond(send, 405, "method not allowed")
        else:
            await _respond(send, 404, "not found")

    return app
