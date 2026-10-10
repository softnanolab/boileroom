"""The HTTP front door: routing, body limits, header passthrough, failure behaviour."""

import asyncio
from typing import cast

from test_core import delivery, world  # noqa: F401

from infra.modal_ci.asgi import make_app
from infra.modal_ci.core import MAX_BODY_BYTES, Core, Response


class Stub:
    def __init__(self, response: Response | None = None, error: Exception | None = None) -> None:
        self.calls: list[tuple[dict[str, str], bytes]] = []
        self.response = response or Response(200, "ok")
        self.error = error

    def handle_webhook(self, headers, body):
        self.calls.append((headers, body))
        if self.error:
            raise self.error
        return self.response


def app_for(stub: Stub):  # noqa: ANN201
    return make_app(cast(Core, stub))


async def _run(app, scope, receive, send) -> None:
    await app(scope, receive, send)


def call(app, method="POST", path="/github", headers=None, chunks=(b"",)):
    """Drive the ASGI app once; returns (status, body)."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    incoming = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]
    sent = []

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(_run(app, scope, receive, send))
    return sent[0]["status"], sent[1]["body"].decode()


def test_health_is_plain_liveness():
    assert call(app_for(Stub()), "GET", "/health") == (200, "ok")


def test_unknown_path_and_wrong_method():
    app = app_for(Stub())
    assert call(app, "GET", "/")[0] == 404
    assert call(app, "GET", "/github")[0] == 405
    assert call(app, "POST", "/health")[0] == 405


def test_body_is_reassembled_and_headers_passed_through():
    stub = Stub(Response(204, "ignored"))
    status, body = call(app_for(stub), headers={"X-GitHub-Event": "ping"}, chunks=(b"ab", b"cd", b"ef"))
    assert (status, body) == (204, "ignored")
    headers, payload = stub.calls[0]
    assert payload == b"abcdef"
    assert headers["x-github-event"] == "ping"


def test_oversized_body_is_refused_without_calling_the_handler():
    stub = Stub()
    status, _ = call(app_for(stub), chunks=(b"x" * MAX_BODY_BYTES, b"y"))
    assert status == 413
    assert stub.calls == []


def test_client_disconnect_mid_body_gets_no_response_and_no_handler_call():
    stub = Stub()
    scope = {"type": "http", "method": "POST", "path": "/github", "headers": []}
    incoming = [{"type": "http.request", "body": b"abc", "more_body": True}, {"type": "http.disconnect"}]
    sent: list[dict] = []

    async def receive():
        return incoming.pop(0)

    async def send(message):
        sent.append(message)

    asyncio.run(_run(app_for(stub), scope, receive, send))
    assert sent == [] and stub.calls == []


def test_handler_crash_is_a_500_that_leaks_nothing():
    status, body = call(app_for(Stub(error=RuntimeError("token ghs_secret"))))
    assert (status, body) == (500, "error")


def test_lifespan_is_acknowledged():
    app = app_for(Stub())
    inbox = [{"type": "lifespan.startup"}, {"type": "lifespan.shutdown"}]
    sent = []

    async def receive():
        return inbox.pop(0)

    async def send(message):
        sent.append(message["type"])

    asyncio.run(_run(app, {"type": "lifespan"}, receive, send))
    assert sent == ["lifespan.startup.complete", "lifespan.shutdown.complete"]


def test_real_core_over_http_rejects_bad_signature_and_launches_good_one(world):  # noqa: F811
    core, _, sandboxes, *_ = world
    app = make_app(core)
    headers, body = delivery()
    bad = {**headers, "X-Hub-Signature-256": "sha256=" + "0" * 64}
    assert call(app, headers=bad, chunks=(body,)) == (401, "bad signature")
    assert sandboxes.created == []
    assert call(app, headers=headers, chunks=(body,)) == (200, "launched")
    assert len(sandboxes.created) == 1
    assert call(app, headers=headers, chunks=(body,)) == (200, "duplicate delivery")
    assert len(sandboxes.created) == 1
