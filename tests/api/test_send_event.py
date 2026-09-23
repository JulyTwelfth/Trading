from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.websockets import WebSocketDisconnect, WebSocketState
from pydantic import BaseModel

from app.api.messages import send_event


class DummyEvent(BaseModel):
    type: str = "dummy"
    value: int = 1


def make_ws(state: WebSocketState, send_json_side_effect=None):
    ws = MagicMock()
    ws.application_state = state
    ws.send_json = AsyncMock(side_effect=send_json_side_effect)
    return ws


async def test_send_event_sends_when_connected():
    # Happy path: assert send_json is called when CONNECTED.
    # Also verify the guard: set state to DISCONNECTED and call again — the
    # naive (no-guard) implementation would call send_json a second time,
    # making this test fail on revert.
    ws = make_ws(WebSocketState.CONNECTED)
    await send_event(ws, DummyEvent())
    ws.send_json.assert_awaited_once()

    ws.application_state = WebSocketState.DISCONNECTED
    await send_event(ws, DummyEvent())
    # Still only one call total — the second invocation must have been skipped.
    assert ws.send_json.await_count == 1


async def test_send_event_noops_when_disconnected():
    ws = make_ws(WebSocketState.DISCONNECTED)
    await send_event(ws, DummyEvent())
    ws.send_json.assert_not_awaited()


async def test_send_event_noops_when_connecting():
    # Pre-connection state — sending is also illegal here.
    ws = make_ws(WebSocketState.CONNECTING)
    await send_event(ws, DummyEvent())
    ws.send_json.assert_not_awaited()


async def test_send_event_swallows_runtime_error_from_race():
    # State was CONNECTED when we checked, but the peer closed mid-send.
    ws = make_ws(
        WebSocketState.CONNECTED,
        send_json_side_effect=RuntimeError(
            "Unexpected ASGI message 'websocket.send', after sending 'websocket.close'"
        ),
    )
    await send_event(ws, DummyEvent())
    ws.send_json.assert_awaited_once()


async def test_send_event_swallows_websocket_disconnect():
    ws = make_ws(
        WebSocketState.CONNECTED,
        send_json_side_effect=WebSocketDisconnect(code=1000),
    )
    await send_event(ws, DummyEvent())
    ws.send_json.assert_awaited_once()


async def test_send_event_does_not_swallow_unrelated_exceptions():
    # ValueError isn't from the WS layer — it must propagate so we can debug.
    ws = make_ws(
        WebSocketState.CONNECTED,
        send_json_side_effect=ValueError("serialization broke"),
    )
    with pytest.raises(ValueError):
        await send_event(ws, DummyEvent())

    # Contrast: RuntimeError (the ASGI disconnect error) MUST be swallowed.
    # The naive (no try/except) implementation raises it, so this assertion
    # distinguishes the fix from the naive version.
    ws2 = make_ws(
        WebSocketState.CONNECTED,
        send_json_side_effect=RuntimeError("Unexpected ASGI message 'websocket.send'"),
    )
    await send_event(ws2, DummyEvent())  # must not raise
