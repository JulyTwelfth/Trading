"""POL-36 Bug 3: routine WS disconnects must log at WARNING (no stack trace),
not via logger.exception (which dumps the full traceback)."""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
import websockets

from app.bot import user_ws as user_ws_mod
from app.bot.user_ws import stream_user_trades


async def test_user_ws_logs_warning_not_exception_on_connection_closed(monkeypatch, caplog):
    call_count = 0

    def fake_connect(url, ssl=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise websockets.ConnectionClosed(None, None)
        raise asyncio.CancelledError()

    monkeypatch.setattr(user_ws_mod.websockets, "connect", fake_connect)
    monkeypatch.setattr(user_ws_mod, "USER_WS_RECONNECT_DELAY_SECONDS", 0)

    client = MagicMock()
    # M2: stream_user_trades now calls client.ws_auth() instead of reading client.creds.*
    client.ws_auth.return_value = {"apiKey": "k", "secret": "s", "passphrase": "p"}

    caplog.set_level(logging.DEBUG, logger="app.bot.user_ws")

    with pytest.raises(asyncio.CancelledError):
        async for _ in stream_user_trades(client):
            pass

    closed_records = [r for r in caplog.records if "connection closed" in r.getMessage()]
    assert closed_records, "Expected a 'connection closed' log line"
    assert all(r.levelno == logging.WARNING for r in closed_records), (
        "ConnectionClosed must log at WARNING, not ERROR/EXCEPTION"
    )
    assert all(r.exc_info is None for r in closed_records), (
        "No stack trace on routine close (must not use logger.exception)"
    )
