"""Line coverage for app.bot.user_ws.stream_user_trades.

The ConnectionClosed reconnect-WARNING path is already covered by
tests/bot/test_ws_reconnect_logging.py, so we deliberately do NOT duplicate it.
Here we cover: happy-path connect + subscribe + yield a UserTrade for a `trade`
frame, skipping non-trade frames, the JSONDecodeError branch, the ValidationError
(logger.exception) branch, and the generic-Exception -> reconnect branch.

No real socket: monkeypatch the module's `websockets.connect` with a fake async
context manager whose async-iterator yields scripted JSON strings; reconnect delay
is set to 0.
"""

import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest

from app.bot import user_ws as user_ws_mod
from app.bot.schemas import UserTrade
from app.bot.user_ws import stream_user_trades


class FakeConnection:
    def __init__(self, messages):
        self._messages = messages
        self.sent = []

    async def send(self, payload):
        self.sent.append(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def __aiter__(self):
        for m in self._messages:
            yield m


def trade_frame() -> dict:
    return {
        "event_type": "trade",
        "id": "trade-1",
        "asset_id": "tok-1",
        "market": "0xmarket",
        "side": "BUY",
        "price": "0.5",
        "size": "10",
        "outcome": "Yes",
        "status": "MATCHED",
        "timestamp": "1718000000",
        "maker_orders": [],
        "taker_order_id": "oid-1",
    }


def make_client() -> MagicMock:
    client = MagicMock()
    # M2: stream_user_trades now calls client.ws_auth() instead of reading client.creds.*
    client.ws_auth.return_value = {"apiKey": "k", "secret": "s", "passphrase": "p"}
    return client


def make_connect_then_cancel(monkeypatch, messages):
    holder = {"conn": None}
    call_count = 0

    def fake_connect(url, ssl=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            holder["conn"] = FakeConnection(messages)
            return holder["conn"]
        raise asyncio.CancelledError()

    monkeypatch.setattr(user_ws_mod.websockets, "connect", fake_connect)
    monkeypatch.setattr(user_ws_mod, "USER_WS_RECONNECT_DELAY_SECONDS", 0)
    return holder


async def collect(client):
    out = []
    with pytest.raises(asyncio.CancelledError):
        async for ev in stream_user_trades(client):
            out.append(ev)
    return out


async def test_yields_user_trade_and_sends_subscribe(monkeypatch):
    holder = make_connect_then_cancel(monkeypatch, [json.dumps(trade_frame())])
    client = make_client()

    events = await collect(client)

    assert len(events) == 1
    assert isinstance(events[0], UserTrade)
    assert events[0].id == "trade-1"
    assert events[0].outcome == "YES"  # normalized from "Yes"

    sent = json.loads(holder["conn"].sent[0])
    assert sent["type"] == "user"
    assert sent["auth"]["apiKey"] == "k"
    assert sent["auth"]["secret"] == "s"
    assert sent["auth"]["passphrase"] == "p"


async def test_skips_non_trade_frames(monkeypatch):
    non_trade = {"event_type": "book", "market": "m"}
    messages = [json.dumps(non_trade), json.dumps(trade_frame())]
    make_connect_then_cancel(monkeypatch, messages)

    events = await collect(make_client())

    assert len(events) == 1
    assert isinstance(events[0], UserTrade)


async def test_list_frame_normalization(monkeypatch):
    # A single message that is a list of frames must be flattened; non-trade skipped.
    messages = [json.dumps([{"event_type": "book"}, trade_frame()])]
    make_connect_then_cancel(monkeypatch, messages)

    events = await collect(make_client())

    assert len(events) == 1
    assert isinstance(events[0], UserTrade)


async def test_non_json_frame_warns_and_continues(monkeypatch, caplog):
    messages = ["not-json{", json.dumps(trade_frame())]
    make_connect_then_cancel(monkeypatch, messages)
    caplog.set_level(logging.WARNING, logger="app.bot.user_ws")

    events = await collect(make_client())

    assert len(events) == 1
    assert isinstance(events[0], UserTrade)
    assert any("non-json frame ignored" in r.getMessage() for r in caplog.records)


async def test_invalid_trade_logs_exception_and_continues(monkeypatch, caplog):
    bad = trade_frame()
    del bad["taker_order_id"]  # required field missing -> ValidationError
    messages = [json.dumps(bad), json.dumps(trade_frame())]
    make_connect_then_cancel(monkeypatch, messages)
    caplog.set_level(logging.ERROR, logger="app.bot.user_ws")

    events = await collect(make_client())

    # Bad trade did not yield; the valid trade after it still yielded.
    assert len(events) == 1
    assert isinstance(events[0], UserTrade)
    exc_records = [r for r in caplog.records if "failed to parse trade frame" in r.getMessage()]
    assert exc_records and exc_records[0].exc_info is not None


async def test_generic_exception_triggers_reconnect(monkeypatch, caplog):
    call_count = 0

    def fake_connect(url, ssl=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        raise asyncio.CancelledError()

    monkeypatch.setattr(user_ws_mod.websockets, "connect", fake_connect)
    monkeypatch.setattr(user_ws_mod, "USER_WS_RECONNECT_DELAY_SECONDS", 0)
    caplog.set_level(logging.ERROR, logger="app.bot.user_ws")

    events = await collect(make_client())

    assert events == []
    assert call_count == 2, "must reconnect after a generic error"
    err_records = [r for r in caplog.records if "unexpected error; reconnecting" in r.getMessage()]
    assert err_records and err_records[0].exc_info is not None
