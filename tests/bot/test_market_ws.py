"""Line coverage for app.bot.market_ws.stream_market_events.

Covers: happy-path connect + send + parse of best_bid_ask and tick_size_change
frames, list-vs-single frame normalization, non-JSON frame (JSONDecodeError ->
warning + continue), a frame that passes json but fails Pydantic validation
(logger.exception branch), the generic-Exception -> reconnect branch, and the
CancelledError termination of the outer `while True`.

No real socket is opened: we monkeypatch the module's `websockets.connect` with a
fake async context manager whose async-iterator yields scripted JSON strings, and
neutralize the reconnect delay to 0.
"""

import asyncio
import json
import logging
from decimal import Decimal

import pytest
import websockets

from app.bot import market_ws as market_ws_mod
from app.bot.market_ws import stream_market_events
from app.bot.schemas import BestBidAsk, TickSizeChange


class FakeConnection:
    """Async context manager + async iterator over a scripted list of messages."""

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


def best_bid_ask_frame() -> dict:
    return {
        "event_type": "best_bid_ask",
        "market": "0xmarket",
        "asset_id": "tok-1",
        "best_bid": "0.45",
        "best_ask": "0.55",
        "spread": "0.10",
        "timestamp": "1718000000",
    }


def tick_size_change_frame() -> dict:
    return {
        "event_type": "tick_size_change",
        "market": "0xmarket",
        "asset_id": "tok-1",
        "old_tick_size": "0.01",
        "new_tick_size": "0.001",
        "timestamp": "1718000001",
    }


def make_connect_then_cancel(monkeypatch, messages):
    """Wire websockets.connect: 1st call -> FakeConnection(messages); 2nd -> cancel."""
    holder = {"conn": None}
    call_count = 0

    def fake_connect(url, ssl=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            holder["conn"] = FakeConnection(messages)
            return holder["conn"]
        raise asyncio.CancelledError()

    monkeypatch.setattr(market_ws_mod.websockets, "connect", fake_connect)
    monkeypatch.setattr(market_ws_mod, "MARKET_WS_RECONNECT_DELAY_SECONDS", 0)
    return holder


async def collect(token_ids):
    out = []
    with pytest.raises(asyncio.CancelledError):
        async for ev in stream_market_events(token_ids):
            out.append(ev)
    return out


async def test_parses_best_bid_ask_and_tick_size_change(monkeypatch):
    messages = [
        json.dumps(best_bid_ask_frame()),
        json.dumps(tick_size_change_frame()),
    ]
    holder = make_connect_then_cancel(monkeypatch, messages)

    events = await collect(["tok-1", "tok-2"])

    assert len(events) == 2
    assert isinstance(events[0], BestBidAsk)
    assert events[0].best_bid == Decimal("0.45")
    assert events[0].best_ask == Decimal("0.55")
    assert isinstance(events[1], TickSizeChange)
    assert events[1].new_tick_size == Decimal("0.001")

    # The subscribe payload was sent and includes the token ids.
    sent = json.loads(holder["conn"].sent[0])
    assert sent["assets_ids"] == ["tok-1", "tok-2"]
    assert sent["type"] == "market"


async def test_list_frame_normalization(monkeypatch):
    # A single message that is a JSON *list* of two frames must be flattened.
    messages = [json.dumps([best_bid_ask_frame(), tick_size_change_frame()])]
    make_connect_then_cancel(monkeypatch, messages)

    events = await collect(["tok-1"])

    assert [type(e).__name__ for e in events] == ["BestBidAsk", "TickSizeChange"]


async def test_non_json_frame_warns_and_continues(monkeypatch, caplog):
    import logging

    messages = ["this-is-not-json{", json.dumps(best_bid_ask_frame())]
    make_connect_then_cancel(monkeypatch, messages)
    caplog.set_level(logging.WARNING, logger="app.bot.market_ws")

    events = await collect(["tok-1"])

    # Malformed frame skipped; the following valid frame still parsed.
    assert len(events) == 1
    assert isinstance(events[0], BestBidAsk)
    assert any("non-json frame ignored" in r.getMessage() for r in caplog.records)


async def test_invalid_best_bid_ask_logs_warning_and_skips(monkeypatch, caplog):
    import logging

    bad = best_bid_ask_frame()
    del bad["best_bid"]  # required field missing -> ValidationError
    good_tick = tick_size_change_frame()
    messages = [json.dumps(bad), json.dumps(good_tick)]
    make_connect_then_cancel(monkeypatch, messages)
    caplog.set_level(logging.WARNING, logger="app.bot.market_ws")

    events = await collect(["tok-1"])

    # Bad frame did not yield; the valid tick_size frame after it still yielded.
    assert len(events) == 1
    assert isinstance(events[0], TickSizeChange)
    # Logged at WARNING with the payload, NOT a per-frame traceback (would spam on book frames).
    warn_records = [
        r for r in caplog.records if "dropping unparseable best_bid_ask" in r.getMessage()
    ]
    assert warn_records, "expected a warning for the invalid best_bid_ask frame"
    assert warn_records[0].exc_info is None, "should not dump a traceback per dropped frame"


async def test_invalid_tick_size_change_logs_warning_and_skips(monkeypatch, caplog):
    import logging

    bad = tick_size_change_frame()
    del bad["new_tick_size"]  # required field missing -> ValidationError
    messages = [json.dumps(bad), json.dumps(best_bid_ask_frame())]
    make_connect_then_cancel(monkeypatch, messages)
    caplog.set_level(logging.WARNING, logger="app.bot.market_ws")

    events = await collect(["tok-1"])

    assert len(events) == 1
    assert isinstance(events[0], BestBidAsk)
    assert any("dropping unparseable tick_size_change" in r.getMessage() for r in caplog.records)


async def test_unknown_event_type_is_ignored(monkeypatch):
    # A frame whose event_type we don't consume (last_trade_price) hits no branch — no yield, no
    # error. (book/price_change ARE now consumed, so use a genuinely-unconsumed type here.)
    other = {"event_type": "last_trade_price", "market": "m", "asset_id": "tok-1", "price": "0.5"}
    messages = [json.dumps(other), json.dumps(best_bid_ask_frame())]
    make_connect_then_cancel(monkeypatch, messages)

    events = await collect(["tok-1"])

    assert len(events) == 1
    assert isinstance(events[0], BestBidAsk)


async def test_generic_exception_triggers_reconnect(monkeypatch, caplog):
    import logging

    # 1st connect: raise a non-ConnectionClosed Exception -> logger.exception + reconnect.
    # 2nd connect: CancelledError to break the outer loop.
    call_count = 0

    def fake_connect(url, ssl=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        raise asyncio.CancelledError()

    monkeypatch.setattr(market_ws_mod.websockets, "connect", fake_connect)
    monkeypatch.setattr(market_ws_mod, "MARKET_WS_RECONNECT_DELAY_SECONDS", 0)
    caplog.set_level(logging.ERROR, logger="app.bot.market_ws")

    events = await collect(["tok-1"])

    assert events == []
    assert call_count == 2, "must have attempted a reconnect after the generic error"
    err_records = [r for r in caplog.records if "unexpected error; reconnecting" in r.getMessage()]
    assert err_records and err_records[0].exc_info is not None


async def test_connection_closed_logs_warning_and_reconnects(monkeypatch, caplog):
    # 1st connect: ConnectionClosed -> WARNING (no stack trace) + reconnect.
    # 2nd connect: CancelledError to break the outer loop.
    call_count = 0

    def fake_connect(url, ssl=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise websockets.ConnectionClosed(None, None)
        raise asyncio.CancelledError()

    monkeypatch.setattr(market_ws_mod.websockets, "connect", fake_connect)
    monkeypatch.setattr(market_ws_mod, "MARKET_WS_RECONNECT_DELAY_SECONDS", 0)
    caplog.set_level(logging.WARNING, logger="app.bot.market_ws")

    events = await collect(["tok-1"])

    assert events == []
    assert call_count == 2, "must reconnect after a connection-closed event"
    closed = [r for r in caplog.records if "connection closed" in r.getMessage()]
    assert closed, "expected a 'connection closed' warning"
    assert all(r.levelno == logging.WARNING for r in closed)
    assert all(r.exc_info is None for r in closed), "routine close must not dump a traceback"
