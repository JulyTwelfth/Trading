"""Line coverage for app.bot.monitor (dead code: imported nowhere, tested directly).

Covers: build_auth (dict shape + digit timestamp), is_fill_event (all branches),
wait_for_fill (REST fallback return, WS-fill return, ConnectionClosed -> continue),
and wait_for_fills (both ids awaited + the timeout path).

No real socket is opened: the WS branch monkeypatches monitor.websockets.connect
with a fake async context manager. REST get_order is a sync MagicMock invoked via
asyncio.to_thread, so we script it with side_effect.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
import websockets

from app.bot import monitor as monitor_mod
from app.bot.monitor import build_auth, is_fill_event, wait_for_fill, wait_for_fills

# build_hmac_signature base64-decodes the secret, so it must be valid base64.
VALID_B64_SECRET = "c3VwZXJzZWNyZXRrZXk="  # base64("supersecretkey")


def make_client(get_order_side_effect=None) -> MagicMock:
    client = MagicMock()
    client.creds.api_key = "k"
    client.creds.api_secret = VALID_B64_SECRET
    client.creds.api_passphrase = "p"
    if get_order_side_effect is not None:
        client.get_order = MagicMock(side_effect=get_order_side_effect)
    return client


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


# --------------------------------------------------------------------------- #
# build_auth
# --------------------------------------------------------------------------- #
def test_build_auth_shape(monkeypatch):
    monkeypatch.setattr(monitor_mod, "build_hmac_signature", lambda *a, **k: "sig")
    client = make_client()

    auth = build_auth(client)

    assert auth["apiKey"] == "k"
    assert auth["secret"] == VALID_B64_SECRET
    assert auth["passphrase"] == "p"
    assert auth["signature"] == "sig"
    assert auth["timestamp"].isdigit()
    assert set(auth) == {"apiKey", "secret", "passphrase", "timestamp", "signature"}


def test_build_auth_runs_real_signer():
    # Let the real build_hmac_signature run to cover the un-mocked call path.
    client = make_client()
    auth = build_auth(client)
    assert isinstance(auth["signature"], str) and auth["signature"]
    assert auth["timestamp"].isdigit()


# --------------------------------------------------------------------------- #
# is_fill_event
# --------------------------------------------------------------------------- #
def test_is_fill_event_non_trade_type():
    assert is_fill_event({"event_type": "book", "status": "MATCHED"}, "oid") is False


def test_is_fill_event_status_not_matched():
    assert is_fill_event({"event_type": "trade", "status": "MINED"}, "oid") is False


def test_is_fill_event_taker_match():
    event = {"event_type": "trade", "status": "MATCHED", "taker_order_id": "oid"}
    assert is_fill_event(event, "oid") is True


def test_is_fill_event_maker_match():
    event = {
        "event_type": "trade",
        "status": "MATCHED",
        "taker_order_id": "other",
        "maker_orders": [{"order_id": "x"}, {"order_id": "oid"}],
    }
    assert is_fill_event(event, "oid") is True


def test_is_fill_event_no_match():
    event = {
        "event_type": "trade",
        "status": "MATCHED",
        "taker_order_id": "other",
        "maker_orders": [{"order_id": "x"}],
    }
    assert is_fill_event(event, "oid") is False


def test_is_fill_event_maker_orders_none():
    # `event.get("maker_orders") or []` must tolerate a None maker_orders.
    event = {"event_type": "trade", "status": "MATCHED", "taker_order_id": "other"}
    assert is_fill_event(event, "oid") is False


# --------------------------------------------------------------------------- #
# wait_for_fill
# --------------------------------------------------------------------------- #
async def test_wait_for_fill_rest_fallback_returns(monkeypatch):
    # REST get_order reports MATCHED on the first poll -> return before touching WS.
    connect_called = False

    def fake_connect(*a, **k):
        nonlocal connect_called
        connect_called = True
        raise AssertionError("WS must not be used when REST already shows MATCHED")

    monkeypatch.setattr(monitor_mod.websockets, "connect", fake_connect)
    client = make_client(get_order_side_effect=[{"status": "MATCHED"}])

    await wait_for_fill(client, "oid")

    assert client.get_order.call_count == 1
    assert connect_called is False


async def test_wait_for_fill_ws_branch_returns(monkeypatch):
    # REST not MATCHED first -> open WS -> a matching fill frame ends the wait.
    client = make_client(get_order_side_effect=[{"status": "MINED"}])
    fill_frame = {"event_type": "trade", "status": "MATCHED", "taker_order_id": "oid"}
    conn = FakeConnection([json.dumps(fill_frame)])

    monkeypatch.setattr(monitor_mod.websockets, "connect", lambda *a, **k: conn)

    await wait_for_fill(client, "oid")

    # Subscribe was sent with the user auth.
    sent = json.loads(conn.sent[0])
    assert sent["type"] == "user"
    assert sent["auth"]["apiKey"] == "k"


async def test_wait_for_fill_ws_ignores_non_matching_then_rest_matches(monkeypatch):
    # WS yields a list of non-matching frames (covers list-normalization + no-match),
    # the iterator ends, loop repeats, and REST now reports MATCHED to terminate.
    client = make_client(get_order_side_effect=[{"status": "MINED"}, {"status": "MATCHED"}])
    non_matching = [{"event_type": "trade", "status": "MATCHED", "taker_order_id": "other"}]
    conns = [FakeConnection([json.dumps(non_matching)]), FakeConnection([])]
    it = iter(conns)
    monkeypatch.setattr(monitor_mod.websockets, "connect", lambda *a, **k: next(it))

    await wait_for_fill(client, "oid")

    assert client.get_order.call_count == 2


async def test_wait_for_fill_connection_closed_continues(monkeypatch):
    # WS connect raises ConnectionClosed -> `continue` -> next loop, REST now MATCHED.
    client = make_client(get_order_side_effect=[{"status": "MINED"}, {"status": "MATCHED"}])
    call_count = 0

    def fake_connect(*a, **k):
        nonlocal call_count
        call_count += 1
        raise websockets.ConnectionClosed(None, None)

    monkeypatch.setattr(monitor_mod.websockets, "connect", fake_connect)

    await wait_for_fill(client, "oid")

    assert call_count == 1
    assert client.get_order.call_count == 2


# --------------------------------------------------------------------------- #
# wait_for_fills
# --------------------------------------------------------------------------- #
async def test_wait_for_fills_awaits_both(monkeypatch):
    seen = []

    async def fake_wait(client, order_id):
        seen.append(order_id)

    monkeypatch.setattr(monitor_mod, "wait_for_fill", AsyncMock(side_effect=fake_wait))

    await wait_for_fills(MagicMock(), MagicMock(), "oid-a", "oid-b", timeout=1.0)

    assert set(seen) == {"oid-a", "oid-b"}


async def test_wait_for_fills_times_out(monkeypatch):
    async def slow_wait(client, order_id):
        await asyncio.sleep(10)

    monkeypatch.setattr(monitor_mod, "wait_for_fill", AsyncMock(side_effect=slow_wait))

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await wait_for_fills(MagicMock(), MagicMock(), "oid-a", "oid-b", timeout=0.01)
