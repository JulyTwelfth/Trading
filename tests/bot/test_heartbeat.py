"""Heartbeat loop behavior.

- POL-36 Bug 4: heartbeat_id must rotate proactively, not just react to server
  invalidation. After rotation_interval seconds elapse since the last reset, the
  next heartbeat must send "" so the server issues a fresh ID.
- 400 recovery: the server rejects an invalid/expired heartbeat_id (including ""
  mid-session) with a 400 whose body carries the correct id. The loop must adopt
  that id and retry quickly instead of resetting to "" (which loops forever).
- M2: heartbeat_loop now calls client.send_heartbeat(heartbeat_id) directly (the
  adapter method) instead of the module-level send_heartbeat(client, hb) helper.
  Tests therefore mock client.send_heartbeat, not heartbeat_mod.send_heartbeat.
- M2: heartbeat_loop early-returns when client.supports_heartbeat is False.
"""

import asyncio
from unittest.mock import MagicMock

import httpx
import pytest
from py_clob_client_v2.exceptions import PolyApiException

from app.bot import heartbeat as heartbeat_mod
from app.bot.heartbeat import corrected_heartbeat_id, heartbeat_loop


def rejection(heartbeat_id: str) -> PolyApiException:
    """The server's 400 for an invalid heartbeat_id, corrected id in the body."""
    body = {"heartbeat_id": heartbeat_id, "error_msg": "Invalid Heartbeat ID"}
    return PolyApiException(httpx.Response(400, json=body))


def scripted_client(actions: list, *, stop_after: int):
    """Build (client, calls, sleeps, fake_time) where client.send_heartbeat plays
    `actions` in order (dict → return, Exception → raise) and asyncio.sleep advances
    fake_time and cancels the loop after `stop_after` sends."""
    calls: list[str] = []
    sleeps: list[float] = []
    fake_time = [0.0]

    async def fake_send(heartbeat_id=""):
        action = actions[len(calls)]
        calls.append(heartbeat_id)
        if isinstance(action, Exception):
            raise action
        return action

    async def fake_sleep(secs):
        sleeps.append(secs)
        fake_time[0] += secs
        if len(calls) >= stop_after:
            raise asyncio.CancelledError()

    client = MagicMock()
    client.send_heartbeat = fake_send
    client.supports_heartbeat = True
    return client, calls, sleeps, fake_time, fake_sleep


async def test_heartbeat_rotates_proactively_after_interval(monkeypatch):
    calls: list[str] = []
    fake_time = [0.0]

    async def fake_send(heartbeat_id=""):
        calls.append(heartbeat_id)
        return {"heartbeat_id": "server-id"}

    async def fake_sleep(secs):
        fake_time[0] += secs
        if len(calls) >= 5:
            raise asyncio.CancelledError()

    client = MagicMock()
    client.send_heartbeat = fake_send
    client.supports_heartbeat = True
    monkeypatch.setattr(heartbeat_mod.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(heartbeat_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        # interval=30s, rotation_interval=100s
        # Expected: send(""), send("server-id") x3 (t=30,60,90), send("") at t=120 (rotation).
        await heartbeat_loop(client, interval=30.0, rotation_interval=100.0)

    assert calls[0] == "", "first call must bootstrap with empty id"
    assert calls[1:4] == ["server-id"] * 3, (
        f"echoed server id while under rotation_interval; got {calls[1:4]}"
    )
    assert calls[4] == "", f"proactive rotation must reset to empty at t=120 (>=100); got {calls}"


async def test_heartbeat_does_not_rotate_before_interval(monkeypatch):
    calls: list[str] = []
    fake_time = [0.0]

    async def fake_send(heartbeat_id=""):
        calls.append(heartbeat_id)
        return {"heartbeat_id": "server-id"}

    async def fake_sleep(secs):
        fake_time[0] += secs
        if len(calls) >= 4:
            raise asyncio.CancelledError()

    client = MagicMock()
    client.send_heartbeat = fake_send
    client.supports_heartbeat = True
    monkeypatch.setattr(heartbeat_mod.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(heartbeat_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        # interval=30s, rotation_interval=10000s → never rotates in this run.
        await heartbeat_loop(client, interval=30.0, rotation_interval=10000.0)

    assert calls[0] == ""
    assert calls[1:] == ["server-id"] * 3, f"must NOT rotate before interval; got {calls[1:]}"


async def test_heartbeat_adopts_corrected_id_after_400(monkeypatch):
    """A 400 carries the correct heartbeat_id; the next beat must send it
    (not "") after the short retry delay, then resume the normal cadence."""
    client, calls, sleeps, fake_time, fake_sleep = scripted_client(
        [
            rejection("corrected-1"),
            {"heartbeat_id": "server-2"},
            {"heartbeat_id": "server-3"},
        ],
        stop_after=3,
    )
    monkeypatch.setattr(heartbeat_mod.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(heartbeat_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await heartbeat_loop(client, interval=30.0, rotation_interval=10000.0, retry_delay=1.0)

    assert calls == ["", "corrected-1", "server-2"]
    assert sleeps[0] == 1.0, "rejected beat must retry after the short delay"
    assert sleeps[1] == 30.0, "successful beat must resume the normal interval"


async def test_heartbeat_rebootstraps_after_400_without_corrected_id(monkeypatch):
    """A 400 whose body has no usable heartbeat_id falls back to "" (bootstrap)."""
    client, calls, sleeps, fake_time, fake_sleep = scripted_client(
        [
            {"heartbeat_id": "server-1"},
            PolyApiException(httpx.Response(400, text="Bad Request")),
            {"heartbeat_id": "server-2"},
        ],
        stop_after=3,
    )
    monkeypatch.setattr(heartbeat_mod.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(heartbeat_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await heartbeat_loop(client, interval=30.0, rotation_interval=10000.0, retry_delay=1.0)

    assert calls == ["", "server-1", ""]
    assert sleeps[1] == 1.0, "400 must retry after the short delay"


async def test_heartbeat_keeps_id_on_transient_error(monkeypatch):
    """Network/5xx failures keep the current id (likely still valid) and wait
    the full interval; if it expired, the next 400 recovers via adoption."""
    client, calls, sleeps, fake_time, fake_sleep = scripted_client(
        [
            {"heartbeat_id": "server-1"},
            PolyApiException(error_msg="Request exception!"),  # status_code=None
            {"heartbeat_id": "server-2"},
        ],
        stop_after=3,
    )
    monkeypatch.setattr(heartbeat_mod.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(heartbeat_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await heartbeat_loop(client, interval=30.0, rotation_interval=10000.0, retry_delay=1.0)

    assert calls == ["", "server-1", "server-1"], "transient error must not drop the id"
    assert sleeps[1] == 30.0, "transient error must wait the full interval"


async def test_heartbeat_rotation_rejection_recovers_without_churn(monkeypatch):
    """If the server rejects the proactive-rotation "" (session still active),
    adopting the corrected id must also reset the rotation timer — otherwise the
    rotation condition re-fires every beat and the loop churns ""→400 forever."""
    client, calls, sleeps, fake_time, fake_sleep = scripted_client(
        [
            {"heartbeat_id": "server-1"},  # t=0: bootstrap, rotation clock starts
            {"heartbeat_id": "server-1"},  # t=30
            {"heartbeat_id": "server-1"},  # t=60
            {"heartbeat_id": "server-1"},  # t=90
            rejection("corrected-9"),  # t=120: rotation sends "", server rejects
            {"heartbeat_id": "server-9"},  # t=121: corrected id accepted
            {"heartbeat_id": "server-9"},  # t=151: must NOT re-rotate to ""
        ],
        stop_after=7,
    )
    monkeypatch.setattr(heartbeat_mod.time, "monotonic", lambda: fake_time[0])
    monkeypatch.setattr(heartbeat_mod.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        await heartbeat_loop(client, interval=30.0, rotation_interval=100.0, retry_delay=1.0)

    assert calls == ["", "server-1", "server-1", "server-1", "", "corrected-9", "server-9"]
    assert sleeps[4] == 1.0, "rejected rotation must retry after the short delay"


async def test_heartbeat_disabled_for_non_supporting_client(monkeypatch):
    """M2: when supports_heartbeat is False, heartbeat_loop returns immediately
    without ever calling send_heartbeat."""
    client = MagicMock()
    client.supports_heartbeat = False
    client.wallet_type = "DEPOSIT_WALLET"
    send_calls = []

    async def spy_send(heartbeat_id=""):
        send_calls.append(heartbeat_id)
        return {}

    client.send_heartbeat = spy_send

    await heartbeat_loop(client)  # must return, not loop forever

    assert send_calls == [], "send_heartbeat must not be called for supports_heartbeat=False"


def test_corrected_heartbeat_id_extraction():
    assert corrected_heartbeat_id(rejection("abc")) == "abc"
    # Non-JSON body → error_msg is a string → no corrected id.
    assert corrected_heartbeat_id(PolyApiException(httpx.Response(400, text="nope"))) == ""
    # JSON body without a heartbeat_id, or with a null one → no corrected id.
    assert corrected_heartbeat_id(PolyApiException(httpx.Response(400, json={"a": 1}))) == ""
    assert (
        corrected_heartbeat_id(PolyApiException(httpx.Response(400, json={"heartbeat_id": None})))
        == ""
    )
