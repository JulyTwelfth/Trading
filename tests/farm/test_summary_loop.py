import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.websockets import WebSocketState

from app.api.farm.messages import FarmPositionsEvent, FarmSummaryEvent
from app.farm import worker


class StopLoop(Exception):
    """Sentinel to break out of the otherwise-infinite summary loop."""


@pytest.fixture(autouse=True)
def _stub_balance(monkeypatch):
    """The loop now fetches the live wallet balance each push — stub the RPC call so
    no test in this module ever touches the network."""
    monkeypatch.setattr(worker, "get_balance", AsyncMock(return_value=Decimal("92.5")))


async def test_push_summary_loop_emits_valid_farm_summary(farm_state, monkeypatch):
    # Regression: push_summary_loop omitted session_loss/max_session_loss, both
    # of which FarmSummaryEvent requires. The event raised ValidationError on the
    # first iteration — before any send — so no farm_summary ever reached the UI.
    farm_state.total_volume = Decimal("250")
    farm_state.rewards_earned = Decimal("4")
    farm_state.session_loss = Decimal("-2")

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    # Stop after the first successful send (the loop is `while True`).
    async def stop(seconds):
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    # Each tick sends two events: farm_summary first, then farm_positions.
    assert ws.send_json.await_count == 2
    payload = ws.send_json.await_args_list[0].args[0]
    assert payload["type"] == "farm_summary"
    assert payload["session_loss"] == "-2"
    assert payload["max_session_loss"] == "5"
    # And it round-trips back into the model (no missing/invalid fields).
    FarmSummaryEvent.model_validate(payload)


async def test_push_summary_loop_includes_expected_rate(farm_state, monkeypatch):
    # The summary carries the expected reward rate (percentages × pool) so the UI
    # can show $/hr and a 24h projection. per_hour is exactly the day total / 24,
    # and the realized session field is gone.
    farm_state.expected_rewards_per_day = Decimal("24")

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    async def stop(seconds):
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    payload = ws.send_json.await_args_list[0].args[0]
    assert payload["rewards_per_day"] == "24"
    assert payload["rewards_per_hour"] == "1"  # 24 / 24
    assert "session_rewards" not in payload
    FarmSummaryEvent.model_validate(payload)


async def test_push_summary_loop_emits_positions_snapshot(farm_state, monkeypatch):
    # The loop also pushes an authoritative per-market positions snapshot built
    # from state.positions (not reconstructed from order events), so the UI never
    # lists stale/closed markets. Held shares are marked to the LIVE BBA mid the
    # position tracks (not the frozen pos.market.midpoint).
    pos = farm_state.positions["market-A"]
    pos.last_best_bid = Decimal("0.58")
    pos.last_best_ask = Decimal("0.62")  # mid 0.60 → 100 YES @ cost 50 → +10

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    async def stop(seconds):
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    # Second event of the tick is the positions snapshot.
    assert ws.send_json.await_count == 2
    payload = ws.send_json.await_args_list[1].args[0]
    assert payload["type"] == "farm_positions"

    entry = payload["positions"][0]
    assert entry["market_id"] == "market-A"
    assert Decimal(entry["yes_price"]) == Decimal("0.5")
    assert Decimal(entry["capital_deployed"]) > 0
    assert Decimal(entry["midpoint"]) == Decimal("0.6")
    assert Decimal(entry["unrealized_pnl"]) == Decimal("10")
    # Round-trips back into the model (no missing/invalid fields).
    FarmPositionsEvent.model_validate(payload)


async def test_push_summary_loop_includes_wallet_balance(farm_state, monkeypatch):
    # The summary carries the live pUSDC balance of the connected wallet (fetched via
    # RPC each push) so the UI can display it beside rewards earned.
    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    async def stop(seconds):
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    payload = ws.send_json.await_args_list[0].args[0]
    assert payload["wallet_balance"] == "92.5"
    FarmSummaryEvent.model_validate(payload)


async def test_push_summary_loop_balance_failure_reuses_last_value(farm_state, monkeypatch):
    # An RPC hiccup must not kill the loop or zero the display: the first push after
    # a failure re-sends the last known balance; before any success it sends None.
    calls = {"n": 0}

    async def flaky(wallet):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rpc down")
        return Decimal("77")

    monkeypatch.setattr(worker, "get_balance", flaky)

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    sleeps = {"n": 0}

    async def stop_after_two(seconds):
        sleeps["n"] += 1
        if sleeps["n"] >= 2:
            raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop_after_two)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    # Push 1: fetch failed, no prior value -> None (serialized as null).
    first = ws.send_json.await_args_list[0].args[0]
    assert first["wallet_balance"] is None
    # Push 2: fetch succeeded -> live value.
    second = ws.send_json.await_args_list[2].args[0]
    assert second["wallet_balance"] == "77"


async def test_push_summary_loop_slow_balance_fetch_does_not_stall(farm_state, monkeypatch):
    # A slow/hanging RPC must not hold up the push cadence: the fetch is hard-capped by
    # BALANCE_FETCH_TIMEOUT_SECONDS, after which the push ships the last known value
    # instead of arriving late with a fresh one.
    async def slow_balance(wallet):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        loop.call_later(0.5, lambda: fut.cancelled() or fut.set_result(Decimal("42")))
        return await fut

    monkeypatch.setattr(worker, "get_balance", slow_balance)
    monkeypatch.setattr(worker, "BALANCE_FETCH_TIMEOUT_SECONDS", 0.01, raising=False)

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    async def stop(seconds):
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", stop)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    payload = ws.send_json.await_args_list[0].args[0]
    assert payload["wallet_balance"] is None


async def test_push_summary_loop_sleep_drift_corrects_for_fetch_time(farm_state, monkeypatch):
    # A slow balance read must come out of the sleep, not stretch the push period:
    # with a ~0.2s fetch the loop sleeps ~4.8s so the cadence stays ~5s.
    async def slow_balance(wallet):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        loop.call_later(0.2, lambda: fut.cancelled() or fut.set_result(Decimal("42")))
        return await fut

    monkeypatch.setattr(worker, "get_balance", slow_balance)

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    slept = []

    async def record_and_stop(seconds):
        slept.append(seconds)
        raise StopLoop

    monkeypatch.setattr(worker.asyncio, "sleep", record_and_stop)

    with pytest.raises(StopLoop):
        await worker.push_summary_loop(ws, farm_state)

    # ~0.2s of fetch (plus a little overhead slack) deducted from the 5s interval.
    assert slept[0] <= worker.SUMMARY_INTERVAL_SECONDS - 0.19
    assert slept[0] >= worker.SUMMARY_INTERVAL_SECONDS - 1


async def test_push_summary_loop_stops_when_killed(farm_state, monkeypatch):
    # Once the kill switch trips, the loop must return immediately — no stale
    # summary/positions may follow farm_killed and repaint a dead farm as live.
    farm_state.killed = True

    ws = MagicMock()
    ws.application_state = WebSocketState.CONNECTED
    ws.send_json = AsyncMock()

    async def boom(seconds):
        raise AssertionError("killed loop should return before sleeping")

    monkeypatch.setattr(worker.asyncio, "sleep", boom)

    await worker.push_summary_loop(ws, farm_state)
    ws.send_json.assert_not_awaited()
