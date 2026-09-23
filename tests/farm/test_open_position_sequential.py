"""Regression for the [Errno 11] EAGAIN partial opens: the two order legs must be
placed one at a time over the CLOB client's single sync socket, never concurrently.

A concurrency probe (max in-flight place calls) distinguishes the fix from the old
asyncio.gather: gather starts both coroutines before either yields, so max in-flight
reaches 2; the sequential placement keeps it at 1."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market
from app.farm.worker import open_position


async def test_open_position_places_legs_sequentially(
    farm_state: FarmState, market: Market, monkeypatch
):
    farm_state.positions.clear()
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}

    in_flight = 0
    max_in_flight = 0
    order_seen: list[str] = []

    async def fake_place(client, order, post_only=False):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        order_seen.append(order.token_id)
        # Yield control: a concurrent gather would let the second leg enter here
        # before the first decrements, pushing max_in_flight to 2.
        await asyncio.sleep(0)
        in_flight -= 1
        return "yes-oid" if order.token_id == market.yes_token_id else "no-oid"

    async def fake_cancel(client, oid):
        pass

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", fake_cancel)

    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)

    assert max_in_flight == 1, f"legs overlapped (max in-flight={max_in_flight}); must be serial"
    # Both legs still attempted, and the position is saved on success.
    assert order_seen == [market.yes_token_id, market.no_token_id]
    assert market.condition_id in farm_state.positions


async def test_open_position_propagates_cancellation(
    farm_state: FarmState, market: Market, monkeypatch
):
    """CancelledError must NOT be swallowed by place_leg's `except Exception` — it has to
    propagate so farm shutdown tears the leg down cleanly. Guards against someone widening
    the catch to BaseException (which would absorb cancellation into a partial failure)."""
    farm_state.positions.clear()
    midpoints = {market.yes_token_id: Decimal("0.5"), market.no_token_id: Decimal("0.5")}

    async def fake_place(client, order, post_only=False):
        raise asyncio.CancelledError()

    async def cancel_explodes(client, oid):
        raise AssertionError("rollback cancel must not run when cancellation propagates")

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", cancel_explodes)

    with pytest.raises(asyncio.CancelledError):
        await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)

    assert market.condition_id not in farm_state.positions
