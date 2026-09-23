"""The fill (user-WS task) and requote (market-WS task) run concurrently and interleave
only at awaits. requote_leg must NOT leave a resting order on a market a fill quarantined
mid-requote — that re-armed leg is what a crashing book double-filled on 2026-06-15.

Two windows, two guards:
  #1 pre-place  — market blacklisted/paused while we awaited the cancel → don't place at all.
  #2 post-place — market blacklisted/paused while we awaited the place → retract the order.
                  It's registered first, so a fill landing before our cancel still routes
                  through handle_trade and is exited (never stranded).
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import requote as requote_mod
from app.farm.health import is_paused, mark_paused
from app.farm.requote import requote_leg
from app.farm.schemas import FarmState
from app.farm.volatility import blacklist_for_fill, is_blacklisted


@pytest.fixture
def record_io(monkeypatch):
    """Record cancels and places; by default both succeed and place returns a fresh oid."""
    io = {"cancelled": [], "placed": []}

    async def fake_cancel(client, oid):
        io["cancelled"].append(oid)

    async def fake_place(client, order, post_only=False):
        io["placed"].append(order)
        return "yes-oid-v2"

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place)
    return io


async def test_requote_proceeds_when_not_blacklisted(farm_state: FarmState, record_io):
    # Control: a healthy market requotes and adopts the new oid as the live leg.
    pos = farm_state.positions["market-A"]

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))

    assert len(record_io["placed"]) == 1, "healthy market must place the replacement"
    assert pos.yes_order_id == "yes-oid-v2"
    assert "yes-oid-v2" in farm_state.order_registry


async def test_requote_aborts_pre_place_when_blacklisted_during_cancel(
    farm_state: FarmState, monkeypatch
):
    # Guard #1: the fill task blacklists the market during the cancel await → no place at all.
    placed: list = []

    async def cancel_then_blacklist(client, oid):
        # Simulate handle_trade (other task) running while we await this cancel.
        blacklist_for_fill(farm_state, "market-A")

    async def fake_place(client, order, post_only=False):
        placed.append(order)
        return "yes-oid-v2"

    monkeypatch.setattr(requote_mod, "cancel_order", cancel_then_blacklist)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place)

    pos = farm_state.positions["market-A"]
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))

    assert placed == [], "must NOT place into a market blacklisted mid-requote"
    assert pos.yes_order_id == "yes-oid", "live leg pointer must stay on the cancelled old oid"
    assert is_blacklisted(farm_state, "market-A") is True


async def test_requote_aborts_pre_place_when_paused_during_cancel(
    farm_state: FarmState, monkeypatch
):
    # Guard #1 also covers the circuit-breaker pause, mirroring handle_bba's top-level gate.
    placed: list = []

    async def cancel_then_pause(client, oid):
        mark_paused(farm_state, "market-A")

    async def fake_place(client, order, post_only=False):
        placed.append(order)
        return "yes-oid-v2"

    monkeypatch.setattr(requote_mod, "cancel_order", cancel_then_pause)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place)

    pos = farm_state.positions["market-A"]
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))

    assert placed == [], "must NOT place into a market paused mid-requote"
    assert is_paused(farm_state, "market-A") is True


async def test_requote_retracts_post_place_when_blacklisted_during_place(
    farm_state: FarmState, monkeypatch
):
    """Guard #2: blacklist lands while we await the place itself. The order is registered (so a
    fill-before-cancel is still handled) then retracted — never adopted as the live leg."""
    cancelled: list[str] = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def place_then_blacklist(client, order, post_only=False):
        # Simulate the fill task blacklisting while the place round-trips.
        blacklist_for_fill(farm_state, "market-A")
        return "yes-oid-v2"

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", place_then_blacklist)

    pos = farm_state.positions["market-A"]
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))

    assert "yes-oid-v2" in farm_state.order_registry, (
        "the placed order must be registered so a race-fill still routes to handle_trade"
    )
    assert pos.yes_order_id == "yes-oid", "a quarantined market must NOT adopt the new order"
    assert "yes-oid-v2" in cancelled, "the just-placed order must be retracted on the exchange"


async def test_requote_retracts_post_place_when_paused_during_place(
    farm_state: FarmState, monkeypatch
):
    # Guard #2 symmetric: a circuit-breaker pause during the place also retracts.
    cancelled: list[str] = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def place_then_pause(client, order, post_only=False):
        mark_paused(farm_state, "market-A")
        return "yes-oid-v2"

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", place_then_pause)

    pos = farm_state.positions["market-A"]
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))

    assert pos.yes_order_id == "yes-oid"
    assert "yes-oid-v2" in cancelled
