"""POL-54: handle_bba samples the YES leg, blacklists on a volatile sequence, cancels
resting orders, and then gates further requoting."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.farm import requote as requote_mod
from app.farm.requote import handle_bba
from app.farm.schemas import FarmState, MarketHealth
from app.farm.volatility import is_blacklisted


@pytest.fixture
def stub_network(monkeypatch):
    cancelled: list = []
    requotes: list = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    async def fake_cancel_orders(client, *ids):
        # Tier 2: the vol-pull now batches via cancel_orders; funnel its ids into the SAME
        # recorder so the existing {yes_oid, no_oid} assertion holds.
        cancelled.extend(i for i in ids if i)

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        requotes.append((outcome, float(price)))

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return {"cancelled": cancelled, "requotes": requotes}


def make_bba(asset_id: str, mid: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=mid - Decimal("0.01"),
        best_ask=mid + Decimal("0.01"),
        spread=Decimal("0.02"),
        timestamp="2026-06-01T12:00:00Z",
    )


async def test_volatile_yes_sequence_blacklists_and_cancels(farm_state: FarmState, stub_network):
    pos = farm_state.positions["market-A"]
    yes = pos.market.yes_token_id

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.50")))
    yes_oid, no_oid = pos.yes_order_id, pos.no_order_id
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, make_bba(yes, Decimal("0.53")))

    assert is_blacklisted(farm_state, "market-A") is True
    # On trigger, both resting legs are cancelled so nothing else can fill.
    assert set(stub_network["cancelled"]) == {yes_oid, no_oid}


async def test_blacklisted_market_does_not_requote(farm_state: FarmState, stub_network):
    farm_state.health["market-A"] = MarketHealth(
        blacklist_until=datetime.now(timezone.utc) + timedelta(minutes=15)
    )
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.40")  # would normally requote (out of zone)

    await handle_bba(
        MagicMock(),
        farm_state,
        AsyncMock(),
        pos,
        make_bba(pos.market.yes_token_id, Decimal("0.50")),
    )

    assert stub_network["requotes"] == [], "blacklisted market must not requote"


async def test_no_leg_event_does_not_sample(farm_state: FarmState, stub_network):
    pos = farm_state.positions["market-A"]
    no = pos.market.no_token_id
    # A big move on the NO leg must not feed the (YES-keyed) volatility buffer.
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.20")))
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, make_bba(no, Decimal("0.80")))
    assert is_blacklisted(farm_state, "market-A") is False
