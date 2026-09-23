"""End-to-end 2026-06-15 incident regression.

A YES maker fill landed on `will-any-ai-model-reach-1520-...`; the market should have
been quarantined the instant fill #1 processed. Instead the sibling/re-quoted leg stayed
live and the book crashed (mid 0.62 → 0.18 in ~3s), double-filling the market for ~$20.

This drives the real two-task sequence end to end:
  1. handle_trade (user-WS) processes fill #1 → fill-blacklist + pull every resting order.
  2. handle_bba (market-WS) then sees the crashing frames → must NOT requote (the
     fill-blacklist gates it), so there is no second order for the crash to hit.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk, UserTrade, UserTradeMakerOrder
from app.farm import fills as fills_mod
from app.farm import requote as requote_mod
from app.farm.fills import handle_trade
from app.farm.requote import handle_bba
from app.farm.schemas import FarmState
from app.farm.volatility import is_blacklisted


@pytest.fixture
def stub_io(monkeypatch):
    """Record requote attempts; swallow all cancels. The fill path and the BBA path both
    funnel cancels through their own module's cancel_order, so stub both."""
    requotes: list = []

    async def fake_cancel(client, oid):
        return None

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        requotes.append((outcome, float(price)))

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return {"requotes": requotes}


def fill_one(pos) -> UserTrade:
    return UserTrade(
        event_type="trade",
        id="trade-crash-1",
        asset_id=pos.market.yes_token_id,
        market="market-A",
        side="BUY",
        price=Decimal("0.38"),
        size=Decimal("50"),
        outcome="YES",
        status="MATCHED",
        timestamp="2026-06-15T22:30:51Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id=pos.market.yes_token_id,
                order_id=pos.yes_order_id,
                matched_amount=Decimal("50"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.38"),
            )
        ],
        taker_order_id="some-taker",
    )


def crashing_frame(pos, mid: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=pos.market.yes_token_id,
        best_bid=mid - Decimal("0.01"),
        best_ask=mid + Decimal("0.01"),
        spread=Decimal("0.02"),
        timestamp="2026-06-15T22:30:52Z",
    )


async def test_fill_then_crash_does_not_requote(farm_state: FarmState, stub_io):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")  # start flat; the fill books the inventory
    pos.yes_price = Decimal("0.38")

    # 1. Fill #1 lands — quarantines the market and pulls its orders.
    await handle_trade(MagicMock(), fill_one(pos), farm_state, AsyncMock())
    assert is_blacklisted(farm_state, "market-A") is True

    # 2. The book now collapses. Every subsequent market-WS frame must be a no-op for quoting.
    for mid in (Decimal("0.40"), Decimal("0.30"), Decimal("0.18")):
        await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, crashing_frame(pos, mid))

    assert stub_io["requotes"] == [], (
        "a fill-blacklisted market must never be re-quoted during the crash — that re-armed "
        "leg is the second fill that turned a ~$10 loss into ~$20"
    )
