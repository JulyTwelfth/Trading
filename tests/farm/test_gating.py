"""Tier 2 — the unified should_quote() gate.

Before this, the requote path (handle_bba, requote_leg) checked only is_paused + is_blacklisted,
while the reconcile/open path checked all the protective flags — so an event-quarantined (F8) or
post-kill (F9) market could still be re-quoted. quote_block_reason() is the single predicate every
quote site now consults.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.farm import requote as requote_mod
from app.farm.gating import quote_block_reason, should_quote
from app.farm.requote import handle_bba
from app.farm.schemas import FarmState, MarketHealth

FUTURE = datetime.now(timezone.utc) + timedelta(minutes=10)


def test_clear_market_is_quotable(farm_state: FarmState):
    assert quote_block_reason(farm_state, "market-A", "") is None
    assert should_quote(farm_state, farm_state.positions["market-A"]) is True


def test_killed_blocks(farm_state: FarmState):
    farm_state.killed = True
    assert quote_block_reason(farm_state, "market-A", "") == "killed"


def test_paused_blocks(farm_state: FarmState):
    farm_state.health["market-A"] = MarketHealth(paused_until=FUTURE)
    assert quote_block_reason(farm_state, "market-A", "") == "paused"


def test_blacklisted_blocks(farm_state: FarmState):
    farm_state.health["market-A"] = MarketHealth(blacklist_until=FUTURE)
    assert quote_block_reason(farm_state, "market-A", "") == "blacklisted"


def test_excluded_market_blocks(farm_state: FarmState):
    farm_state.excluded_markets.add("market-A")
    assert quote_block_reason(farm_state, "market-A", "") == "excluded_market"


def test_excluded_event_blocks(farm_state: FarmState):
    farm_state.excluded_events["evt-1"] = FUTURE
    assert quote_block_reason(farm_state, "market-A", "evt-1") == "excluded_event"
    # An empty event_slug must never match the exclusion map.
    assert quote_block_reason(farm_state, "market-A", "") is None


# ── F8/F9 integration: handle_bba must not requote a quarantined market ────────


@pytest.fixture
def stub(monkeypatch):
    requotes: list = []

    async def fake_cancel(client, oid):
        return None

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        requotes.append((outcome, float(price)))

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    return requotes


def threatening_frame(pos) -> BestBidAsk:
    # best_bid == our price → "threatened" → handle_bba would normally requote.
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=pos.market.yes_token_id,
        best_bid=Decimal("0.50"),
        best_ask=Decimal("0.52"),
        spread=Decimal("0.02"),
        timestamp="2026-06-17T00:00:00Z",
    )


async def test_handle_bba_requotes_clear_market(farm_state: FarmState, stub):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")  # isolate from the mark-to-market kill

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, threatening_frame(pos))

    assert stub, "a clear, threatened market should requote"


async def test_handle_bba_skips_event_excluded_market(farm_state: FarmState, stub):
    # F8: a sibling under a quarantined event family must NOT be re-quoted by the market-WS path.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.market.event_slug = "evt-1"
    farm_state.excluded_events["evt-1"] = FUTURE

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, threatening_frame(pos))

    assert stub == [], "event-excluded market must not be re-quoted (F8)"


async def test_handle_bba_skips_killed_session(farm_state: FarmState, stub):
    # F9: once killed, the market-WS path must not place new quotes.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    farm_state.killed = True

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, threatening_frame(pos))

    assert stub == [], "killed session must not be re-quoted (F9)"
