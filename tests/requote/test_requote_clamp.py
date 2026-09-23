from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BestBidAsk
from app.farm import requote as requote_mod
from app.farm.requote import handle_bba
from app.farm.schemas import FarmState, MarketHealth


@pytest.fixture
def stub_network(monkeypatch):
    place_calls: list = []

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append((order.token_id, order.side, order.size, order.price, post_only))
        return f"oid-{len(place_calls)}"

    async def fake_cancel_order(client, oid):
        return None

    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel_order)
    return place_calls


def stub_compute_quote(monkeypatch, bid: Decimal, ask: Decimal | None = None):
    """Force compute_quote to return a specific (bid, ask) so we can drive
    handle_bba into the clamp path regardless of the market's actual quote params."""
    ask = ask if ask is not None else bid + Decimal("0.02")

    def fake_compute_quote(market, midpoint, quote_depth="safe"):
        return (bid, ask)

    monkeypatch.setattr(requote_mod, "compute_quote", fake_compute_quote)


def make_bba(asset_id: str, best_bid: Decimal, best_ask: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        timestamp="2026-05-19T12:00:00Z",
    )


async def test_handle_bba_updates_last_best_bid_ask(farm_state: FarmState, stub_network: list):
    pos = farm_state.positions["market-A"]
    bba = make_bba(pos.market.yes_token_id, Decimal("0.49"), Decimal("0.51"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert pos.last_best_bid == Decimal("0.49")
    assert pos.last_best_ask == Decimal("0.51")


async def test_requote_clamps_when_quote_would_cross_book(
    farm_state: FarmState, stub_network: list, monkeypatch
):
    # Force compute_quote to return a bid AT best_ask — would cross without clamp.
    stub_compute_quote(monkeypatch, bid=Decimal("0.50"))
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.40")  # out_of_zone vs midpoint, triggers requote
    bba = make_bba(pos.market.yes_token_id, best_bid=Decimal("0.49"), best_ask=Decimal("0.50"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network, "expected a placement after clamping"
    _, _, _, placed_price, post_only = stub_network[0]
    # Clamp must bring price strictly below best_ask. tick=0.01 → 0.49.
    assert placed_price == 0.49
    assert post_only is True


async def test_requote_skips_when_clamp_would_zero(
    farm_state: FarmState, stub_network: list, monkeypatch
):
    # best_ask=0.01, tick=0.01 → safe_price = best_ask - tick = 0 → degenerate, skip.
    # Force out_of_zone so hysteresis doesn't short-circuit before the clamp.
    stub_compute_quote(monkeypatch, bid=Decimal("0.50"))
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.20")
    bba = make_bba(pos.market.yes_token_id, best_bid=Decimal("0"), best_ask=Decimal("0.01"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network == []


async def test_safe_depth_blocks_upward_retighten(
    farm_state: FarmState, stub_network: list, monkeypatch
):
    # safe: every in-zone tick scores ~equally, so an upward (toward-mid) move is refused
    # to avoid thrash. our_price 0.48, target 0.49, in-zone, threatened → must NOT requote.
    farm_state.config.quote_depth = "safe"
    stub_compute_quote(monkeypatch, bid=Decimal("0.49"))
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.48")
    bba = make_bba(pos.market.yes_token_id, best_bid=Decimal("0.49"), best_ask=Decimal("0.51"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network == [], "safe must not move up toward mid while in zone"


async def test_aggressive_depth_allows_upward_retighten(
    farm_state: FarmState, stub_network: list, monkeypatch
):
    # aggressive/normal: ticks closer to mid score strictly higher, so re-tightening upward
    # toward the target IS allowed (otherwise the leg ratchets to the edge and stays there).
    farm_state.config.quote_depth = "aggressive"
    stub_compute_quote(monkeypatch, bid=Decimal("0.49"))
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.48")
    bba = make_bba(pos.market.yes_token_id, best_bid=Decimal("0.49"), best_ask=Decimal("0.51"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network, "aggressive must re-tighten upward toward mid"
    _, _, _, placed_price, _ = stub_network[0]
    assert placed_price == 0.49


async def test_downward_move_when_threatened_works_for_all_depths(
    farm_state: FarmState, stub_network: list, monkeypatch
):
    # Safety invariant: a downward (toward-safety) requote is never blocked, even for safe.
    farm_state.config.quote_depth = "safe"
    stub_compute_quote(monkeypatch, bid=Decimal("0.47"))
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.49")
    bba = make_bba(pos.market.yes_token_id, best_bid=Decimal("0.50"), best_ask=Decimal("0.52"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert stub_network, "downward safety requote must always fire"


async def test_handle_bba_skips_when_market_paused(farm_state: FarmState, stub_network: list):
    pos = farm_state.positions["market-A"]
    pos.yes_price = Decimal("0.40")
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=1)
    )
    bba = make_bba(pos.market.yes_token_id, Decimal("0.49"), Decimal("0.51"))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    # last_best_bid/ask updated even while paused (cheap state, useful elsewhere).
    assert pos.last_best_bid == Decimal("0.49")
    assert stub_network == []
