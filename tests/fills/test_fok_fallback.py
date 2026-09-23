from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import BookLevel, OrderBook, UserTrade
from app.farm import exits as exits_mod
from app.farm.fills import handle_trade
from app.farm.health import is_paused
from app.farm.schemas import FarmState, FokExitInfo, MarketHealth


@pytest.fixture
def stub_network(monkeypatch):
    market_calls: list = []
    limit_calls: list = []

    async def fake_place_market_order(client, token_id, side, amount):
        market_calls.append((token_id, side, amount))
        return f"fok-{len(market_calls)}"

    async def fake_place_limit_order(client, order, post_only=False):
        limit_calls.append((order.token_id, order.side, order.size, order.price, post_only))
        return f"gtc-{len(limit_calls)}"

    async def fake_cancel_order(client, oid):
        return None

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market_order)
    monkeypatch.setattr(exits_mod, "place_limit_order", fake_place_limit_order)
    # exit_position_leg calls cancel_orders when cancel_resting=True (the default).
    monkeypatch.setattr(exits_mod, "cancel_order", fake_cancel_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    return {"market": market_calls, "limit": limit_calls}


def stub_order_book(monkeypatch, best_bid: Decimal) -> list:
    """Stub get_order_book; returns a list that captures every call for assertions."""
    calls: list = []

    async def fake_get_order_book(token_id: str) -> OrderBook:
        calls.append(token_id)
        return OrderBook(
            market="market-A",
            asset_id=token_id,
            timestamp="2026-05-19T12:00:00Z",
            bids=[BookLevel(price=best_bid, size=Decimal("10"))] if best_bid > 0 else [],
            asks=[BookLevel(price=Decimal("0.55"), size=Decimal("10"))],
            min_order_size=Decimal("1"),
            tick_size=Decimal("0.01"),
            neg_risk=False,
            hash="x",
        )

    monkeypatch.setattr(exits_mod, "get_order_book", fake_get_order_book)
    return calls


def stub_failing_market_order(monkeypatch):
    async def fake(client, token_id, side, amount):
        raise RuntimeError("FAK couldn't find any matching orders")

    monkeypatch.setattr(exits_mod, "place_market_order", fake)


def make_mined_trade() -> UserTrade:
    return UserTrade(
        event_type="trade",
        id="trade-1",
        asset_id="tok-yes",
        market="market-A",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("100"),
        outcome="YES",
        status="MINED",
        timestamp="2026-05-19T12:00:00Z",
        maker_orders=[],
        taker_order_id="entry-oid",
    )


def stage_pending_fok(state: FarmState) -> None:
    state.pending_fok_exits["trade-1"] = FokExitInfo(
        token_id="tok-yes", size=Decimal("100"), outcome="YES", slug="m1"
    )


async def test_fok_failure_falls_back_to_gtc_limit_at_best_bid(
    farm_state: FarmState, stub_network, monkeypatch
):
    stage_pending_fok(farm_state)
    stub_failing_market_order(monkeypatch)
    stub_order_book(monkeypatch, best_bid=Decimal("0.42"))

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    assert stub_network["limit"], "expected a GTC fallback placement"
    token_id, side, size, price, post_only = stub_network["limit"][0]
    assert side == "SELL"
    assert price == 0.42
    assert post_only is False
    assert "trade-1" not in farm_state.pending_fok_exits


async def test_fok_fallback_sells_at_very_low_price(
    farm_state: FarmState, stub_network, monkeypatch
):
    # No floor — even at 3¢ we sell rather than orphan.
    stage_pending_fok(farm_state)
    stub_failing_market_order(monkeypatch)
    stub_order_book(monkeypatch, best_bid=Decimal("0.03"))

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    assert stub_network["limit"], "expected GTC fallback at any positive price"
    _, side, _, price, _ = stub_network["limit"][0]
    assert side == "SELL"
    assert price == 0.03


async def test_fok_fallback_abandons_at_zero_bid(farm_state: FarmState, stub_network, monkeypatch):
    stage_pending_fok(farm_state)
    stub_failing_market_order(monkeypatch)
    book_fetches = stub_order_book(monkeypatch, best_bid=Decimal("0"))

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    # The fallback path MUST have been entered (book fetched) — that's how we
    # know we got past the FAK failure into the GTC code at all.
    assert book_fetches, "FAK fallback path did not run (get_order_book never called)"
    # ... but at best_bid=0 there's no price to sell at, so no placement.
    assert stub_network["limit"] == []
    assert "trade-1" not in farm_state.pending_fok_exits


async def test_fok_sells_instantly_even_when_market_paused(
    farm_state: FarmState, stub_network, monkeypatch
):
    # Filled inventory must ALWAYS sell instantly. A pause stops us OPENING/re-quoting a
    # market; it must never delay EXITING shares we already hold (that delay is the loss we're
    # avoiding). Regression for the tuyo incident: the exit-loss guard paused the market right
    # at the fill, and the old skip-on-pause then deferred the sell ~80s to the reconcile sweep.
    stage_pending_fok(farm_state)
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=1)
    )
    stub_order_book(monkeypatch, best_bid=Decimal("0.40"))

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    assert len(stub_network["market"]) == 1  # FAK SELL fired despite the pause
    assert "trade-1" not in farm_state.pending_fok_exits


async def test_fok_succeeds_first_try_does_not_invoke_fallback(
    farm_state: FarmState, stub_network, monkeypatch
):
    stage_pending_fok(farm_state)
    stub_order_book(monkeypatch, best_bid=Decimal("0.40"))

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    assert len(stub_network["market"]) == 1
    assert stub_network["limit"] == []
    assert "trade-1" not in farm_state.pending_fok_exits


async def test_orphan_dust_below_book_min_written_off_not_sold(
    farm_state: FarmState, stub_network, monkeypatch
):
    # Orphan exit path: the position was already dropped, so exit_position_leg runs with
    # pos=None and the proactive (pos-based) dust guard can't fire. The FAK is rejected; the
    # GTC fallback must then recognise the sub-min residual via book.min_order_size and write
    # it off — NOT place a doomed GTC and pause the (phantom) market. (Copilot PR #31 edge.)
    stub_failing_market_order(monkeypatch)
    stub_order_book(monkeypatch, best_bid=Decimal("0.40"))  # book min_order_size = 1
    await exits_mod.exit_position_leg(
        MagicMock(), farm_state, "tok-orphan", Decimal("0.5"), "orphan-cid", "m1", "YES"
    )
    assert stub_network["limit"] == [], "sub-min orphan dust must NOT place a GTC SELL"
    assert is_paused(farm_state, "orphan-cid") is False, "dust write-off must not pause"


async def test_orphan_above_book_min_still_sells_via_gtc(
    farm_state: FarmState, stub_network, monkeypatch
):
    # Counterpart guard: a non-dust orphan residual (>= book min) must STILL sell via the GTC
    # fallback — the new dust check must not swallow legitimately sellable shares.
    stub_failing_market_order(monkeypatch)
    stub_order_book(monkeypatch, best_bid=Decimal("0.40"))
    await exits_mod.exit_position_leg(
        MagicMock(), farm_state, "tok-orphan", Decimal("50"), "orphan-cid", "m1", "YES"
    )
    assert stub_network["limit"], "non-dust orphan must sell via the GTC fallback"
