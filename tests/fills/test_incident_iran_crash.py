"""Replay of the 2026-06-11 incident (iran-close-airspace-aug-31 YES, −$8.00).

A maker BUY filled at 0.67 seconds before the market crashed 0.71 → 0.255. The MINED
frame was lost in the burst, so the staged exit sat until the sweep's staleness window
elapsed, and the fallback path priced off bids[0] — the WORST bid in the CLOB's
ascending book. These tests drive the real flow (handle_trade MATCHED → staged exit →
lost MINED → held-share sweep → FAK failure → GTC fallback) and pin the guarantees
that bound the damage:

1. the GTC fallback sells at the actual best bid, never the 0.01 penny ladder;
2. a lost MINED frame is re-driven after STALE_STAGED_EXIT_SECONDS, not 120s;
3. a "balance: 0" rejection from a not-yet-mined entry strikes instead of clearing,
   so the shares are sold once settled rather than stranded for manual exit.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from py_clob_client_v2.exceptions import PolyApiException

from app.bot.schemas import BookLevel, OrderBook, UserTrade, UserTradeMakerOrder
from app.constants import STALE_STAGED_EXIT_SECONDS
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm.exits import exit_held_legs
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState

ENTRY_PRICE = Decimal("0.67")
ENTRY_SIZE = Decimal("20")


def crashed_book(token_id: str) -> OrderBook:
    """The Aug-31 YES book seconds after the news hit: real CLOB ordering (bids
    ascending, best LAST), a deep penny ladder, and only thin real bids left."""
    return OrderBook(
        market="market-A",
        asset_id=token_id,
        timestamp="2026-06-11T17:30:00Z",
        bids=[
            BookLevel(price=Decimal("0.01"), size=Decimal("100000")),
            BookLevel(price=Decimal("0.02"), size=Decimal("50000")),
            BookLevel(price=Decimal("0.10"), size=Decimal("40")),
            BookLevel(price=Decimal("0.27"), size=Decimal("25")),
        ],
        asks=[
            BookLevel(price=Decimal("0.99"), size=Decimal("4000")),
            BookLevel(price=Decimal("0.34"), size=Decimal("10")),
        ],
        min_order_size=Decimal("5"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="x",
    )


def zero_balance_exc() -> PolyApiException:
    return PolyApiException(
        error_msg={
            "error": (
                "not enough balance / allowance: the balance is not enough "
                "-> balance: 0, order amount: 20000000"
            )
        }
    )


def adverse_fill() -> UserTrade:
    """The taker SELL that hit our resting YES bid at 0.67 right before the crash."""
    return UserTrade(
        event_type="trade",
        id="trade-crash",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=ENTRY_PRICE,
        size=ENTRY_SIZE,
        outcome="YES",
        status="MATCHED",
        timestamp="2026-06-11T17:29:28Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="yes-oid",
                matched_amount=ENTRY_SIZE,
                outcome="YES",
                owner="us",
                price=ENTRY_PRICE,
            )
        ],
        taker_order_id="taker-oid",
    )


@pytest.fixture
def incident_state(farm_state: FarmState) -> FarmState:
    # The fixture position pre-holds 100 YES shares; the incident starts flat.
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_cost_basis = Decimal("0")
    return farm_state


@pytest.fixture
def quiet_cancels(monkeypatch):
    async def fake_cancel(client, *oids):
        return None

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel)


async def fill_and_lose_mined_frame(state: FarmState) -> None:
    """Run the real MATCHED fill handler, then age the staged exit past the staleness
    window — the MINED frame never arrives (dropped during the burst)."""
    await handle_trade(MagicMock(), adverse_fill(), state, AsyncMock())
    pos = state.positions["market-A"]
    assert pos.yes_shares == ENTRY_SIZE, "fill must book the picked-up inventory"
    assert pos.yes_cost_basis == ENTRY_SIZE * ENTRY_PRICE
    info = state.pending_fok_exits["trade-crash"]
    info.staged_at = datetime.now(timezone.utc) - timedelta(seconds=STALE_STAGED_EXIT_SECONDS + 5)


async def sweep(state: FarmState) -> None:
    await exit_held_legs(MagicMock(), state, cancel_resting=True, skip_in_flight=True)


async def test_crash_exit_sells_at_best_bid_not_penny_ladder(
    incident_state: FarmState, quiet_cancels, monkeypatch
):
    """The −$8 path: FAK finds no takers in the swept book, and the fallback must
    price at the surviving best bid (0.27) — never at bids[0] (0.01), which crossed
    and dumped the position at any price."""
    limit_orders: list = []

    async def fak_no_takers(client, token_id, side, amount):
        raise RuntimeError("no match")

    async def capture_limit(client, order, post_only=False):
        limit_orders.append(order)
        return "gtc-1"

    async def book(token_id):
        return crashed_book(token_id)

    monkeypatch.setattr(exits_mod, "place_market_order", fak_no_takers)
    monkeypatch.setattr(exits_mod, "place_limit_order", capture_limit)
    monkeypatch.setattr(exits_mod, "get_order_book", book)

    await fill_and_lose_mined_frame(incident_state)
    await sweep(incident_state)

    assert len(limit_orders) == 1, "lost-MINED leg must be re-driven by the sweep"
    order = limit_orders[0]
    assert order.side == "SELL"
    assert order.size == float(ENTRY_SIZE)
    assert order.price == 0.27, "exit must rest at the best bid, not cross at 0.01"
    assert "gtc-1" in incident_state.pending_exit_order_ids


async def test_fresh_staged_exit_is_not_double_sold(
    incident_state: FarmState, quiet_cancels, monkeypatch
):
    """Counterpart guard: while the staged exit is still fresh (MINED may be seconds
    away), the sweep must NOT fire a second sell for the same shares."""
    sells: list = []

    async def capture_fak(client, token_id, side, amount):
        sells.append((token_id, side, amount))
        return "fak-1"

    monkeypatch.setattr(exits_mod, "place_market_order", capture_fak)

    await handle_trade(MagicMock(), adverse_fill(), incident_state, AsyncMock())
    # staged_at is now() — well inside the freshness window.
    await sweep(incident_state)

    assert sells == [], "sweep must skip a leg whose staged exit is still fresh"


async def test_unmined_entry_is_retried_until_sold_not_stranded(
    incident_state: FarmState, quiet_cancels, monkeypatch
):
    """The manual-sale path: the entry hadn't mined, so every SELL bounced with
    "balance: 0". The old code cleared the leg on the first bounce — stranding the
    shares the moment they settled. Now each bounce strikes and the sweep keeps
    retrying, so the sell lands as soon as the shares exist on-chain."""
    attempts: list = []

    async def fak_unmined_then_ok(client, token_id, side, amount):
        attempts.append(side)
        if len(attempts) < 3:
            raise zero_balance_exc()  # entry still settling on-chain
        return "fak-ok"  # mined — the sell finally goes through

    monkeypatch.setattr(exits_mod, "place_market_order", fak_unmined_then_ok)

    await fill_and_lose_mined_frame(incident_state)
    pos = incident_state.positions["market-A"]

    await sweep(incident_state)
    assert pos.yes_shares == ENTRY_SIZE, "strike 1 must not clear a possibly-unmined leg"
    await sweep(incident_state)
    assert pos.yes_shares == ENTRY_SIZE, "strike 2 must not clear a possibly-unmined leg"
    await sweep(incident_state)

    assert len(attempts) == 3, "sweep must keep re-driving the leg until the sell lands"
    assert "fak-ok" in incident_state.pending_exit_order_ids
    assert pos.zero_balance_since == {}, "successful sell resets the give-up clock"
    assert pos.yes_shares == ENTRY_SIZE, "shares clear via the fill reconcile, not the strike path"
