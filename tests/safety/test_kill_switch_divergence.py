"""POL-55: the kill switch must book realized PnL from the ACTUAL exit fill, even
when local position state has diverged (shares dropped to 0, or the position was
removed entirely) between entry and exit. Before the fix, a divergent exit booked
proceeds=0 / loss=0, so session_loss never moved and the stop-loss never tripped.

These drive the real flow through handle_trade only (entry MATCHED -> MINED ->
exit MATCHED), so they fail cleanly when the accounting fix is reverted.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm import kill_switch as ks_mod
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState


@pytest.fixture
def stub_io(monkeypatch):
    """Stub all outbound network calls but let the REAL exit_position_leg run, so
    the entry cost is carried end-to-end the way it is in production."""

    async def fake_place_market_order(_client, token_id, _side, _size):
        return "exit-oid-1"

    async def noop(*args, **kwargs):
        return None

    # exits.py I/O — keep exit_position_leg itself real.
    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", noop)
    # fills.py I/O
    monkeypatch.setattr(fills_mod, "cancel_order", noop)
    monkeypatch.setattr(fills_mod, "send_event", noop)
    # kill_switch.py I/O (trigger_kill)
    monkeypatch.setattr(ks_mod, "cancel_all", noop)
    monkeypatch.setattr(ks_mod, "send_event", noop)
    monkeypatch.setattr(exits_mod, "exit_position_leg", noop)


def entry_match() -> UserTrade:
    # Buy 20 NO @ 0.90 -> entry cost 18.00.
    return UserTrade(
        event_type="trade",
        id="trade-entry",
        asset_id="tok-no",
        market="market-A",
        side="BUY",
        price=Decimal("0.90"),
        size=Decimal("20"),
        outcome="NO",
        status="MATCHED",
        timestamp="2026-05-31T22:13:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-no",
                order_id="no-oid",
                matched_amount=Decimal("20"),
                outcome="NO",
                owner="0xowner",
                price=Decimal("0.90"),
            )
        ],
        taker_order_id="taker-buy-1",
    )


def entry_mined() -> UserTrade:
    return UserTrade(
        event_type="trade",
        id="trade-entry",
        asset_id="tok-no",
        market="market-A",
        side="BUY",
        price=Decimal("0.90"),
        size=Decimal("20"),
        outcome="NO",
        status="MINED",
        timestamp="2026-05-31T22:13:01Z",
        maker_orders=[],
        taker_order_id="taker-buy-1",
    )


def exit_match() -> UserTrade:
    # The deferred FAK SELL (oid "exit-oid-1") fills 20 NO @ 0.30 -> proceeds 6.00.
    # Real loss = 18.00 entry - 6.00 proceeds = 12.00, well over the $5 threshold.
    return UserTrade(
        event_type="trade",
        id="trade-exit",
        asset_id="tok-no",
        market="market-A",
        side="SELL",
        price=Decimal("0.30"),
        size=Decimal("20"),
        outcome="NO",
        status="MINED",
        timestamp="2026-05-31T22:15:48Z",
        maker_orders=[],
        taker_order_id="exit-oid-1",
    )


async def test_reopened_empty_position_books_real_loss_and_kills(farm_state: FarmState, stub_io):
    """Position reopened with no_shares=0 between entry and exit (the tabi incident):
    the exit SELL clamps locally, but the kill switch must still see the real loss."""
    client, ws = MagicMock(), AsyncMock()

    await handle_trade(client, entry_match(), farm_state, ws)
    pos = farm_state.positions["market-A"]
    assert pos.no_shares == Decimal("20")  # entry recorded

    # Simulate the worker close/reopen race: shares & cost wiped to 0.
    pos.no_shares = Decimal("0")
    pos.no_cost_basis = Decimal("0")

    await handle_trade(client, entry_mined(), farm_state, ws)  # fires deferred SELL
    await handle_trade(client, exit_match(), farm_state, ws)  # exit fills

    assert farm_state.session_loss == Decimal("12.00"), (
        f"real loss must be booked despite clamp; got {farm_state.session_loss}"
    )
    assert farm_state.killed is True, "stop-loss must trip on the real loss"


async def test_dropped_position_books_real_loss_and_kills(farm_state: FarmState, stub_io):
    """Position removed entirely before the exit fill arrives (pos is None): the loss
    must still be booked from the carried entry cost."""
    client, ws = MagicMock(), AsyncMock()

    await handle_trade(client, entry_match(), farm_state, ws)
    await handle_trade(client, entry_mined(), farm_state, ws)  # fires deferred SELL

    # Position dropped after the exit was placed but before it settles.
    farm_state.positions.pop("market-A")

    await handle_trade(client, exit_match(), farm_state, ws)

    assert farm_state.session_loss == Decimal("12.00"), (
        f"loss must be booked even when pos is gone; got {farm_state.session_loss}"
    )
    assert farm_state.killed is True, "stop-loss must trip even with no live position"
