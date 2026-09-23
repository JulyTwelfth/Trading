"""Exits book on SETTLEMENT (MINED/CONFIRMED), not the provisional MATCHED — and a terminal
FAILED re-fires the sell instantly off the real-time frame.

The 6-stuck-positions incident: the bot booked the sale on MATCHED ("sent to the executor", NOT
settled), dropped the order, and so ignored the trade never settling — leaving shares on-chain
while it reported flat. Now MATCHED is provisional; we only decrement/record on MINED/CONFIRMED,
and react to FAILED immediately.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState


@pytest.fixture
def stub_net(monkeypatch):
    async def fake_cancel(client, oid):
        return None

    async def fake_market_order(client, token_id, side, size):
        return "resell-oid"

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(exits_mod, "place_market_order", fake_market_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", AsyncMock())


def sell(status: str, *, size="100", price="0.45", oid="exit-oid-1") -> UserTrade:
    return UserTrade(
        event_type="trade",
        id=f"exit-{status.lower()}",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=Decimal(price),
        size=Decimal(size),
        outcome="YES",
        status=status,
        timestamp="2026-06-17T12:00:00Z",
        maker_orders=[],
        taker_order_id=oid,
    )


async def test_matched_exit_does_not_book(farm_state: FarmState, stub_net):
    # The core fix: a SELL's MATCHED is provisional — must NOT decrement shares or book PnL,
    # and must keep the order tracked so a later FAILED is still recognized.
    farm_state.config.max_session_loss = Decimal("100")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(MagicMock(), sell("MATCHED"), farm_state, AsyncMock())

    assert farm_state.positions["market-A"].yes_shares == Decimal("100"), "MATCHED must not book"
    assert "exit-oid-1" in farm_state.pending_exit_order_ids, "must stay tracked until settled"
    assert farm_state.session_loss == Decimal("0")


async def test_mined_exit_books(farm_state: FarmState, stub_net):
    farm_state.config.max_session_loss = Decimal("100")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(MagicMock(), sell("MATCHED"), farm_state, AsyncMock())  # provisional
    await handle_trade(MagicMock(), sell("MINED"), farm_state, AsyncMock())  # settled → book

    assert farm_state.positions["market-A"].yes_shares == Decimal("0"), "booked on MINED"
    assert "exit-oid-1" not in farm_state.pending_exit_order_ids
    # entry cost 50 − proceeds (100 * 0.45 = 45) = $5 realized loss.
    assert farm_state.session_loss == Decimal("5")


async def test_confirmed_books_when_mined_frame_lost(farm_state: FarmState, stub_net):
    # If the MINED frame is missed, CONFIRMED (terminal success) still books the sale.
    farm_state.config.max_session_loss = Decimal("100")
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(MagicMock(), sell("CONFIRMED"), farm_state, AsyncMock())

    assert farm_state.positions["market-A"].yes_shares == Decimal("0")
    assert "exit-oid-1" not in farm_state.pending_exit_order_ids


async def test_failed_exit_refires_sell_immediately(farm_state: FarmState, monkeypatch):
    # Terminal FAILED → the shares are still ours; re-fire the sell NOW (no 60s wait).
    redrives: list = []

    async def fake_exit(client, state, token_id, size, cid, slug, outcome, **kw):
        redrives.append((token_id, size, outcome))

    async def fake_cancel(client, oid):
        return None

    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(MagicMock(), sell("FAILED"), farm_state, AsyncMock())

    assert redrives == [("tok-yes", Decimal("100"), "YES")], (
        "FAILED must instantly re-fire the sell"
    )
    assert farm_state.positions["market-A"].yes_shares == Decimal("100"), "not booked on FAILED"
    assert "exit-oid-1" not in farm_state.pending_exit_order_ids, "old failed oid cleared"


async def test_retrying_exit_waits(farm_state: FarmState, monkeypatch):
    # RETRYING = operator auto-retrying the SAME trade → wait; don't book and don't fire a
    # competing sell (that would double-sell).
    redrives: list = []

    async def fake_exit(*a, **k):
        redrives.append(a)

    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)
    farm_state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(MagicMock(), sell("RETRYING"), farm_state, AsyncMock())

    assert redrives == [], "RETRYING must not fire a competing sell"
    assert "exit-oid-1" in farm_state.pending_exit_order_ids, "still tracked (awaiting retry)"
    assert farm_state.positions["market-A"].yes_shares == Decimal("100"), "not booked"
