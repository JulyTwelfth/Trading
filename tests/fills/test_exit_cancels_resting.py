"""POL-50: exit_position_leg must cancel resting orders on the same market
before placing the SELL — otherwise the CLOB allowance check rejects the
SELL with "sum of matched orders > balance".
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import UserTrade
from app.farm import exits as exits_mod
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState, FokExitInfo


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


async def test_exit_cancels_resting_orders_before_sell(farm_state: FarmState, monkeypatch):
    """Recorded call order must be: cancel(yes-oid, no-oid) then place SELL.
    If the SELL fires first, the CLOB would reject it for over-committed allowance."""
    stage_pending_fok(farm_state)
    events: list[tuple[str, tuple]] = []

    async def fake_cancel(client, *order_ids):
        events.append(("cancel", order_ids))

    async def fake_place_market(client, token_id, side, amount):
        events.append(("sell", (token_id, side, amount)))
        return "exit-oid"

    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel)
    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market)

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    assert events == [
        ("cancel", ("yes-oid", "no-oid")),
        ("sell", ("tok-yes", "SELL", 100.0)),
    ], f"expected cancel-then-sell ordering, got {events}"
    assert "exit-oid" in farm_state.pending_exit_order_ids
