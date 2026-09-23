"""On a maker ENTRY fill, handle_trade must pull EVERY resting order on that market —
both legs, plus any rotated-away filled oid — not just the filled order.

The 2026-06-15 incident: a YES leg filled, the market was fill-blacklisted, but the
sibling NO order (and a leg the market-WS loop re-quoted in the race) stayed resting
for ~2s until the MINED exit cancelled them. A crashing book hit a re-armed leg in that
gap and double-filled the quarantined market (~$20 vs the ~$10 a single fill cost). The
fix mirrors the volatility-blacklist pull: cancel all of the market's orders the instant
a fill lands, so a blacklisted market is left with nothing that can fill.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.farm.messages import OrderCancelledEvent
from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import fills as fills_mod
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState, OrderInfo
from app.farm.volatility import is_blacklisted


@pytest.fixture
def record_cancels(monkeypatch):
    """Record every oid handle_trade cancels; stub the staged-exit market order too."""
    cancelled: list[str] = []

    async def fake_cancel(client, oid):
        cancelled.append(oid)

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    return cancelled


def maker_fill(
    *,
    order_id: str = "yes-oid",
    asset_id: str = "tok-yes",
    outcome: str = "YES",
    size: Decimal = Decimal("50"),
    price: Decimal = Decimal("0.38"),
    trade_id: str = "trade-fill-1",
) -> UserTrade:
    """A maker BUY that just filled (the adverse-selection entry we want to quarantine)."""
    return UserTrade(
        event_type="trade",
        id=trade_id,
        asset_id=asset_id,
        market="market-A",
        side="BUY",
        price=price,
        size=size,
        outcome=outcome,
        status="MATCHED",
        timestamp="2026-06-15T00:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id=asset_id,
                order_id=order_id,
                matched_amount=size,
                outcome=outcome,
                owner="0xowner",
                price=price,
            )
        ],
        taker_order_id="some-other-taker",
    )


async def test_yes_fill_cancels_both_legs(farm_state: FarmState, record_cancels):
    # YES leg fills → the resting NO sibling must be pulled too, not left exposed.
    await handle_trade(MagicMock(), maker_fill(), farm_state, AsyncMock())

    assert set(record_cancels) == {"yes-oid", "no-oid"}, (
        "a fill must cancel the sibling leg, not only the filled order"
    )


async def test_no_fill_cancels_both_legs(farm_state: FarmState, record_cancels):
    # Symmetric: a NO fill pulls the resting YES sibling.
    await handle_trade(
        MagicMock(),
        maker_fill(order_id="no-oid", asset_id="tok-no", outcome="NO"),
        farm_state,
        AsyncMock(),
    )

    assert set(record_cancels) == {"yes-oid", "no-oid"}


async def test_fill_cancels_rotated_live_leg_too(farm_state: FarmState, record_cancels):
    """POL-36 race: the fill lands on a rotated-away oid while the live YES pointer is a
    newer order. We must cancel the FILLED oid AND the live leg AND the sibling — leaving
    the newer live order resting would be exactly the double-fill exposure we're closing."""
    pos = farm_state.positions["market-A"]
    pos.yes_order_id = "yes-oid-v2"  # a requote rotated the live pointer forward
    farm_state.order_registry["yes-oid-v2"] = OrderInfo(
        condition_id="market-A", outcome="YES", token_id=pos.market.yes_token_id
    )

    # The fill quotes the OLD (rotated-away) oid as maker.
    await handle_trade(MagicMock(), maker_fill(order_id="yes-oid"), farm_state, AsyncMock())

    assert set(record_cancels) == {"yes-oid", "yes-oid-v2", "no-oid"}, (
        "the live rotated leg must also be cancelled, not just the filled oid"
    )


async def test_fill_with_missing_sibling_oid_is_safe(farm_state: FarmState, record_cancels):
    # If a leg has no resting order (oid == ""), it's skipped — no empty-oid cancel, no crash.
    pos = farm_state.positions["market-A"]
    pos.no_order_id = ""

    await handle_trade(MagicMock(), maker_fill(), farm_state, AsyncMock())

    assert record_cancels == ["yes-oid"], "empty sibling oid must be skipped, not cancelled"
    assert is_blacklisted(farm_state, "market-A") is True


async def test_fill_emits_filled_exit_cancel_event_for_both_legs(
    farm_state: FarmState, monkeypatch
):
    # The UI must learn BOTH legs are gone so it stops showing the resting sibling.
    events: list = []

    async def fake_cancel(client, oid):
        return None

    async def fake_send_event(ws, event):
        events.append(event)

    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(fills_mod, "send_event", fake_send_event)

    await handle_trade(MagicMock(), maker_fill(), farm_state, AsyncMock())

    cancels = [
        e for e in events if isinstance(e, OrderCancelledEvent) and e.reason == "filled_exit"
    ]
    assert {(c.outcome, c.order_id) for c in cancels} == {("YES", "yes-oid"), ("NO", "no-oid")}


async def test_fill_still_blacklists_and_stages_exit(farm_state: FarmState, record_cancels):
    # Regression: pulling both legs must not disturb the rest of the fill flow — the market is
    # still blacklisted and the filled inventory is still staged for the MINED exit.
    await handle_trade(MagicMock(), maker_fill(), farm_state, AsyncMock())

    assert is_blacklisted(farm_state, "market-A") is True
    assert "trade-fill-1" in farm_state.pending_fok_exits
    fok = farm_state.pending_fok_exits["trade-fill-1"]
    assert fok.outcome == "YES" and fok.size == Decimal("50")
    assert farm_state.positions["market-A"].yes_shares == Decimal("150")  # 100 seeded + 50


async def test_sibling_cancel_failure_does_not_abort_fill(farm_state: FarmState, monkeypatch):
    """A cancel raising on one leg must never crash fill handling — that would take down the
    user-WS loop and stop ALL fill/exit processing. The blacklist + staged exit must still land."""
    attempted: list[str] = []

    async def flaky_cancel(client, oid):
        attempted.append(oid)
        if oid == "no-oid":
            raise RuntimeError("cancel rejected")

    monkeypatch.setattr(fills_mod, "cancel_order", flaky_cancel)

    await handle_trade(MagicMock(), maker_fill(), farm_state, AsyncMock())

    assert set(attempted) == {"yes-oid", "no-oid"}, "both legs attempted despite the failure"
    assert is_blacklisted(farm_state, "market-A") is True
    assert "trade-fill-1" in farm_state.pending_fok_exits
