"""POL-36 Bug 1: a MATCHED frame on an order_id that requote_leg just rotated
away from the live pointer must still be recognized. Today, handle_trade keys
recognition on pos.yes_order_id/no_order_id (the live pointer); after the fix,
it keys on state.order_registry — so a fill on old_oid still registers a FAK
exit even after pos.yes_order_id == new_oid."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm import requote as requote_mod
from app.farm.fills import handle_trade
from app.farm.requote import requote_leg
from app.farm.schemas import FarmState


@pytest.fixture
def stub_network(monkeypatch):
    async def fake_cancel(client, oid):
        return None

    async def fake_place_limit(client, order, post_only=False):
        return "yes-oid-v2"

    async def fake_place_market(client, token_id, side, size):
        return "exit-oid-1"

    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(requote_mod, "place_limit_order", fake_place_limit)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)
    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market)


async def test_match_on_rotated_old_oid_is_recognized(farm_state: FarmState, stub_network):
    pos = farm_state.positions["market-A"]
    old_oid = pos.yes_order_id  # "yes-oid", seeded into registry by conftest
    yes_shares_before = pos.yes_shares

    # 1. Drive requote_leg for real — rotates pos.yes_order_id from old_oid to "yes-oid-v2".
    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.48"))
    assert pos.yes_order_id == "yes-oid-v2"
    assert old_oid in farm_state.order_registry, (
        "old_oid must NOT be evicted on requote — that's the bug we're fixing"
    )
    assert "yes-oid-v2" in farm_state.order_registry

    # 2. MATCHED frame arrives quoting the OLD oid as maker (the race).
    matched = UserTrade(
        event_type="trade",
        id="trade-race-1",
        asset_id=pos.market.yes_token_id,
        market="market-A",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("10"),
        outcome="YES",
        status="MATCHED",
        timestamp="2026-05-24T00:00:00Z",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id=pos.market.yes_token_id,
                order_id=old_oid,
                matched_amount=Decimal("10"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.5"),
            )
        ],
        taker_order_id="some-taker-id",
    )

    await handle_trade(MagicMock(), matched, farm_state, AsyncMock())

    # 3. Recognition succeeded → all the downstream accounting fires.
    assert "trade-race-1" in farm_state.pending_fok_exits, (
        "fill on rotated-away oid must register an FAK exit"
    )
    fok = farm_state.pending_fok_exits["trade-race-1"]
    assert fok.token_id == pos.market.yes_token_id
    assert fok.outcome == "YES"
    assert fok.size == Decimal("10")
    assert pos.yes_shares == yes_shares_before + Decimal("10")
