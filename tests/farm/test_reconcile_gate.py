"""reconcile_tick opens every filtered candidate the bankroll can back, gated
per-market by two_leg_cost <= effective_bankroll. There is no ranking and
no global budget: limit orders don't escrow against the bankroll across markets,
so two markets that each fit are BOTH opened even when their costs sum past it.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market
from app.farm.worker import reconcile_tick


def wire(monkeypatch, markets: list[Market], balance: Decimal, place_calls: list):
    async def fake_fetch_eligible_markets(http):
        return markets

    async def fake_fetch_midpoints(http, token_ids):
        return {tid: Decimal("0.5") for tid in token_ids}

    async def fake_get_balance(addr):
        return balance

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append(order.token_id)
        return f"oid-{len(place_calls)}"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)


def _market(base: Market, cid: str, min_size: Decimal) -> Market:
    return base.model_copy(
        update={
            "condition_id": cid,
            "yes_token_id": f"{cid}-yes",
            "no_token_id": f"{cid}-no",
            "rewards_min_size": min_size,  # size = max(min_order_size, this); cost = size×(bids)
        }
    )


async def test_opens_only_markets_within_bankroll(
    farm_state: FarmState, market: Market, monkeypatch
):
    # bankroll 100. cheap costs $100 (fits, == bankroll); pricey costs $200 (skipped).
    farm_state.positions.clear()
    cheap = _market(market, "cheap", Decimal("100"))
    pricey = _market(market, "pricey", Decimal("200"))
    place_calls: list = []
    wire(monkeypatch, [cheap, pricey], balance=Decimal("1000"), place_calls=place_calls)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert set(place_calls) == {"cheap-yes", "cheap-no"}
    assert "pricey-yes" not in place_calls and "pricey-no" not in place_calls


async def test_opens_all_affordable_no_global_cap(
    farm_state: FarmState, market: Market, monkeypatch
):
    # Both markets cost $100 each and the bankroll is $100. Combined cost ($200)
    # exceeds the bankroll, yet BOTH open: the gate is per-market, not global.
    farm_state.positions.clear()
    a = _market(market, "a", Decimal("100"))
    b = _market(market, "b", Decimal("100"))
    place_calls: list = []
    wire(monkeypatch, [a, b], balance=Decimal("1000"), place_calls=place_calls)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert set(place_calls) == {"a-yes", "a-no", "b-yes", "b-no"}
