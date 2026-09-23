"""Wiring tests: verify that the helpers from bug 3 are actually invoked at
their intended call sites. Without these, a bug-fix could pass unit tests while
the production path silently bypasses the helper entirely."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import exits as exits_mod
from app.farm import requote as requote_mod
from app.farm import worker as worker_mod
from app.farm.filters import passes_all
from app.farm.requote import requote_leg
from app.farm.schemas import (
    FarmState,
    Market,
    MarketHealth,
)
from app.farm.worker import reconcile_tick
from tests.fills.test_fok_fallback import (
    make_mined_trade,
    stage_pending_fok,
    stub_failing_market_order,
    stub_order_book,
)

# (1) passes_all → passes_live_event_filter wiring


def test_passes_all_excludes_market_within_6h_of_game_start(market: Market):
    live_market = market.model_copy(
        update={
            "game_start_time": datetime.now(timezone.utc) + timedelta(hours=2),
        }
    )
    # All other filter fields default-pass on the fixture's permissive config.
    from app.farm.schemas import FarmFilters

    filters = FarmFilters(
        vol_min=Decimal(0),
        vol_max=Decimal(100000),
        liq_min=Decimal(0),
        liq_max=Decimal(100000),
        spread_min=Decimal(0),
        spread_max=Decimal(100),
        reward_min=Decimal(0),
        time_remaining="all",
        created_date="all",
        change_24h="all",
    )
    assert passes_all(live_market, filters) is False


# (2) reconcile_tick → is_paused wiring


async def test_reconcile_tick_excludes_paused_markets(
    farm_state: FarmState, market: Market, monkeypatch
):
    farm_state.positions.clear()
    farm_state.config = farm_state.config.model_copy(update={"bankroll": Decimal("10000")})
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=5)
    )
    place_calls: list = []

    async def fake_fetch_eligible_markets(http):
        return [market]

    async def fake_fetch_midpoints(http, token_ids):
        return {tid: Decimal("0.5") for tid in token_ids}

    async def fake_get_balance(addr):
        return Decimal("10000")

    async def fake_place_limit_order(client, order, post_only=False):
        place_calls.append(order.token_id)
        return f"oid-{len(place_calls)}"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls == [], f"Paused market should not have been opened, but got {place_calls}"


# (3) requote_leg → record_market_failure wiring


async def test_requote_failure_records_failure_in_health(farm_state: FarmState, monkeypatch):
    pos = farm_state.positions["market-A"]

    async def failing_place(client, order, post_only=False):
        raise RuntimeError("post-only crosses book")

    async def fake_cancel(client, oid):
        return None

    monkeypatch.setattr(requote_mod, "place_limit_order", failing_place)
    monkeypatch.setattr(requote_mod, "cancel_order", fake_cancel)

    await requote_leg(MagicMock(), farm_state, AsyncMock(), pos, "YES", Decimal("0.40"))

    health = farm_state.health.get("market-A")
    assert health is not None and len(health.recent_failures) == 1, (
        "requote placement failure must be recorded in MarketHealth"
    )


# (4) fills.py FAK + GTC both fail → record_market_failure wiring


async def test_fok_and_gtc_both_failing_record_failure_in_health(
    farm_state: FarmState, monkeypatch
):
    stage_pending_fok(farm_state)
    stub_failing_market_order(monkeypatch)
    stub_order_book(monkeypatch, best_bid=Decimal("0.40"))

    async def failing_limit_place(client, order, post_only=False):
        raise RuntimeError("GTC also rejected")

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_limit_order", failing_limit_place)
    # M2: cancel_orders is a thin shim that awaits client.cancel_orders; patch at exits level.
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)

    from app.farm.fills import handle_trade

    await handle_trade(MagicMock(), make_mined_trade(), farm_state, AsyncMock())

    health = farm_state.health.get("market-A")
    assert health is not None and len(health.recent_failures) == 1, (
        "FAK+GTC double failure must be recorded in MarketHealth"
    )
