"""POL-54: reconcile_tick must exclude volatility-blacklisted markets from new opens."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.schemas import FarmState, Market, MarketHealth
from app.farm.worker import reconcile_tick


def wire(monkeypatch, market: Market, place_calls: list):
    async def fake_fetch_eligible_markets(http):
        return [market]

    async def fake_get_balance(addr):
        return Decimal("150")

    async def fake_fetch_midpoints(http, token_ids):
        return {t: Decimal("0.5") for t in token_ids}

    async def fake_place(client, order, post_only=False):
        place_calls.append(order.token_id)
        return f"oid-{len(place_calls)}"

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_fetch_eligible_markets)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)
    monkeypatch.setattr(worker_mod, "passes_all", lambda m, f: True)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)


async def test_non_blacklisted_market_is_opened(farm_state: FarmState, market: Market, monkeypatch):
    farm_state.positions.clear()
    place_calls: list = []
    wire(monkeypatch, market, place_calls)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls, "control: a clean candidate should be opened"


async def test_blacklisted_market_is_not_opened(farm_state: FarmState, market: Market, monkeypatch):
    farm_state.positions.clear()
    farm_state.health["market-A"] = MarketHealth(
        blacklist_until=datetime.now(timezone.utc) + timedelta(minutes=15)
    )
    place_calls: list = []
    wire(monkeypatch, market, place_calls)

    await reconcile_tick(MagicMock(), MagicMock(), farm_state, AsyncMock())

    assert place_calls == [], "blacklisted candidate must not be opened"
