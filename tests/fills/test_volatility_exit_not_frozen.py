"""POL-54 decision: a volatility blacklist is an entry/quoting filter, NOT an exit
freeze. The MINED exit no longer gates on pause/blacklist at all (the is_paused skip was
removed), so a blacklisted or paused market still instantly sells out shares it already
holds ("always exit, never hold")."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import UserTrade
from app.farm import fills as fills_mod
from app.farm.fills import handle_trade
from app.farm.schemas import FarmState, FokExitInfo, MarketHealth


async def test_exit_fires_even_when_market_blacklisted(farm_state: FarmState, monkeypatch):
    exits: list = []

    async def fake_exit(client, state, token_id, size, cid, _slug, outcome, **kwargs):
        exits.append((token_id, float(size), outcome))

    async def fake_cancel(client, oid):
        pass

    monkeypatch.setattr(fills_mod, "exit_position_leg", fake_exit)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)

    farm_state.health["market-A"] = MarketHealth(blacklist_permanent=True)
    farm_state.pending_fok_exits["trade-x"] = FokExitInfo(
        token_id="tok-no", size=Decimal("20"), outcome="NO", slug="m1", entry_cost=Decimal("18")
    )
    mined = UserTrade(
        event_type="trade",
        id="trade-x",
        asset_id="tok-no",
        market="market-A",
        side="BUY",
        price=Decimal("0.9"),
        size=Decimal("20"),
        outcome="NO",
        status="MINED",
        timestamp="2026-06-01T12:00:00Z",
        maker_orders=[],
        taker_order_id="taker",
    )

    await handle_trade(MagicMock(), mined, farm_state, AsyncMock())

    assert exits == [("tok-no", 20.0, "NO")], "volatility blacklist must NOT freeze exits"
