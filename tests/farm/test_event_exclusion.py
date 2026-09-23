"""Event-family exclusion (ported from #29): an adverse fill quarantines the whole event
(excluded_events via event_slug), and reconcile then skips every sibling market under it."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.bot.schemas import UserTrade, UserTradeMakerOrder
from app.farm import exits as exits_mod
from app.farm import fills as fills_mod
from app.farm import worker as worker_mod
from app.farm.discovery import parent_event
from app.farm.fills import handle_trade
from app.farm.health import is_event_excluded
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market, OrderInfo
from app.farm.worker import reconcile_tick

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def maker_fill(trade_id: str) -> UserTrade:
    return UserTrade(
        event_type="trade",
        id=trade_id,
        asset_id="tok-yes",
        market="market-A",
        side="BUY",
        price=Decimal("0.5"),
        size=Decimal("100"),
        outcome="YES",
        status="MATCHED",
        timestamp="2026-01-01T00:00:00Z",
        taker_order_id="taker-x",
        maker_orders=[
            UserTradeMakerOrder(
                asset_id="tok-yes",
                order_id="yes-oid",
                matched_amount=Decimal("100"),
                outcome="YES",
                owner="0xowner",
                price=Decimal("0.5"),
            )
        ],
    )


def stub_fill_network(monkeypatch):
    async def fake_market_order(client, token_id, side, size):
        return "x"

    async def fake_cancel(client, oid):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_market_order)
    monkeypatch.setattr(fills_mod, "cancel_order", fake_cancel)


async def test_fill_excludes_event_family(farm_state, monkeypatch):
    stub_fill_network(monkeypatch)
    farm_state.positions["market-A"].market.event_slug = "world-cup-group-d"

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    assert "world-cup-group-d" in farm_state.excluded_events


async def test_fill_with_no_event_slug_adds_nothing(farm_state, monkeypatch):
    # event_slug defaults to "" → guard against quarantining the empty-string "event".
    stub_fill_network(monkeypatch)

    await handle_trade(MagicMock(), maker_fill("t2"), farm_state, AsyncMock())

    assert farm_state.excluded_events == {}


async def test_fill_logs_event_family_exclusion(farm_state, monkeypatch, caplog):
    # The exclusion must be observable in the logs (it was previously a silent set.add).
    stub_fill_network(monkeypatch)
    farm_state.positions["market-A"].market.event_slug = "world-cup-group-d"

    with caplog.at_level(logging.INFO, logger="app.farm.fills"):
        await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "excluded event family world-cup-group-d" in text


async def test_event_family_exclusion_logged_once(farm_state, monkeypatch, caplog):
    # Logged on the state transition only — a second fill on the already-excluded family
    # must not re-log (keeps the signal clean, no per-fill spam).
    stub_fill_network(monkeypatch)
    farm_state.positions["market-A"].market.event_slug = "world-cup-group-d"

    with caplog.at_level(logging.INFO, logger="app.farm.fills"):
        await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())
        await handle_trade(MagicMock(), maker_fill("t2"), farm_state, AsyncMock())

    hits = [r for r in caplog.records if "excluded event family" in r.getMessage()]
    assert len(hits) == 1, "exclusion should log once per family, not on every subsequent fill"


def add_resting_sibling(farm_state, *, cid: str, shares: Decimal) -> None:
    """Clone market-A into a same-family sibling with two resting BUY orders and `shares` held."""
    sib = farm_state.positions["market-A"].model_copy(deep=True)
    sib.market.condition_id = cid
    sib.market.slug = f"sibling-{cid}"
    sib.market.event_slug = "world-cup-group-d"
    sib.yes_shares = shares
    sib.no_shares = Decimal("0")
    sib.yes_order_id = f"{cid}-yes-oid"
    sib.no_order_id = f"{cid}-no-oid"
    farm_state.positions[cid] = sib
    # Mirror open_position: the resting orders are tracked in the registry.
    farm_state.order_registry[f"{cid}-yes-oid"] = OrderInfo(
        condition_id=cid, outcome="YES", token_id=sib.market.yes_token_id
    )
    farm_state.order_registry[f"{cid}-no-oid"] = OrderInfo(
        condition_id=cid, outcome="NO", token_id=sib.market.no_token_id
    )


async def test_fill_cancels_and_drops_resting_siblings(farm_state, monkeypatch):
    # The cascade fix: when one market in a family fills, pull resting orders on every 0-share
    # sibling in the same family AND drop the position (so handle_bba can't re-quote it).
    stub_fill_network(monkeypatch)
    pulled: list = []

    async def fake_cancel_orders(client, *oids):
        pulled.extend(oids)

    monkeypatch.setattr(fills_mod, "cancel_orders", fake_cancel_orders)
    farm_state.positions["market-A"].market.event_slug = "world-cup-group-d"
    add_resting_sibling(farm_state, cid="market-B", shares=Decimal("0"))

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    assert "world-cup-group-d" in farm_state.excluded_events
    assert set(pulled) == {"market-B-yes-oid", "market-B-no-oid"}, "sibling orders must be pulled"
    assert "market-B" not in farm_state.positions, "0-share sibling must be dropped"
    assert "market-A" in farm_state.positions, "the filled market is handled by its own exit"
    # Registry oids dropped too (like close_position) so a late fill can't mis-resolve if the
    # market is re-opened after the exclusion expires.
    assert "market-B-yes-oid" not in farm_state.order_registry
    assert "market-B-no-oid" not in farm_state.order_registry


async def test_fill_survives_family_cancel_failure(farm_state, monkeypatch):
    # A sibling-cancel raising must NOT crash handle_trade (that would kill the user-WS loop
    # and stop all fill/exit processing). The exclusion is still recorded.
    stub_fill_network(monkeypatch)

    async def boom(*a, **k):
        raise RuntimeError("cancel blew up")

    monkeypatch.setattr(fills_mod, "cancel_family_orders", boom)
    farm_state.positions["market-A"].market.event_slug = "world-cup-group-d"

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())  # must not raise

    assert "world-cup-group-d" in farm_state.excluded_events


async def test_fill_leaves_sibling_holding_shares_alone(farm_state, monkeypatch):
    # A sibling that already holds inventory is owned by its own exit flow — never pull/drop it.
    stub_fill_network(monkeypatch)
    pulled: list = []

    async def fake_cancel_orders(client, *oids):
        pulled.extend(oids)

    monkeypatch.setattr(fills_mod, "cancel_orders", fake_cancel_orders)
    farm_state.positions["market-A"].market.event_slug = "world-cup-group-d"
    add_resting_sibling(farm_state, cid="market-B", shares=Decimal("50"))

    await handle_trade(MagicMock(), maker_fill("t1"), farm_state, AsyncMock())

    assert "market-B" in farm_state.positions, "sibling holding shares must be left in place"
    assert pulled == [], "must not pull orders on a sibling that holds inventory"


def test_is_event_excluded_is_timed():
    config = FarmConfig(
        filters=_filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50")
    )
    state = FarmState(config=config)
    state.excluded_events["fam-live"] = datetime(2099, 1, 1, tzinfo=timezone.utc)
    state.excluded_events["fam-expired"] = datetime(2020, 1, 1, tzinfo=timezone.utc)
    assert is_event_excluded(state, "fam-live") is True
    assert is_event_excluded(state, "fam-expired") is False  # timed out → eligible again
    assert is_event_excluded(state, "fam-unknown") is False


async def test_reconcile_reenters_expired_event(monkeypatch, caplog):
    # After the quarantine expires the family is a candidate again (auto re-entry).
    m = _market(
        condition_id="x",
        slug="x",
        yes_token_id="xy",
        no_token_id="xn",
        event_slug="world-cup-group-d",
    )
    config = FarmConfig(
        filters=_filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50")
    )
    state = FarmState(config=config, wallet_address="0xabc")
    state.excluded_events["world-cup-group-d"] = datetime(2020, 1, 1, tzinfo=timezone.utc)

    async def fake_markets(http):
        return [m]

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidates=1" in text, "expired exclusion must let the family back in"
    assert "excluded_event" not in text


def _market(**ov) -> Market:
    base = dict(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="y",
        no_token_id="n",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("5"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
    )
    base.update(ov)
    return Market(**base)


def _filters() -> FarmFilters:
    return FarmFilters(
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


def patch_common(monkeypatch):
    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_place_limit_order(client, order, post_only=False):
        return "oid"

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    async def fake_fetch_midpoints(http, token_ids):
        return {}

    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place_limit_order)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)


async def test_reconcile_skips_market_in_excluded_event(monkeypatch, caplog):
    m = _market(
        condition_id="x",
        slug="x",
        yes_token_id="xy",
        no_token_id="xn",
        event_slug="world-cup-group-d",
    )
    config = FarmConfig(
        filters=_filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50")
    )
    state = FarmState(config=config, wallet_address="0xabc")
    # Far-future expiry => currently quarantined.
    state.excluded_events["world-cup-group-d"] = datetime(2099, 1, 1, tzinfo=timezone.utc)

    async def fake_markets(http):
        return [m]

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel excluded_event=1" in text
    assert "candidates=0" in text


# ── parent_event extraction (feeds event_slug, which the exclusion keys on) ───


def test_parent_event_returns_first_dict():
    assert parent_event({"events": [{"id": "9", "slug": "e9"}, {}]}) == {"id": "9", "slug": "e9"}


@pytest.mark.parametrize("events", [None, [], ["junk"], "x", [123]])
def test_parent_event_malformed_returns_empty(events):
    assert parent_event({"events": events}) == {}
