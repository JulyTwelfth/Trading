"""Fix D1 — post-guard-pull re-quote cooldown.

After a max-fill-loss guard pull, PLACEMENT into that market (a requote or a fresh open) is blocked
for DEPTH_GUARD_PULL_COOLDOWN_SECONDS so a released market can't immediately reopen into the same
bad book. The cooldown gates placement only — the guard itself, and exits, must stay free.

  * mark_guard_pulled / in_guard_pull_cooldown — the health-state primitive (timed, auto-expires).
  * quote_block_reason returns "guard_cooldown" while active (gates handle_bba's requote).
  * reconcile_tick bins a cooling market as "guard_cooldown" so it never becomes a candidate.
  * exit_position_leg ignores the cooldown entirely.

handle_bba tests zero the held YES inventory first (mirrors tests/requote/test_crash_guards.py) so
the mark-to-market kill switch can't pre-empt the code path under test.
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import BestBidAsk
from app.constants import DEPTH_GUARD_PULL_COOLDOWN_SECONDS, REQUOTE_GIVEUP_COOLDOWN_SECONDS
from app.farm import exits as exits_mod
from app.farm import requote as requote_mod
from app.farm import worker as worker_mod
from app.farm.exits import exit_position_leg
from app.farm.gating import quote_block_reason
from app.farm.health import in_guard_pull_cooldown, mark_guard_pulled
from app.farm.requote import depth_fill_loss_guard, handle_bba
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, LiveBook, Market
from app.farm.worker import reconcile_tick

NOW = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)


def make_bba(asset_id: str, best_bid: Decimal, best_ask: Decimal) -> BestBidAsk:
    return BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=asset_id,
        best_bid=best_bid,
        best_ask=best_ask,
        spread=best_ask - best_bid,
        timestamp="2026-07-04T12:00:00Z",
    )


def cooldown_market() -> Market:
    return Market(
        condition_id="0xcool",
        slug="cooling-market",
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


def cooldown_state() -> FarmState:
    config = FarmConfig(
        filters=FarmFilters(
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
        ),
        bankroll=Decimal("1000"),
        max_session_loss=Decimal("50"),
    )
    return FarmState(config=config, wallet_address="0xabc")


def patch_worker_common(monkeypatch):
    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    async def fake_fetch_midpoints(http, token_ids):
        return {}

    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)


# ── 1. the health-state primitive ─────────────────────────────────────────────


def test_mark_and_query_cooldown(farm_state: FarmState):
    assert in_guard_pull_cooldown(farm_state, "market-A", now=NOW) is False

    mark_guard_pulled(farm_state, "market-A", now=NOW)

    assert in_guard_pull_cooldown(farm_state, "market-A", now=NOW) is True
    almost = NOW + timedelta(seconds=DEPTH_GUARD_PULL_COOLDOWN_SECONDS - 1)
    assert in_guard_pull_cooldown(farm_state, "market-A", now=almost) is True
    past = NOW + timedelta(seconds=DEPTH_GUARD_PULL_COOLDOWN_SECONDS + 1)
    assert in_guard_pull_cooldown(farm_state, "market-A", now=past) is False, "auto-expires"


def test_mark_guard_pulled_custom_duration(farm_state: FarmState):
    # The give-up pull passes a longer cooldown via cooldown_s; a default call still lands at 180s,
    # so existing depth-guard / exit-loss callers are unaffected.
    mark_guard_pulled(farm_state, "market-A", now=NOW, cooldown_s=REQUOTE_GIVEUP_COOLDOWN_SECONDS)
    assert farm_state.health["market-A"].guard_pull_until == NOW + timedelta(
        seconds=REQUOTE_GIVEUP_COOLDOWN_SECONDS
    ), "cooldown_s must set the custom (600s) window"

    farm_state.health.pop("market-A", None)
    mark_guard_pulled(farm_state, "market-A", now=NOW)
    assert farm_state.health["market-A"].guard_pull_until == NOW + timedelta(
        seconds=DEPTH_GUARD_PULL_COOLDOWN_SECONDS
    ), "the default arg still lands at the 180s depth-guard cooldown"


# ── 2. quote_block_reason wiring ───────────────────────────────────────────────


def test_quote_block_reason_reports_guard_cooldown(farm_state: FarmState):
    assert quote_block_reason(farm_state, "market-A", "") is None
    mark_guard_pulled(farm_state, "market-A")
    assert quote_block_reason(farm_state, "market-A", "") == "guard_cooldown"


# ── 3. the depth guard starts the cooldown ─────────────────────────────────────


async def test_depth_guard_pull_starts_cooldown(farm_state: FarmState, monkeypatch):
    async def fake_cancel_orders(client, *ids):
        return None

    async def fake_send_event(websocket, event):
        return None

    monkeypatch.setattr(requote_mod, "cancel_orders", fake_cancel_orders)
    monkeypatch.setattr(requote_mod, "send_event", fake_send_event)

    farm_state.config.filters.max_fill_loss = Decimal("1")
    farm_state.live_books["tok-yes"] = LiveBook(bids={Decimal("0.50"): Decimal("10")})
    pos = farm_state.positions["market-A"]
    assert in_guard_pull_cooldown(farm_state, "market-A") is False

    pulled = await depth_fill_loss_guard(MagicMock(), farm_state, AsyncMock(), pos, "tok-yes")

    assert pulled is True
    assert in_guard_pull_cooldown(farm_state, "market-A") is True, "a depth-guard pull must cool"


# ── 4. handle_bba requote is blocked while cooling ─────────────────────────────


async def test_handle_bba_requote_blocked_during_cooldown(farm_state: FarmState, monkeypatch):
    requotes: list = []

    async def fake_requote_leg(client, state, ws, pos, outcome, price):
        requotes.append((outcome, float(price)))

    async def noop_cancel_orders(client, *ids):
        return None

    async def noop_cancel_order(client, oid):
        return None

    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    monkeypatch.setattr(requote_mod, "cancel_orders", noop_cancel_orders)
    monkeypatch.setattr(requote_mod, "cancel_order", noop_cancel_order)

    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")  # neutralize the mark-to-market kill
    pos.yes_cost_basis = Decimal("0")
    mark_guard_pulled(farm_state, "market-A")

    # A threatened frame that would normally requote the YES leg.
    bba = make_bba(pos.market.yes_token_id, Decimal("0.50"), Decimal("0.52"))
    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, bba)

    assert requotes == [], "a market in guard cooldown must not requote"


# ── 5. the discovery loop bins a cooling market ────────────────────────────────


async def test_candidate_loop_skips_cooldown_market(monkeypatch, caplog):
    m = cooldown_market()
    state = cooldown_state()
    mark_guard_pulled(state, "0xcool")

    async def fake_markets(http):
        return [m]

    open_mock = AsyncMock()
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "open_position", open_mock)
    patch_worker_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel guard_cooldown=1" in text, "the cooling market must be binned"
    assert "candidates=0" in text
    open_mock.assert_not_awaited()
    assert "0xcool" not in state.positions


# ── 6. exits ignore the cooldown ───────────────────────────────────────────────


async def test_exit_paths_ignore_cooldown(farm_state: FarmState, monkeypatch):
    placed: list = []

    async def fake_place_market_order(client, token_id, side, amount):
        placed.append((token_id, side, amount))
        return "exit-oid"

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_place_market_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)

    mark_guard_pulled(farm_state, "market-A")
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("20")
    pos.yes_cost_basis = Decimal("6")

    await exit_position_leg(
        MagicMock(), farm_state, pos.market.yes_token_id, Decimal("20"), "market-A", "m1", "YES"
    )

    assert placed == [("tok-yes", "SELL", 20.0)], "an exit SELL must fire even while cooling"
