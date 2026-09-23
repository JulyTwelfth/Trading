"""The user-blacklist gate: a market whose condition_id is in state.excluded_markets must never
be opened or quoted. Two enforcement points are exercised here through the real reconcile_tick:

  1. The primary discovery gate (worker.py: `elif m.condition_id in state.excluded_markets`) —
     the market is binned "excluded" and never becomes a candidate.
  2. The §5d post-fill open-loop guard (worker.py: the `market.condition_id in
     state.excluded_markets` sub-clause of the or-chain) — a market that became a candidate but is
     added to the exclusion set mid-tick (e.g. a live blacklist_add firing during an await) is
     still not opened.

Mirrors the harness in test_event_exclusion.py.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.farm import worker as worker_mod
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market
from app.farm.worker import reconcile_tick


def _market(**ov) -> Market:
    base = dict(
        condition_id="0xexcluded",
        slug="excluded-market",
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


def _state() -> FarmState:
    config = FarmConfig(
        filters=_filters(), bankroll=Decimal("1000"), max_session_loss=Decimal("50")
    )
    return FarmState(config=config, wallet_address="0xabc")


def patch_common(monkeypatch):
    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    async def fake_fetch_midpoints(http, token_ids):
        return {}

    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)


# ── 1. Primary discovery gate ─────────────────────────────────────────────────


async def test_reconcile_bins_excluded_market_and_opens_nothing(monkeypatch, caplog):
    m = _market()
    state = _state()
    state.excluded_markets.add("0xexcluded")

    async def fake_markets(http):
        return [m]

    open_mock = AsyncMock()
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "open_position", open_mock)
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "filter_funnel excluded=1" in text
    assert "candidates=0" in text
    # Never opened.
    open_mock.assert_not_awaited()
    assert "0xexcluded" not in state.positions


async def test_reconcile_opens_market_not_excluded(monkeypatch, caplog):
    # Control: the SAME market, NOT excluded, sails through to candidate (proves the gate, not a
    # filter, is what binned it above).
    m = _market()
    state = _state()

    async def fake_markets(http):
        return [m]

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "open_position", AsyncMock())
    patch_common(monkeypatch)

    with caplog.at_level(logging.INFO, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidates=1" in text
    assert "excluded=" not in text


# ── 2. §5d post-fill open-loop guard (mid-tick exclusion / fill race) ──────────


async def test_open_loop_skips_market_excluded_mid_tick(monkeypatch, caplog):
    """A market passes the primary gate (becomes a candidate) but is added to excluded_markets
    before the open loop runs — simulating a live blacklist_add (or an earlier fill) landing during
    the fetch_midpoints await. The §5d guard must skip it, opening nothing."""
    m = _market()
    state = _state()  # excluded_markets starts EMPTY → m becomes a candidate

    async def fake_markets(http):
        return [m]

    # The mid-tick injection: fetch_midpoints both supplies usable quotes AND mutates the
    # exclusion set, as the live handler would while this await is in flight. Returning real
    # midpoints lets the candidate reach the open loop, where the §5d guard then trips.
    async def fake_fetch_midpoints(http, token_ids):
        state.excluded_markets.add("0xexcluded")
        return {tid: Decimal("0.5") for tid in token_ids}

    open_mock = AsyncMock()
    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "open_position", open_mock)

    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)

    with caplog.at_level(logging.DEBUG, logger="app.farm.worker"):
        await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    # It WAS a candidate this tick (primary gate let it through)...
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "candidates=1" in text
    # ...but the §5d guard skipped the open.
    open_mock.assert_not_awaited()
    assert "0xexcluded" not in state.positions


async def test_open_loop_opens_when_not_excluded_mid_tick(monkeypatch):
    """Control for the §5d guard: identical setup, but nothing mutates the exclusion set, so the
    market IS opened. Proves the skip above is caused by the exclusion, not the harness."""
    m = _market()
    state = _state()

    async def fake_markets(http):
        return [m]

    async def fake_fetch_midpoints(http, token_ids):
        return {tid: Decimal("0.5") for tid in token_ids}

    opened: list = []

    async def fake_open_position(client, state_, websocket, market, midpoints):
        opened.append(market.condition_id)

    async def fake_get_balance(addr):
        return Decimal("1000")

    async def fake_exit_held_legs(*args, **kwargs):
        return None

    monkeypatch.setattr(worker_mod, "fetch_eligible_markets", fake_markets)
    monkeypatch.setattr(worker_mod, "open_position", fake_open_position)
    monkeypatch.setattr(worker_mod, "get_balance", fake_get_balance, raising=False)
    monkeypatch.setattr(worker_mod, "exit_held_legs", fake_exit_held_legs)
    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_fetch_midpoints)

    await reconcile_tick(MagicMock(), MagicMock(), state, AsyncMock())

    assert opened == ["0xexcluded"], "a non-excluded candidate must be opened"
