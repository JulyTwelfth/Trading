"""Expected reward rate: percentage × pool projection.

The bot derives its headline $/hr and $/24h from Polymarket's
`/rewards/user/percentages` endpoint, which returns the share (0-100) of each
market's daily reward pool the user is currently capturing. Multiplying each
share by that market's `rewards_rate_per_day` and summing gives the projected
day total; the hourly rate is just that / 24. These tests pin the join math,
the percent (not fraction) scaling, the skip of markets we don't hold, and the
poll loop's resilience to fetch failures and cancellation.
"""

import asyncio
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import rewards as rewards_mod
from app.farm.rewards import (
    expected_rewards_per_day,
    fetch_market_earnings,
    fetch_total_earnings,
    record_market_earnings,
    rewards_poll_loop,
)
from app.farm.schemas import MarketPosition


class Stop(Exception):
    """Sentinel to break out of the otherwise-infinite poll loop."""


def add_position(state, condition_id, rate):
    """Clone the fixture market into a second held position with a different
    condition_id and daily reward pool."""
    base = next(iter(state.positions.values())).market
    market = base.model_copy(
        update={"condition_id": condition_id, "rewards_rate_per_day": Decimal(str(rate))}
    )
    state.positions[condition_id] = MarketPosition(
        market=market,
        yes_order_id=f"{condition_id}-yes",
        no_order_id=f"{condition_id}-no",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
    )


def drive_loop(monkeypatch, earnings, percentages, stop_after):
    """Wire fetch_total_earnings / fetch_reward_percentages to yield the given
    side-effect lists and make asyncio.sleep raise _Stop on its `stop_after`-th
    call, so the while-True loop runs a fixed number of iterations."""
    monkeypatch.setattr(rewards_mod, "fetch_total_earnings", AsyncMock(side_effect=list(earnings)))
    monkeypatch.setattr(rewards_mod, "fetch_market_earnings", AsyncMock(return_value={}))
    monkeypatch.setattr(
        rewards_mod, "fetch_reward_percentages", AsyncMock(side_effect=list(percentages))
    )
    calls = {"n": 0}

    async def fake_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] >= stop_after:
            raise Stop

    monkeypatch.setattr(rewards_mod.asyncio, "sleep", fake_sleep)


# ── expected_rewards_per_day: the percentage × pool join ──────────────────────


def test_expected_per_day_joins_percentage_and_pool(farm_state):
    # market-A pool=$1/day; capturing 50% of it → $0.50/day projected.
    assert expected_rewards_per_day({"market-A": Decimal("50")}, farm_state.positions) == Decimal(
        "0.5"
    )


def test_expected_per_day_scales_percent_not_fraction(farm_state):
    # 10 means 10%, not 10×: $1 pool × 0.10 = $0.10.
    assert expected_rewards_per_day({"market-A": Decimal("10")}, farm_state.positions) == Decimal(
        "0.1"
    )


def test_expected_per_day_sums_across_markets(farm_state):
    add_position(farm_state, "market-B", rate=2)
    # A: 50% × $1 = 0.5; B: 25% × $2 = 0.5 → $1.00/day.
    percentages = {"market-A": Decimal("50"), "market-B": Decimal("25")}
    assert expected_rewards_per_day(percentages, farm_state.positions) == Decimal("1.0")


def test_expected_per_day_skips_unheld_markets(farm_state):
    # A percentage for a market we hold no position for has no pool on hand and
    # must be skipped, not crash or count as zero-pool noise.
    percentages = {"market-A": Decimal("50"), "ghost-market": Decimal("100")}
    assert expected_rewards_per_day(percentages, farm_state.positions) == Decimal("0.5")


def test_expected_per_day_zero_when_no_percentages(farm_state):
    assert expected_rewards_per_day({}, farm_state.positions) == Decimal("0")


# ── per-market reward attribution logging (for net-per-category analysis) ─────


def test_expected_per_day_logs_per_market_reward(farm_state, caplog):
    # Each held market with a positive share logs a reward_market line (slug + est_day) so
    # reward can be netted against fill-loss per market/category offline.
    with caplog.at_level(logging.INFO, logger="strat"):
        expected_rewards_per_day({"market-A": Decimal("50")}, farm_state.positions)
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.name == "strat" and r.getMessage().startswith("reward_market")
    ]
    assert len(lines) == 1
    assert "slug=" in lines[0] and "est_day=" in lines[0]


def test_expected_per_day_does_not_log_zero_share(farm_state, caplog):
    # A 0% share earns nothing — no per-market line (keeps the signal clean).
    with caplog.at_level(logging.INFO, logger="strat"):
        expected_rewards_per_day({"market-A": Decimal("0")}, farm_state.positions)
    assert not [r for r in caplog.records if r.getMessage().startswith("reward_market")]


async def test_fetch_total_earnings_delegates_to_adapter(caplog):
    # M2: fetch_total_earnings is now a thin shim that delegates to
    # client.total_earnings_today(). The parsing/summing logic lives in the adapter.
    client = AsyncMock()
    client.total_earnings_today = AsyncMock(return_value=Decimal("1.75"))
    with caplog.at_level(logging.INFO, logger="strat"):
        total = await fetch_total_earnings(client)
    assert total == Decimal("1.75")
    client.total_earnings_today.assert_awaited_once()
    assert not [r for r in caplog.records if r.getMessage().startswith("reward_earned")]


async def test_fetch_market_earnings_delegates_to_adapter():
    # M2: thin shim; parsing logic is in the adapter.
    client = AsyncMock()
    expected = {"c1": Decimal("2.00"), "c2": Decimal("0.25")}
    client.market_earnings_today = AsyncMock(return_value=expected)
    out = await fetch_market_earnings(client)
    assert out == expected
    client.market_earnings_today.assert_awaited_once()


def test_record_market_earnings_logs_session_delta(farm_state, caplog):
    # First poll baselines the cumulative (session=0); a later, higher cumulative logs the delta.
    with caplog.at_level(logging.INFO, logger="strat"):
        record_market_earnings(farm_state, {"market-A": Decimal("1.00")})
        record_market_earnings(farm_state, {"market-A": Decimal("1.75")})
    assert farm_state.market_rewards["market-A"] == Decimal("1.75")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("reward_earned")]
    # market-A is the fixture position → its slug ("m1") is used, not the raw condition_id.
    assert "slug=m1" in lines[-1]
    assert "earned_today=1.75" in lines[-1]
    assert "session=0.75" in lines[-1]  # 1.75 - 1.00 baseline


def test_record_market_earnings_handles_midnight_reset(farm_state):
    record_market_earnings(farm_state, {"market-A": Decimal("3.00")})  # baseline 3.00
    record_market_earnings(farm_state, {"market-A": Decimal("0.40")})  # daily counter reset
    # Baseline re-set to the post-rollover value so session-earned keeps counting from there.
    assert farm_state.market_rewards_baseline["market-A"] == Decimal("0.40")
    assert farm_state.market_rewards["market-A"] == Decimal("0.40")


def test_record_market_earnings_uses_condition_id_when_unheld(farm_state, caplog):
    # A market we no longer hold a position for falls back to the condition_id as the label.
    with caplog.at_level(logging.INFO, logger="strat"):
        record_market_earnings(farm_state, {"ghost-cid": Decimal("0.10")})
    line = next(
        r.getMessage() for r in caplog.records if r.getMessage().startswith("reward_earned")
    )
    assert "market=ghost-cid" in line and "slug=ghost-cid" in line


async def test_fetch_total_earnings_propagates_adapter_return_value():
    # M2: thin shim; verify the return value from the adapter is propagated unchanged.
    client = AsyncMock()
    client.total_earnings_today = AsyncMock(return_value=Decimal("1.50"))
    total = await fetch_total_earnings(client)
    assert total == Decimal("1.50")


# ── FarmState.rewards_per_hour / elapsed_seconds ─────────────────────────────


def test_rewards_per_hour_is_day_total_over_24(farm_state):
    farm_state.expected_rewards_per_day = Decimal("24")
    assert farm_state.rewards_per_hour() == Decimal("1")


def test_rewards_per_hour_zero_when_no_projection(farm_state):
    assert farm_state.expected_rewards_per_day == Decimal("0")
    assert farm_state.rewards_per_hour() == Decimal("0")


def test_elapsed_seconds_counts_forward(farm_state):
    from datetime import timedelta

    now = farm_state.started_at + timedelta(minutes=30)
    assert farm_state.elapsed_seconds(now) == 1800


def test_elapsed_seconds_clamps_clock_skew(farm_state):
    # A backwards clock (or started_at slightly in the future) must clamp to 0,
    # never go negative.
    from datetime import timedelta

    past = farm_state.started_at - timedelta(seconds=10)
    assert farm_state.elapsed_seconds(past) == 0


# ── rewards_poll_loop ─────────────────────────────────────────────────────────


async def test_loop_updates_earned_and_expected(farm_state, monkeypatch):
    drive_loop(
        monkeypatch,
        earnings=[Decimal("30")],
        percentages=[{"market-A": Decimal("50")}],
        stop_after=1,
    )
    with pytest.raises(Stop):
        await rewards_poll_loop(MagicMock(), farm_state)
    assert farm_state.rewards_earned == Decimal("30")
    assert farm_state.expected_rewards_per_day == Decimal("0.5")


async def test_loop_recomputes_expected_each_tick(farm_state, monkeypatch):
    # The projection is a snapshot of current positioning, not an accumulator:
    # a later, lower percentage must lower the rate, not stack on the earlier one.
    drive_loop(
        monkeypatch,
        earnings=[Decimal("10"), Decimal("11")],
        percentages=[{"market-A": Decimal("80")}, {"market-A": Decimal("20")}],
        stop_after=2,
    )
    with pytest.raises(Stop):
        await rewards_poll_loop(MagicMock(), farm_state)
    # Last tick: 20% × $1 = $0.20/day — the 80% reading does not persist.
    assert farm_state.expected_rewards_per_day == Decimal("0.2")


async def test_loop_survives_transient_failure(farm_state, monkeypatch):
    # A failed earnings fetch mid-stream must not crash the loop; the next good
    # iteration refreshes both numbers. Percentages is consumed only on the two
    # iterations whose earnings fetch succeeds.
    drive_loop(
        monkeypatch,
        earnings=[Decimal("10"), RuntimeError("boom"), Decimal("14")],
        percentages=[{"market-A": Decimal("20")}, {"market-A": Decimal("60")}],
        stop_after=3,
    )
    with pytest.raises(Stop):
        await rewards_poll_loop(MagicMock(), farm_state)
    assert farm_state.rewards_earned == Decimal("14")
    assert farm_state.expected_rewards_per_day == Decimal("0.6")


async def test_cancellation_propagates(farm_state, monkeypatch):
    # Clean shutdown: CancelledError must bubble out, not be swallowed by the
    # broad except that handles fetch errors.
    monkeypatch.setattr(
        rewards_mod, "fetch_total_earnings", AsyncMock(side_effect=asyncio.CancelledError)
    )
    with pytest.raises(asyncio.CancelledError):
        await rewards_poll_loop(MagicMock(), farm_state)


async def test_emits_farm_perf_strat_line(farm_state, monkeypatch, caplog):
    drive_loop(
        monkeypatch,
        earnings=[Decimal("13")],
        percentages=[{"market-A": Decimal("50")}],
        stop_after=1,
    )
    with caplog.at_level(logging.INFO, logger="strat"):
        with pytest.raises(Stop):
            await rewards_poll_loop(MagicMock(), farm_state)
    perf = [
        r.getMessage()
        for r in caplog.records
        if r.name == "strat" and r.getMessage().startswith("farm_perf")
    ]
    assert perf, "expected at least one farm_perf strat line"
    last = perf[-1]
    for key in (
        "earned_today=",
        "expected_per_day=",
        "per_hour=",
        "elapsed_s=",
        "markets=",
        "volume=",
    ):
        assert key in last
    # markets reflects the single position in the fixture.
    assert "markets=1" in last
