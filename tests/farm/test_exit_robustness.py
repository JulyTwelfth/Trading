"""#3 exit/open robustness:
A) DUST-strand fix — a residual below min_order_size is un-sellable, so exit_position_leg
   writes it off (clears tracking) instead of re-driving the FAK/GTC forever.
B) naked-leg fix — a partial open's rollback cancel is retried so a failed cancel doesn't
   strand an untracked resting BUY."""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import exits as exits_mod
from app.farm import worker as worker_mod
from app.farm.exits import exit_held_legs, exit_position_leg
from app.farm.schemas import FarmConfig, FarmFilters, FarmState, Market

# ── A) dust write-off (exit_position_leg) ────────────────────────────────────


@pytest.fixture
def stub_sells(monkeypatch):
    sells: list = []

    async def fake_market_order(client, token_id, side, amount):
        sells.append(("FAK", token_id, side, amount))
        return "fak-oid"

    async def fake_limit_order(client, order, post_only=False):
        sells.append(("GTC", order.token_id, order.side, order.size))
        return "gtc-oid"

    async def fake_cancel_orders(client, *oids):
        return None

    monkeypatch.setattr(exits_mod, "place_market_order", fake_market_order)
    monkeypatch.setattr(exits_mod, "place_limit_order", fake_limit_order)
    monkeypatch.setattr(exits_mod, "cancel_orders", fake_cancel_orders)
    return sells


async def leg(state, size):
    # market-A min_order_size is 5 (conftest).
    pos = state.positions["market-A"]
    await exit_position_leg(
        MagicMock(), state, pos.market.yes_token_id, Decimal(str(size)), "market-A", "m1", "YES"
    )


async def test_sub_min_residual_written_off_no_sell(farm_state: FarmState, stub_sells):
    farm_state.positions["market-A"].yes_shares = Decimal("3.5")  # < min 5
    await leg(farm_state, "3.5")
    assert stub_sells == [], "must not attempt a sell on un-sellable dust"
    assert farm_state.positions["market-A"].yes_shares == Decimal("0"), "dust written off"


async def test_tiny_dust_written_off(farm_state: FarmState, stub_sells):
    farm_state.positions["market-A"].yes_shares = Decimal("0.007")
    await leg(farm_state, "0.007")
    assert stub_sells == []
    assert farm_state.positions["market-A"].yes_shares == Decimal("0")


async def test_exactly_min_size_sells_normally(farm_state: FarmState, stub_sells):
    farm_state.positions["market-A"].yes_shares = Decimal("5")  # == min, NOT dust
    await leg(farm_state, "5")
    assert stub_sells == [("FAK", "tok-yes", "SELL", 5.0)], "min-size order is sellable"


async def test_above_min_sells_normally(farm_state: FarmState, stub_sells):
    farm_state.positions["market-A"].yes_shares = Decimal("50")
    await leg(farm_state, "50")
    assert stub_sells and stub_sells[0][0] == "FAK"


async def test_sweep_writes_off_dust_then_stops(farm_state: FarmState, stub_sells):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("2")  # dust
    pos.no_shares = Decimal("0")
    await exit_held_legs(MagicMock(), farm_state, cancel_resting=False, skip_in_flight=False)
    assert stub_sells == []
    assert pos.yes_shares == Decimal("0")
    # A second sweep has nothing left to re-drive (the infinite loop is gone).
    await exit_held_legs(MagicMock(), farm_state, cancel_resting=False, skip_in_flight=False)
    assert stub_sells == []


# ── B) rollback-cancel retry (open_position) ─────────────────────────────────


def _market() -> Market:
    return Market(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="cy",
        no_token_id="cn",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("50"),
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


def _state() -> FarmState:
    f = FarmFilters(
        vol_min=Decimal(0),
        vol_max=Decimal(1e9),
        liq_min=Decimal(0),
        liq_max=Decimal(1e9),
        spread_min=Decimal(0),
        spread_max=Decimal(100),
        reward_min=Decimal(0),
        time_remaining="all",
        created_date="all",
        change_24h="all",
    )
    cfg = FarmConfig(filters=f, bankroll=Decimal(1000), max_session_loss=Decimal(50))
    return FarmState(config=cfg)


def partial_open(monkeypatch):
    """YES leg places, NO leg rejects → triggers the rollback."""
    placed = []

    async def fake_place(client, order, post_only=True):
        placed.append(order.token_id)
        if len(placed) == 1:
            return "yes-oid"
        raise RuntimeError("NO leg rejected: not enough balance")

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)


async def test_rollback_cancel_retried_until_success(monkeypatch):
    partial_open(monkeypatch)
    cancels = []

    async def flaky_cancel(client, oid):
        cancels.append(oid)
        if len(cancels) < 3:  # fail twice, succeed on the 3rd
            raise RuntimeError("cancel network error")
        return None

    monkeypatch.setattr(worker_mod, "cancel_order", flaky_cancel)
    state = _state()
    await worker_mod.open_position(
        MagicMock(), state, AsyncMock(), _market(), {"cy": Decimal("0.5"), "cn": Decimal("0.5")}
    )
    assert cancels == ["yes-oid", "yes-oid", "yes-oid"], "retried the rollback cancel to success"
    assert "c" not in state.positions, "partial open is not recorded as a position"


async def test_rollback_cancel_exhausts_and_warns(monkeypatch, caplog):
    partial_open(monkeypatch)
    cancels = []

    async def always_fail(client, oid):
        cancels.append(oid)
        raise RuntimeError("cancel down")

    monkeypatch.setattr(worker_mod, "cancel_order", always_fail)
    state = _state()
    with caplog.at_level(logging.ERROR, logger="app.farm.worker"):
        await worker_mod.open_position(
            MagicMock(), state, AsyncMock(), _market(), {"cy": Decimal("0.5"), "cn": Decimal("0.5")}
        )
    assert len(cancels) == 3, "tried 3 times before giving up"
    assert "naked resting leg" in "\n".join(r.getMessage() for r in caplog.records)
