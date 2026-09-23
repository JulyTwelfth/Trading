"""Taker-fee tracking: the protocol skims a taker fee off every FAK exit, and the bot
must book it against session_loss so the kill switch and P&L aren't blind to it."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.bot.schemas import UserTrade
from app.farm.discovery import extract_taker_fee_rate
from app.farm.fees import taker_fee
from app.farm.fills import handle_trade
from app.farm.kill_switch import record_realized_pnl
from app.farm.schemas import (
    FarmConfig,
    FarmFilters,
    FarmState,
    Market,
    MarketPosition,
)

# --- taker_fee: verified against Polymarket's published per-category fee tables ---
# https://docs.polymarket.com/trading/fees  (100-share tables; fee peaks at p=0.50).


@pytest.mark.parametrize(
    "rate, price, expected",
    [
        (Decimal("0.07"), Decimal("0.50"), Decimal("1.75")),  # crypto peak
        (Decimal("0.07"), Decimal("0.10"), Decimal("0.63")),  # crypto @ 0.10
        (Decimal("0.07"), Decimal("0.90"), Decimal("0.63")),  # symmetric around 0.50
        (Decimal("0.03"), Decimal("0.50"), Decimal("0.75")),  # sports peak
        (Decimal("0.04"), Decimal("0.50"), Decimal("1.00")),  # finance/politics peak
        (Decimal("0.05"), Decimal("0.50"), Decimal("1.25")),  # economics/culture peak
    ],
)
def test_taker_fee_matches_published_tables(rate, price, expected):
    assert taker_fee(Decimal("100"), price, rate) == expected


def test_taker_fee_zero_for_fee_free_market():
    assert taker_fee(Decimal("100"), Decimal("0.5"), Decimal("0")) == Decimal(0)


@pytest.mark.parametrize("price", [Decimal("0"), Decimal("1"), Decimal("-0.1"), Decimal("1.5")])
def test_taker_fee_zero_at_degenerate_prices(price):
    assert taker_fee(Decimal("100"), price, Decimal("0.07")) == Decimal(0)


def test_taker_fee_rounds_sub_min_to_zero():
    # 0.0001 shares * 0.07 * 0.5 * 0.5 = 0.00000175 USDC, below the 0.00001 floor.
    assert taker_fee(Decimal("0.0001"), Decimal("0.5"), Decimal("0.07")) == Decimal(0)


# --- extract_taker_fee_rate: sourcing the rate from Gamma market data ---


def test_extract_fee_rate_prefers_schedule_rate():
    # takerBaseFee (1000bps) disagrees with the schedule rate (0.07); schedule wins.
    gamma = {
        "feesEnabled": True,
        "takerBaseFee": 1000,
        "feeSchedule": {"exponent": 1, "rate": 0.07, "takerOnly": True},
    }
    assert extract_taker_fee_rate(gamma) == Decimal("0.07")


def test_extract_fee_rate_zero_when_fees_disabled():
    assert extract_taker_fee_rate({"feesEnabled": False, "feeSchedule": {"rate": 0.07}}) == Decimal(
        0
    )


def test_extract_fee_rate_falls_back_to_taker_base_fee_bps():
    assert extract_taker_fee_rate({"feesEnabled": True, "takerBaseFee": 400}) == Decimal("0.04")


def test_extract_fee_rate_zero_when_absent():
    assert extract_taker_fee_rate({}) == Decimal(0)
    assert extract_taker_fee_rate({"feesEnabled": True}) == Decimal(0)


# --- record_realized_pnl: the fee adds directly to session_loss ---


def _bare_state() -> FarmState:
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
        bankroll=Decimal(100),
        max_session_loss=Decimal(100),
    )
    return FarmState(config=config)


def test_record_realized_pnl_books_fee_into_session_loss():
    state = _bare_state()
    # Break-even trade (entry == proceeds) with a $0.36 taker fee => a $0.36 loss.
    record_realized_pnl(
        state, Decimal("12"), Decimal("30"), Decimal("0.40"), "slug", "YES", fee=Decimal("0.36")
    )
    assert state.session_loss == Decimal("0.36")


def test_record_realized_pnl_default_fee_is_zero():
    state = _bare_state()
    record_realized_pnl(state, Decimal("15"), Decimal("30"), Decimal("0.40"), "slug", "YES")
    assert state.session_loss == Decimal("3")  # 15 cost - 12 proceeds, no fee


# --- end to end: an exit fill on a fee-enabled market charges the fee ---


def _fee_market(rate: str) -> Market:
    return Market(
        condition_id="market-A",
        slug="m1",
        question="?",
        yes_token_id="tok-yes",
        no_token_id="tok-no",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("1"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1),
        created_at=datetime(2025, 1, 1),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
        taker_fee_rate=Decimal(rate),
    )


def test_market_rejects_fee_rate_above_one():
    # A garbage upstream rate (>1, e.g. bps mistakenly passed as a fraction) must fail
    # validation — build_market catches the ValidationError (a ValueError) and skips
    # the market with a warning — rather than silently inflate every loss estimate.
    with pytest.raises(ValidationError):
        _fee_market("1.5")


def _exit_trade(size: str, price: str) -> UserTrade:
    return UserTrade(
        event_type="trade",
        id="t-exit-1",
        asset_id="tok-yes",
        market="market-A",
        side="SELL",
        price=Decimal(price),
        size=Decimal(size),
        outcome="YES",
        status="MINED",
        timestamp="2026-07-01T10:00:00Z",
        maker_orders=[],
        taker_order_id="exit-oid-1",
    )


async def test_exit_fill_on_fee_market_books_fee():
    state = _bare_state()
    state.positions["market-A"] = MarketPosition(
        market=_fee_market("0.05"),
        yes_order_id="yes-oid",
        no_order_id="no-oid",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=Decimal("100"),
        yes_cost_basis=Decimal("50"),
    )
    state.pending_exit_order_ids.add("exit-oid-1")

    # Sell all 100 shares @ 0.40: entry 50, proceeds 40 => 10 spread loss, plus a taker
    # fee of 100 * 0.05 * 0.40 * 0.60 = 1.20 => session_loss 11.20 (not 10.00).
    await handle_trade(MagicMock(), _exit_trade("100", "0.40"), state, AsyncMock())

    assert state.session_loss == Decimal("11.20")


async def test_exit_fill_on_fee_free_market_unchanged():
    state = _bare_state()
    state.positions["market-A"] = MarketPosition(
        market=_fee_market("0"),
        yes_order_id="yes-oid",
        no_order_id="no-oid",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=Decimal("100"),
        yes_cost_basis=Decimal("50"),
    )
    state.pending_exit_order_ids.add("exit-oid-1")

    await handle_trade(MagicMock(), _exit_trade("100", "0.40"), state, AsyncMock())

    assert state.session_loss == Decimal("10")  # 50 - 40, no fee
