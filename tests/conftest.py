from datetime import datetime
from decimal import Decimal

import pytest

from app.farm.schemas import (
    FarmConfig,
    FarmFilters,
    FarmState,
    Market,
    MarketPosition,
    OrderInfo,
)


@pytest.fixture
def market() -> Market:
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
    )


@pytest.fixture
def farm_state(market: Market) -> FarmState:
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
        max_session_loss=Decimal(5),
    )
    state = FarmState(config=config)
    state.positions["market-A"] = MarketPosition(
        market=market,
        yes_order_id="yes-oid",
        no_order_id="no-oid",
        yes_price=Decimal("0.5"),
        no_price=Decimal("0.5"),
        yes_shares=Decimal("100"),
        yes_cost_basis=Decimal("50"),
        no_shares=Decimal("0"),
        no_cost_basis=Decimal("0"),
    )
    state.order_registry["yes-oid"] = OrderInfo(
        condition_id="market-A", outcome="YES", token_id=market.yes_token_id
    )
    state.order_registry["no-oid"] = OrderInfo(
        condition_id="market-A", outcome="NO", token_id=market.no_token_id
    )
    return state
