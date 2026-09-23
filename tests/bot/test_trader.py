"""trader.py is now a thin delegation shim; these tests verify it forwards correctly
to the execution adapter's methods and propagates the returned value."""

from unittest.mock import AsyncMock

from app.bot.schemas import LimitOrder
from app.bot.trader import place_limit_order, place_market_order


async def test_place_limit_order_delegates_and_returns_id():
    client = AsyncMock()
    client.place_limit_order = AsyncMock(return_value="oid-123")
    order = LimitOrder(token_id="tok-1", side="BUY", size=100.0, price=0.48)

    result = await place_limit_order(client, order, post_only=True)

    assert result == "oid-123"
    client.place_limit_order.assert_awaited_once_with(order, post_only=True)


async def test_place_limit_order_post_only_defaults_false():
    client = AsyncMock()
    client.place_limit_order = AsyncMock(return_value="x")
    order = LimitOrder(token_id="t", side="SELL", size=5.0, price=0.9)

    await place_limit_order(client, order)

    client.place_limit_order.assert_awaited_once_with(order, post_only=False)


async def test_place_market_order_delegates_and_returns_id():
    client = AsyncMock()
    client.place_market_order = AsyncMock(return_value="mkt-9")

    result = await place_market_order(client, "tok-2", "SELL", 25.0)

    assert result == "mkt-9"
    client.place_market_order.assert_awaited_once_with("tok-2", "SELL", 25.0)
