"""cancel.py is now a thin delegation shim; these tests verify it forwards to the
execution adapter and propagates the returned value."""

from unittest.mock import AsyncMock

from app.bot.cancel import cancel_all, cancel_order, cancel_orders


async def test_cancel_order_delegates_and_returns_result():
    client = AsyncMock()
    client.cancel_order = AsyncMock(return_value={"canceled": ["oid-1"]})

    result = await cancel_order(client, "oid-1")

    assert result == {"canceled": ["oid-1"]}
    client.cancel_order.assert_awaited_once_with("oid-1")


async def test_cancel_all_delegates_and_returns_result():
    client = AsyncMock()
    client.cancel_all = AsyncMock(return_value={"canceled": "all"})

    result = await cancel_all(client)

    assert result == {"canceled": "all"}
    client.cancel_all.assert_awaited_once_with()


async def test_cancel_orders_delegates_with_positional_ids():
    client = AsyncMock()
    client.cancel_orders = AsyncMock(return_value=None)

    await cancel_orders(client, "a", "b")

    client.cancel_orders.assert_awaited_once_with("a", "b")


async def test_cancel_orders_no_ids_delegates_empty_call():
    # The shim passes through; all filtering logic lives in the adapter.
    client = AsyncMock()
    client.cancel_orders = AsyncMock(return_value=None)

    await cancel_orders(client)

    client.cancel_orders.assert_awaited_once_with()
