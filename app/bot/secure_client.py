import asyncio
import logging

from polymarket import CancelOrdersResponse, SecureClient

from app.bot.schemas import LimitOrder
from app.exceptions import SecureOrderError

logger = logging.getLogger(__name__)


def build_secure_client(private_key: str, proxy_address: str | None = None) -> SecureClient:
    """Construct a SecureClient for a deposit (1271) wallet. Sync; bootstraps L2 creds
    over the network. Passing proxy_address avoids the relayer RPC and validates the proxy
    derives from the EOA (raises UserInputError otherwise)."""
    return SecureClient.create(private_key=private_key, wallet=proxy_address)


def wallet_type_for(client: SecureClient) -> str:
    return client.wallet_type


def is_deposit_wallet(client: SecureClient) -> bool:
    return client.wallet_type == "DEPOSIT_WALLET"


async def place_limit_order(
    client: SecureClient, order: LimitOrder, post_only: bool = False
) -> str:
    result = await asyncio.to_thread(
        client.place_limit_order,
        token_id=order.token_id,
        price=order.price,
        size=order.size,
        side=order.side,
        post_only=post_only,
    )
    if result.ok:
        return result.order_id
    raise SecureOrderError(result.code, result.message)


async def place_market_order(
    client: SecureClient, token_id: str, side: str, amount: float
) -> str:
    # BUY orders use `amount` (USDC); SELL orders use `shares`. The legacy MarketOrderArgsV2
    # docstring confirms: "SELL orders: Shares to sell" — same quantity, different kwarg.
    size_kwargs = {"shares": amount} if side == "SELL" else {"amount": amount}
    result = await asyncio.to_thread(
        client.place_market_order,
        token_id=token_id,
        side=side,
        order_type="FAK",
        **size_kwargs,
    )
    if result.ok:
        return result.order_id
    raise SecureOrderError(result.code, result.message)


async def cancel_order(client: SecureClient, order_id: str) -> CancelOrdersResponse:
    return await asyncio.to_thread(client.cancel_order, order_id=order_id)


async def cancel_all(client: SecureClient) -> CancelOrdersResponse:
    return await asyncio.to_thread(client.cancel_all)


async def cancel_orders(client: SecureClient, *order_ids: str) -> None:
    ids = [oid for oid in order_ids if oid]
    if not ids:
        return
    try:
        result = await asyncio.to_thread(client.cancel_orders, order_ids=ids)
    except Exception as exc:
        logger.warning("cancel_orders: batch cancel of %d ids failed: %s", len(ids), exc)
        return
    not_canceled = getattr(result, "not_canceled", None) or {}
    for oid, reason in not_canceled.items():
        logger.debug("cancel_orders: id %s not canceled: %s", oid, reason)
