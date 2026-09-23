import asyncio
import logging
from typing import Any

import httpx
from polymarket import errors as pm_errors
from py_clob_client_v2.exceptions import PolyApiException

from app.constants import CANCEL_RETRY_DELAY_SECONDS

logger = logging.getLogger(__name__)


async def cancel_order(client, order_id: str) -> Any:
    """Cancel a single order by its Polymarket order ID."""
    return await client.cancel_order(order_id)


async def cancel_all(client) -> Any:
    """Cancel ALL open orders for the wallet. Used as deadman switch on worker shutdown."""
    return await client.cancel_all()


async def cancel_orders(client, *order_ids: str) -> None:
    """Cancel a specific set of orders. Delegates to the execution adapter, which handles
    empty-id filtering, batch grouping, error swallowing, and not_canceled logging."""
    await client.cancel_orders(*order_ids)


def is_transient_cancel_error(exc: Exception) -> bool:
    """True for cancel failures worth retrying (network blips, rate limits, 5xx/429 responses);
    False for anything that looks like a permanent rejection."""
    if isinstance(
        exc, (pm_errors.TransportError, pm_errors.TimeoutError, pm_errors.RateLimitError)
    ):
        return True
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
        return True
    if isinstance(exc, (ConnectionError, OSError)):
        return True
    if isinstance(exc, pm_errors.RequestRejectedError):
        return "cloudflare" in str(exc).lower()
    if isinstance(exc, PolyApiException):
        status = getattr(exc, "status_code", None)
        return status is not None and (status >= 500 or status == 429)
    return False


async def cancel_order_with_retry(client, order_id: str) -> bool:
    """Cancel a single order, with one delayed retry on a transient failure. Never raises — True
    on a confirmed cancel, False if the retry also fails (or the first error wasn't transient)."""
    try:
        await cancel_order(client, order_id)
        return True
    except Exception as exc:
        if not is_transient_cancel_error(exc):
            logger.warning("cancel_order failed for %s: %s", order_id, exc)
            return False
        logger.warning("cancel_order failed for %s (transient), retrying once: %s", order_id, exc)
        await asyncio.sleep(CANCEL_RETRY_DELAY_SECONDS)
    try:
        await cancel_order(client, order_id)
        return True
    except Exception as exc:
        logger.warning("cancel_order retry failed for %s: %s", order_id, exc)
        return False


async def cancel_all_with_retry(client) -> bool:
    """Cancel ALL open orders, with one delayed retry on a transient failure. Never raises —
    True on a confirmed cancel_all, False if the retry also fails or the first error wasn't
    transient."""
    try:
        await cancel_all(client)
        return True
    except Exception as exc:
        if not is_transient_cancel_error(exc):
            logger.warning("cancel_all failed: %s", exc)
            return False
        logger.warning("cancel_all failed (transient), retrying once: %s", exc)
        await asyncio.sleep(CANCEL_RETRY_DELAY_SECONDS)
    try:
        await cancel_all(client)
        return True
    except Exception as exc:
        logger.warning("cancel_all retry failed: %s", exc)
        return False
