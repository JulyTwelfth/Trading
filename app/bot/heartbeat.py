import asyncio
import logging
import time
from typing import Any

from py_clob_client_v2.exceptions import PolyApiException

from app.constants import (
    HEARTBEAT_ID_ROTATION_SECONDS,
    HEARTBEAT_INTERVAL_SECONDS,
    HEARTBEAT_RETRY_DELAY_SECONDS,
)

logger = logging.getLogger(__name__)


async def send_heartbeat(client: Any, heartbeat_id: str = "") -> dict[str, Any]:
    """Send one heartbeat via the raw ClobClient. Pass the heartbeat_id from the previous
    response; first call uses an empty string and Polymarket issues a fresh ID."""
    return await asyncio.to_thread(client.post_heartbeat, heartbeat_id)


def corrected_heartbeat_id(exc: PolyApiException) -> str:
    """A 400 from /heartbeats carries the correct heartbeat_id in its body
    ({"heartbeat_id": ..., "error_msg": "Invalid Heartbeat ID"}); empty string
    if the body didn't include one."""
    if isinstance(exc.error_msg, dict):
        return str(exc.error_msg.get("heartbeat_id") or "")
    return ""


async def heartbeat_loop(
    client: Any,
    interval: float = HEARTBEAT_INTERVAL_SECONDS,
    rotation_interval: float = HEARTBEAT_ID_ROTATION_SECONDS,
    retry_delay: float = HEARTBEAT_RETRY_DELAY_SECONDS,
) -> None:
    if not getattr(client, "supports_heartbeat", True):
        logger.info(
            "heartbeat: disabled for %s wallet — CLOB deadman unavailable",
            getattr(client, "wallet_type", "?"),
        )
        return
    heartbeat_id = ""
    last_rotation = time.monotonic()
    while True:
        if heartbeat_id and time.monotonic() - last_rotation >= rotation_interval:
            heartbeat_id = ""
        delay = interval
        try:
            response = await client.send_heartbeat(heartbeat_id)
            if not heartbeat_id:
                last_rotation = time.monotonic()
            heartbeat_id = response.get("heartbeat_id", "")
            logger.debug("heartbeat ok")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(exc, PolyApiException) and exc.status_code == 400:
                # The 400 body carries the heartbeat_id the server expects
                # (sending "" mid-session is also rejected). Adopt it and retry
                # quickly: a rejected beat doesn't count toward the ~15s
                # no-heartbeat window after which all open orders are cancelled.
                heartbeat_id = corrected_heartbeat_id(exc)
                last_rotation = time.monotonic()
                delay = retry_delay
                logger.warning(
                    "heartbeat rejected: %s — retrying with server-corrected id %r",
                    exc,
                    heartbeat_id,
                )
            else:
                # Transient failure (network, 5xx): the id we hold is likely
                # still valid, so keep it; if it expired meanwhile, the next
                # beat gets a 400 with the corrected id and recovers above.
                logger.warning("heartbeat failed: %s — keeping heartbeat_id", exc)
        await asyncio.sleep(delay)
