import asyncio
import json
import logging
import ssl
from collections.abc import AsyncIterator
from typing import Any

import certifi
import websockets
from pydantic import ValidationError

from app.bot.schemas import UserTrade
from app.constants import USER_WS_RECONNECT_DELAY_SECONDS, WS_USER_URL

logger = logging.getLogger(__name__)


async def stream_user_trades(client: Any) -> AsyncIterator[UserTrade]:
    """Connect to Polymarket's authenticated user WS and yield trade frames.
    Skips non-trade events. Reconnects on disconnect; re-raises CancelledError."""
    auth = client.ws_auth()
    subscribe = json.dumps({"auth": auth, "type": "user", "markets": []})

    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    while True:
        try:
            async with websockets.connect(WS_USER_URL, ssl=ssl_ctx) as ws:
                await ws.send(subscribe)
                logger.info("user_ws: connected and subscribed")
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("user_ws: non-json frame ignored")
                        continue
                    frames = data if isinstance(data, list) else [data]
                    for frame in frames:
                        if frame.get("event_type") != "trade":
                            continue
                        try:
                            yield UserTrade.model_validate(frame)
                        except ValidationError:
                            logger.exception("user_ws: failed to parse trade frame")
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            logger.warning("user_ws: connection closed (%s); reconnecting", exc)
            await asyncio.sleep(USER_WS_RECONNECT_DELAY_SECONDS)
        except Exception:
            logger.exception("user_ws: unexpected error; reconnecting")
            await asyncio.sleep(USER_WS_RECONNECT_DELAY_SECONDS)
