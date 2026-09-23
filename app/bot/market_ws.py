import asyncio
import json
import logging
import ssl
from collections.abc import AsyncIterator

import certifi
import websockets
from pydantic import ValidationError

from app.bot.schemas import (
    BestBidAsk,
    BookSnapshot,
    LastTradePrice,
    MarketEvent,
    PriceChange,
    TickSizeChange,
)
from app.constants import MARKET_WS_RECONNECT_DELAY_SECONDS, WS_MARKET_URL

# event_type -> model for the frames we consume off the market channel.
FRAME_MODELS = {
    "best_bid_ask": BestBidAsk,
    "tick_size_change": TickSizeChange,
    "book": BookSnapshot,
    "price_change": PriceChange,
    "last_trade_price": LastTradePrice,
}

logger = logging.getLogger(__name__)


async def stream_market_events(token_ids: list[str]) -> AsyncIterator[MarketEvent]:
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    subscribe = json.dumps(
        {
            "assets_ids": token_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
    )

    while True:
        try:
            async with websockets.connect(WS_MARKET_URL, ssl=ssl_ctx) as ws:
                await ws.send(subscribe)
                logger.info("market_ws: connected, assets=%d", len(token_ids))
                async for message in ws:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("market_ws: non-json frame ignored")
                        continue
                    frames = data if isinstance(data, list) else [data]
                    for frame in frames:
                        model = FRAME_MODELS.get(frame.get("event_type"))
                        if model is None:
                            continue
                        try:
                            yield model.model_validate(frame)
                        except ValidationError:
                            logger.warning(
                                "market_ws: dropping unparseable %s frame: %s",
                                frame.get("event_type"),
                                frame,
                            )
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            logger.warning("market_ws: connection closed (%s); reconnecting", exc)
            await asyncio.sleep(MARKET_WS_RECONNECT_DELAY_SECONDS)
        except Exception:
            logger.exception("market_ws: unexpected error; reconnecting")
            await asyncio.sleep(MARKET_WS_RECONNECT_DELAY_SECONDS)
