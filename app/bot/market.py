import json
from decimal import Decimal
from urllib.parse import urlparse

import httpx

from app.bot.schemas import OrderBook
from app.constants import CLOB_HOST, GAMMA_API


async def resolve_market(url: str) -> tuple[str, str, str]:
    slug = urlparse(url).path.rstrip("/").split("/")[-1]

    async with httpx.AsyncClient() as client:
        response = await client.get(f"{GAMMA_API}/events", params={"slug": slug})
        response.raise_for_status()

    events = response.json()
    if not events:
        raise ValueError(f"No event found for slug: {slug}")

    market = events[0]["markets"][0]
    token_ids = json.loads(market["clobTokenIds"])

    return token_ids[0], token_ids[1], market["question"]


async def resolve_event_markets(url: str) -> list[dict]:
    """Resolve a Polymarket URL to EVERY market under its event.
    Returns a list of {"condition_id", "slug", "question"} dicts (one per market).
    Raises ValueError if no event/markets resolve."""
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{GAMMA_API}/events", params={"slug": slug})
        response.raise_for_status()
    events = response.json()
    if not events:
        raise ValueError(f"No event found for slug: {slug}")
    markets = events[0].get("markets") or []
    out = []
    for m in markets:
        cid = m.get("conditionId")
        if not cid:
            continue
        out.append({"condition_id": cid, "slug": m.get("slug"), "question": m.get("question")})
    if not out:
        raise ValueError(f"Event has no markets with a conditionId: {slug}")
    return out


async def get_order_book(token_id: str) -> OrderBook:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CLOB_HOST}/book", params={"token_id": token_id})
        response.raise_for_status()
    return OrderBook.model_validate(response.json())


async def get_best_bid_ask(token_id: str) -> tuple[Decimal, Decimal]:
    book = await get_order_book(token_id)
    best_bid = max((level.price for level in book.bids), default=Decimal("0"))
    best_ask = min((level.price for level in book.asks), default=Decimal("1"))
    return best_bid, best_ask
