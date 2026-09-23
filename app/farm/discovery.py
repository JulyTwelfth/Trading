import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

import httpx
from dateutil import parser as dateutil_parser

from app.bot.schemas import OrderBook
from app.constants import (
    BATCH_PRICES_HISTORY_LIMIT,
    CLOB_HOST,
    DATA_API,
    EXIT_DUST_BALANCE_SHARES,
    GAMMA_API,
    GAMMA_MARKETS_BATCH_LIMIT,
    GAMMA_MARKETS_MAX_RETRIES,
    GAMMA_RETRY_BASE_DELAY_SECONDS,
    PAGINATION_END_CURSOR,
    SPREADS_BATCH_LIMIT,
)
from app.farm.schemas import Market

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OnchainPosition:
    """A live outcome-token holding for a wallet, from the Polymarket Data API."""

    token_id: str
    condition_id: str
    outcome: str
    slug: str
    size: Decimal
    avg_price: Decimal


async def fetch_open_positions(
    http: httpx.AsyncClient, wallet_address: str
) -> list[OnchainPosition]:
    """Current outcome-token holdings for a wallet (Data API /positions). Used at startup to
    find shares stranded by a prior crash/restart — the farm holds no inventory by design, so
    anything returned here is residue to liquidate. Returns [] on any error (best-effort)."""
    try:
        response = await http.get(
            f"{DATA_API}/positions",
            params={"user": wallet_address, "sizeThreshold": str(EXIT_DUST_BALANCE_SHARES)},
        )
        response.raise_for_status()
    except Exception:
        logger.exception("fetch_open_positions failed for %s", wallet_address)
        return []
    out: list[OnchainPosition] = []
    for p in response.json():
        try:
            size = Decimal(str(p["size"]))
            if size <= 0:
                continue
            out.append(
                OnchainPosition(
                    token_id=str(p["asset"]),
                    condition_id=str(p.get("conditionId") or ""),
                    outcome=str(p.get("outcome") or "").strip().upper(),
                    slug=str(p.get("slug") or p.get("title") or p.get("conditionId") or "unknown"),
                    size=size,
                    avg_price=Decimal(str(p.get("avgPrice") or "0")),
                )
            )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            logger.warning("fetch_open_positions: skipping malformed entry %s", p)
    return out


async def fetch_eligible_markets(http: httpx.AsyncClient) -> list[Market]:
    """Discover reward-eligible markets and return enriched Market objects:
    /sampling-simplified-markets → Gamma /markets enrichment → /spreads → Market (skip failures)."""
    sampling = await fetch_sampling_simplified(http)
    if not sampling:
        logger.warning(
            "discovery: sampling-simplified returned 0 markets — skipping tick "
            "(held positions deferred, not churned, if this is transient)"
        )
        return []

    condition_ids = [m["condition_id"] for m in sampling]
    token_ids = [t["token_id"] for m in sampling for t in m.get("tokens", [])]

    gamma, spreads = await asyncio.gather(
        fetch_gamma_markets(http, condition_ids),
        fetch_spreads(http, token_ids),
    )

    markets: list[Market] = []
    for raw in sampling:
        market = build_market(raw, gamma, spreads)
        if market is not None:
            markets.append(market)
    logger.info(
        "discovery sampling=%d built=%d dropped=%d",
        len(sampling),
        len(markets),
        len(sampling) - len(markets),
    )
    return markets


async def fetch_sampling_simplified(http: httpx.AsyncClient) -> list[dict]:
    """Paginated fetch of /sampling-simplified-markets via keyset cursor."""
    results: list[dict] = []
    cursor = ""
    while True:
        params = {"next_cursor": cursor} if cursor else {}
        response = await http.get(f"{CLOB_HOST}/sampling-simplified-markets", params=params)
        response.raise_for_status()
        body = response.json()
        results.extend(body.get("data", []))
        cursor = body.get("next_cursor", "")
        if not cursor or cursor == PAGINATION_END_CURSOR:
            break
    return results


async def fetch_gamma_markets(http: httpx.AsyncClient, condition_ids: list[str]) -> dict[str, dict]:
    """Fetch Gamma /markets for given condition_ids in batches. Returns dict keyed by conditionId.

    Each condition_id is ~66 chars; passing 100+ in one query overflows URL length limits.
    We chunk into GAMMA_MARKETS_BATCH_LIMIT IDs per request.

    Note: Gamma uses camelCase keys (`conditionId`, `volume24hr`, `endDate`, `createdAt`).
    """
    if not condition_ids:
        return {}
    result: dict[str, dict] = {}
    for i in range(0, len(condition_ids), GAMMA_MARKETS_BATCH_LIMIT):
        batch = condition_ids[i : i + GAMMA_MARKETS_BATCH_LIMIT]
        for m in await fetch_gamma_batch(http, batch):
            if "conditionId" in m:
                result[m["conditionId"]] = m
    return result


async def fetch_gamma_batch(http: httpx.AsyncClient, batch: list[str]) -> list[dict]:
    """GET one Gamma /markets batch with exponential-backoff retries; raises once exhausted so the
    tick skips rather than churns positions on partial discovery (real outage fails fast, ~1.5s)."""
    last_exc: Exception | None = None
    for attempt in range(GAMMA_MARKETS_MAX_RETRIES):
        try:
            response = await http.get(
                f"{GAMMA_API}/markets",
                params=[("condition_ids", cid) for cid in batch],
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            last_exc = exc
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if 400 <= status < 500 and status != 429:
                    raise
            if attempt + 1 < GAMMA_MARKETS_MAX_RETRIES:
                logger.warning(
                    "gamma batch (%d ids) failed (attempt %d/%d): %s — retrying",
                    len(batch),
                    attempt + 1,
                    GAMMA_MARKETS_MAX_RETRIES,
                    exc,
                )
                await asyncio.sleep(GAMMA_RETRY_BASE_DELAY_SECONDS * 2**attempt)
    assert last_exc is not None
    raise last_exc


async def fetch_spreads(http: httpx.AsyncClient, token_ids: list[str]) -> dict[str, Decimal]:
    """POST /spreads in batches of <=500. Returns dict of token_id -> spread in CENTS (the API
    sends decimal probability, e.g. "0.04" = 4 cents; we multiply by 100)."""
    if not token_ids:
        return {}
    spreads: dict[str, Decimal] = {}
    for i in range(0, len(token_ids), SPREADS_BATCH_LIMIT):
        batch = token_ids[i : i + SPREADS_BATCH_LIMIT]
        response = await http.post(
            f"{CLOB_HOST}/spreads",
            json=[{"token_id": t} for t in batch],
        )
        response.raise_for_status()
        for tok, spread in response.json().items():
            spreads[tok] = Decimal(str(spread)) * 100
    return spreads


async def fetch_midpoints(http: httpx.AsyncClient, token_ids: list[str]) -> dict[str, Decimal]:
    """POST /midpoints in batches of <=500. Returns dict of token_id -> midpoint as Decimal."""
    if not token_ids:
        return {}
    out: dict[str, Decimal] = {}
    for i in range(0, len(token_ids), SPREADS_BATCH_LIMIT):
        batch = token_ids[i : i + SPREADS_BATCH_LIMIT]
        response = await http.post(
            f"{CLOB_HOST}/midpoints",
            json=[{"token_id": t} for t in batch],
        )
        response.raise_for_status()
        for tok, mid in response.json().items():
            out[tok] = Decimal(str(mid))
    return out


async def fetch_books(http: httpx.AsyncClient, token_ids: list[str]) -> dict[str, OrderBook]:
    """POST /books in batches of <=500. Returns dict of token_id -> OrderBook, keyed by asset_id
    (/books returns a LIST, unlike /spreads and /midpoints; last_trade_price is ignored)."""
    if not token_ids:
        return {}
    out: dict[str, OrderBook] = {}
    for i in range(0, len(token_ids), SPREADS_BATCH_LIMIT):
        batch = token_ids[i : i + SPREADS_BATCH_LIMIT]
        response = await http.post(
            f"{CLOB_HOST}/books",
            json=[{"token_id": t} for t in batch],
        )
        response.raise_for_status()
        for raw in response.json():
            book = OrderBook.model_validate(raw)
            out[book.asset_id] = book
    return out


async def fetch_price_ranges(http: httpx.AsyncClient, token_ids: list[str]) -> dict[str, Decimal]:
    """POST /batch-prices-history (<=20 tokens/call, interval=1d). Returns token_id -> 24h price
    RANGE (max-min: the swing oneDayPriceChange understates); no-history tokens are omitted."""
    if not token_ids:
        return {}
    out: dict[str, Decimal] = {}
    for i in range(0, len(token_ids), BATCH_PRICES_HISTORY_LIMIT):
        batch = token_ids[i : i + BATCH_PRICES_HISTORY_LIMIT]
        response = await http.post(
            f"{CLOB_HOST}/batch-prices-history",
            json={"markets": batch, "interval": "1d", "fidelity": 60},
        )
        response.raise_for_status()
        history = response.json().get("history", {})
        for tok, points in history.items():
            prices = [Decimal(str(pt["p"])) for pt in points if "p" in pt]
            if prices:
                out[tok] = max(prices) - min(prices)
    return out


def build_market(raw: dict, gamma: dict[str, dict], spreads: dict[str, Decimal]) -> Market | None:
    """Map raw API responses into our internal Market model. Returns None on missing data."""
    cid = raw.get("condition_id")
    if not cid:
        return None

    gamma_data = gamma.get(cid)
    if not gamma_data:
        return None

    tokens = raw.get("tokens", [])
    if len(tokens) < 2:
        return None

    yes_id, no_id = split_yes_no(tokens)
    rewards = raw.get("rewards") or {}

    try:
        if Decimal(str(rewards.get("max_spread") or 0)) <= 0:
            return None
        if Decimal(str(rewards.get("min_size") or 0)) <= 0:
            return None
    except (ValueError, ArithmeticError):
        return None

    try:
        return Market(
            condition_id=cid,
            slug=gamma_data.get("slug", ""),
            question=gamma_data.get("question") or raw.get("question", ""),
            yes_token_id=yes_id,
            no_token_id=no_id,
            rewards_max_spread_cents=Decimal(str(rewards["max_spread"])),
            rewards_min_size=Decimal(str(rewards["min_size"])),
            rewards_rate_per_day=extract_rate_per_day(rewards),
            tick_size=Decimal(str(raw.get("minimum_tick_size", "0.01"))),
            min_order_size=Decimal(str(raw.get("minimum_order_size", "1"))),
            end_date=parse_datetime(gamma_data.get("endDate")),
            created_at=parse_datetime(gamma_data.get("createdAt")),
            volume_24h=Decimal(str(gamma_data.get("volume24hr", 0))),
            liquidity=Decimal(str(gamma_data.get("liquidity", 0))),
            spread_cents=spreads.get(yes_id, Decimal("0")),
            price_change_24h=Decimal(str(gamma_data.get("oneDayPriceChange", 0))),
            game_start_time=parse_optional_datetime(gamma_data.get("gameStartTime")),
            event_slug=str(parent_event(gamma_data).get("slug") or ""),
            taker_fee_rate=extract_taker_fee_rate(gamma_data),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("skipping market %s — build failed: %s", cid, exc)
        return None


def parent_event(gamma_data: dict) -> dict:
    """The parent Gamma event a market belongs to (its event grouping),
    or {} when absent/malformed."""
    events = gamma_data.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        return events[0]
    return {}


def split_yes_no(tokens: list[dict]) -> tuple[str, str]:
    """Find Yes/No tokens by outcome label; fall back to array order."""
    yes = next(
        (t for t in tokens if (t.get("outcome") or "").lower() == "yes"),
        None,
    )
    no = next(
        (t for t in tokens if (t.get("outcome") or "").lower() == "no"),
        None,
    )
    if yes and no:
        return yes["token_id"], no["token_id"]
    return tokens[0]["token_id"], tokens[1]["token_id"]


def extract_rate_per_day(rewards: dict) -> Decimal:
    """Pull the first daily reward rate from rewards.rates (shape varies: list of
    {asset_address, rewards_daily_rate} entries or a dict); default 0 if missing."""
    rates = rewards.get("rates")
    if isinstance(rates, list) and rates and isinstance(rates[0], dict):
        return Decimal(str(rates[0].get("rewards_daily_rate", 0)))
    if isinstance(rates, dict):
        return Decimal(str(rates.get("rewards_daily_rate", 0)))
    return Decimal("0")


def extract_taker_fee_rate(gamma_data: dict) -> Decimal:
    """Taker fee rate from Gamma data; 0 if disabled. Prefers `feeSchedule.rate` over
    `takerBaseFee` bps, which can disagree with it (observed 1000bps vs 0.07)."""
    if not gamma_data.get("feesEnabled"):
        return Decimal(0)
    schedule = gamma_data.get("feeSchedule")
    if isinstance(schedule, dict) and schedule.get("rate") is not None:
        try:
            return Decimal(str(schedule["rate"]))
        except (ValueError, ArithmeticError):
            pass
    taker_bps = gamma_data.get("takerBaseFee")
    if taker_bps is not None:
        try:
            return Decimal(str(taker_bps)) / Decimal(10000)
        except (ValueError, ArithmeticError):
            pass
    return Decimal(0)


def parse_datetime(value: str | None) -> datetime:
    """Parse ISO 8601 timestamp from API. Handles trailing 'Z'. Defaults to epoch if missing."""
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_optional_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return dateutil_parser.parse(value)
    except (ValueError, TypeError):
        return None
