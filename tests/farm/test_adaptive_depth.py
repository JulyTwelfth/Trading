from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from app.bot.schemas import BestBidAsk, BookLevel, OrderBook
from app.farm import requote as requote_mod
from app.farm import worker as worker_mod
from app.farm.quoting import compute_quote
from app.farm.requote import handle_bba, leg_quote_depth
from app.farm.schemas import FarmState, LiveBook, Market, MarketPosition
from app.farm.worker import annotate_live_metrics, open_position

MID = Decimal("0.50")


def book(price: str, size: str, asset_id: str = "t") -> OrderBook:
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=datetime.now(timezone.utc),
        bids=[BookLevel(price=Decimal(price), size=Decimal(size))],
        asks=[],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="h",
    )


async def entry_prices(farm_state: FarmState, market: Market, monkeypatch) -> dict[str, float]:
    farm_state.positions.clear()
    placed: dict[str, float] = {}

    async def fake_place(client, order, post_only=False):
        placed[order.token_id] = order.price
        return "yes-oid" if order.token_id == market.yes_token_id else "no-oid"

    monkeypatch.setattr(worker_mod, "place_limit_order", fake_place)
    monkeypatch.setattr(worker_mod, "cancel_order", AsyncMock())
    midpoints = {market.yes_token_id: MID, market.no_token_id: MID}
    await open_position(MagicMock(), farm_state, AsyncMock(), market, midpoints)
    return placed


async def test_entry_quotes_safe_when_sole(farm_state: FarmState, market: Market, monkeypatch):
    market.effective_depth = "aggressive"
    market.adaptive_depth = "safe"
    placed = await entry_prices(farm_state, market, monkeypatch)

    safe_px = float(compute_quote(market, MID, "safe")[0])
    aggr_px = float(compute_quote(market, MID, "aggressive")[0])
    assert safe_px != aggr_px, "fixture sanity: safe and aggressive prices must differ"
    assert placed[market.yes_token_id] == safe_px
    assert placed[market.no_token_id] == safe_px


async def test_entry_unchanged_when_adaptive_none(
    farm_state: FarmState, market: Market, monkeypatch
):
    farm_state.config.quote_depth = "safe"
    market.effective_depth = "aggressive"
    market.adaptive_depth = None
    placed = await entry_prices(farm_state, market, monkeypatch)

    assert placed[market.yes_token_id] == float(compute_quote(market, MID, "aggressive")[0])


async def test_annotate_sets_safe_when_sole(market: Market, monkeypatch):
    async def fake_mid(http, toks):
        return {market.yes_token_id: MID, market.no_token_id: MID}

    async def fake_books(http, toks):
        return {market.yes_token_id: book("0.49", "5"), market.no_token_id: book("0.49", "5")}

    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_mid)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)

    await annotate_live_metrics(MagicMock(), [market], need_books=True)
    assert market.adaptive_depth == "safe"


async def test_annotate_no_safe_when_competitor(market: Market, monkeypatch):
    async def fake_mid(http, toks):
        return {market.yes_token_id: MID, market.no_token_id: MID}

    async def fake_books(http, toks):
        return {market.yes_token_id: book("0.49", "100"), market.no_token_id: book("0.49", "5")}

    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_mid)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)

    await annotate_live_metrics(MagicMock(), [market], need_books=True)
    assert market.adaptive_depth is None


def pos(market: Market) -> MarketPosition:
    return MarketPosition(
        market=market,
        yes_order_id="y",
        no_order_id="n",
        yes_price=Decimal("0.48"),
        no_price=Decimal("0.48"),
    )


def test_leg_depth_honors_entry_safe_decision(farm_state: FarmState, market: Market):
    farm_state.config.quote_depth = "aggressive"
    market.effective_depth = "aggressive"
    market.adaptive_depth = "safe"
    assert leg_quote_depth(farm_state, pos(market)) == "safe"


def test_leg_depth_falls_back_to_tier_when_not_safe(farm_state: FarmState, market: Market):
    farm_state.config.quote_depth = "aggressive"
    market.adaptive_depth = None
    assert leg_quote_depth(farm_state, pos(market)) == "aggressive"


def test_leg_depth_ignores_live_book(farm_state: FarmState, market: Market):
    farm_state.config.quote_depth = "aggressive"
    market.adaptive_depth = "safe"
    farm_state.live_books[market.yes_token_id] = LiveBook(bids={Decimal("0.49"): Decimal(99999)})
    assert leg_quote_depth(farm_state, pos(market)) == "safe"


def test_leg_depth_tier_from_effective_not_global(farm_state: FarmState, market: Market):
    farm_state.config.quote_depth = "safe"
    market.effective_depth = "normal"
    market.adaptive_depth = None
    assert leg_quote_depth(farm_state, pos(market)) == "normal"


def test_leg_depth_matches_open_position_formula(farm_state: FarmState, market: Market):
    farm_state.config.quote_depth = "normal"
    for adaptive, effective, expected in (
        ("safe", "aggressive", "safe"),
        (None, "aggressive", "aggressive"),
        (None, None, "normal"),
    ):
        market.adaptive_depth = adaptive
        market.effective_depth = effective
        entry_formula = (
            market.adaptive_depth or market.effective_depth or farm_state.config.quote_depth
        )
        assert leg_quote_depth(farm_state, pos(market)) == expected == entry_formula


async def test_annotate_no_adaptive_when_one_book_missing(market: Market, monkeypatch):
    async def fake_mid(http, toks):
        return {market.yes_token_id: MID, market.no_token_id: MID}

    async def fake_books(http, toks):
        return {market.yes_token_id: book("0.49", "5")}

    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_mid)
    monkeypatch.setattr(worker_mod, "fetch_books", fake_books)

    await annotate_live_metrics(MagicMock(), [market], need_books=True)
    assert market.adaptive_depth is None


async def test_annotate_skips_adaptive_when_books_not_fetched(market: Market, monkeypatch):
    async def fake_mid(http, toks):
        return {market.yes_token_id: MID, market.no_token_id: MID}

    monkeypatch.setattr(worker_mod, "fetch_midpoints", fake_mid)
    await annotate_live_metrics(MagicMock(), [market], need_books=False)
    assert market.adaptive_depth is None


async def test_handle_bba_requotes_at_entry_depth_ignoring_book(farm_state: FarmState, monkeypatch):
    pos = farm_state.positions["market-A"]
    pos.yes_shares = Decimal("0")
    pos.yes_price = Decimal("0.50")
    pos.market.effective_depth = "aggressive"
    pos.market.adaptive_depth = "safe"
    farm_state.live_books[pos.market.yes_token_id] = LiveBook(
        bids={Decimal("0.49"): Decimal(99999)}
    )

    captured: list = []

    async def fake_requote_leg(client, state, ws, p, outcome, price):
        captured.append((outcome, float(price)))

    monkeypatch.setattr(requote_mod, "requote_leg", fake_requote_leg)
    monkeypatch.setattr(requote_mod, "cancel_order", AsyncMock())

    frame = BestBidAsk(
        event_type="best_bid_ask",
        market="market-A",
        asset_id=pos.market.yes_token_id,
        best_bid=Decimal("0.50"),
        best_ask=Decimal("0.52"),
        spread=Decimal("0.02"),
        timestamp="2026-06-17T00:00:00Z",
    )
    midpoint = (frame.best_bid + frame.best_ask) / 2
    safe_px = compute_quote(pos.market, midpoint, "safe")[0]
    assert safe_px < pos.yes_price, (
        "precondition: safe edge must sit below our price to re-quote down"
    )
    expected = float(min(safe_px, frame.best_ask - pos.market.tick_size))

    await handle_bba(MagicMock(), farm_state, AsyncMock(), pos, frame)

    assert captured == [("YES", expected)], (
        "re-quoted at the ENTRY safe depth — NOT tightened by the live competitor"
    )
