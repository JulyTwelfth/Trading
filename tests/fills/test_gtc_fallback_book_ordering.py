"""The CLOB /book REST endpoint returns bids ascending and asks descending — the
best level is LAST, not first. These tests pin the GTC exit fallback (and
get_best_bid_ask) to the best price regardless of how the API orders levels."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.bot import market as market_mod
from app.bot.market import get_best_bid_ask
from app.bot.schemas import BookLevel, OrderBook
from app.farm import exits as exits_mod
from app.farm.exits import exit_position_leg
from app.farm.schemas import FarmState


def make_book(bids: list[str], asks: list[str]) -> OrderBook:
    return OrderBook(
        market="market-A",
        asset_id="tok-yes",
        timestamp="2026-06-11T17:30:00Z",
        bids=[BookLevel(price=Decimal(p), size=Decimal("10")) for p in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal("10")) for p in asks],
        min_order_size=Decimal("5"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="x",
    )


@pytest.fixture
def gtc_fallback(monkeypatch):
    """Force the FAK to fail and capture the GTC fallback placement."""
    limit_calls: list = []

    async def failing_market_order(client, token_id, side, amount):
        raise RuntimeError("no match")

    async def fake_place_limit_order(client, order, post_only=False):
        limit_calls.append(order)
        return f"gtc-{len(limit_calls)}"

    monkeypatch.setattr(exits_mod, "place_market_order", failing_market_order)
    monkeypatch.setattr(exits_mod, "place_limit_order", fake_place_limit_order)
    return limit_calls


def stub_book(monkeypatch, module, book: OrderBook) -> None:
    async def fake_get_order_book(token_id: str) -> OrderBook:
        return book

    monkeypatch.setattr(module, "get_order_book", fake_get_order_book)


async def run_exit(farm_state: FarmState) -> None:
    await exit_position_leg(
        MagicMock(),
        farm_state,
        "tok-yes",
        Decimal("20"),
        "market-A",
        "m1",
        "YES",
        cancel_resting=False,
    )


# ── GTC fallback price selection ────────────────────────────────────────────


async def test_fallback_picks_best_bid_from_ascending_book(
    farm_state: FarmState, gtc_fallback, monkeypatch
):
    # Real CLOB ordering: penny-ladder first, best bid last.
    stub_book(monkeypatch, exits_mod, make_book(["0.01", "0.02", "0.10", "0.27"], ["0.99", "0.34"]))

    await run_exit(farm_state)

    assert len(gtc_fallback) == 1
    assert gtc_fallback[0].price == 0.27


async def test_fallback_picks_best_bid_from_descending_book(
    farm_state: FarmState, gtc_fallback, monkeypatch
):
    # Defensive: if the API ever returns best-first, selection must not regress.
    stub_book(monkeypatch, exits_mod, make_book(["0.27", "0.10", "0.02", "0.01"], ["0.34", "0.99"]))

    await run_exit(farm_state)

    assert len(gtc_fallback) == 1
    assert gtc_fallback[0].price == 0.27


async def test_fallback_picks_best_bid_from_unsorted_book(
    farm_state: FarmState, gtc_fallback, monkeypatch
):
    stub_book(monkeypatch, exits_mod, make_book(["0.10", "0.27", "0.01"], ["0.34"]))

    await run_exit(farm_state)

    assert len(gtc_fallback) == 1
    assert gtc_fallback[0].price == 0.27


async def test_fallback_single_bid_level(farm_state: FarmState, gtc_fallback, monkeypatch):
    stub_book(monkeypatch, exits_mod, make_book(["0.42"], ["0.55"]))

    await run_exit(farm_state)

    assert len(gtc_fallback) == 1
    assert gtc_fallback[0].price == 0.42


async def test_fallback_tied_bid_levels(farm_state: FarmState, gtc_fallback, monkeypatch):
    stub_book(monkeypatch, exits_mod, make_book(["0.30", "0.30"], ["0.55"]))

    await run_exit(farm_state)

    assert len(gtc_fallback) == 1
    assert gtc_fallback[0].price == 0.30


async def test_fallback_abandons_on_empty_bids(farm_state: FarmState, gtc_fallback, monkeypatch):
    stub_book(monkeypatch, exits_mod, make_book([], ["0.55"]))

    await run_exit(farm_state)

    assert gtc_fallback == []


# ── get_best_bid_ask ────────────────────────────────────────────────────────


async def test_best_bid_ask_real_clob_ordering(monkeypatch):
    # Bids ascending, asks descending — best of each is the LAST element.
    stub_book(
        monkeypatch, market_mod, make_book(["0.01", "0.02", "0.31"], ["0.99", "0.96", "0.34"])
    )

    best_bid, best_ask = await get_best_bid_ask("tok-yes")

    assert best_bid == Decimal("0.31")
    assert best_ask == Decimal("0.34")


async def test_best_bid_ask_unsorted(monkeypatch):
    stub_book(monkeypatch, market_mod, make_book(["0.31", "0.02"], ["0.34", "0.99"]))

    best_bid, best_ask = await get_best_bid_ask("tok-yes")

    assert best_bid == Decimal("0.31")
    assert best_ask == Decimal("0.34")


async def test_best_bid_ask_empty_book_defaults(monkeypatch):
    stub_book(monkeypatch, market_mod, make_book([], []))

    best_bid, best_ask = await get_best_bid_ask("tok-yes")

    assert best_bid == Decimal("0")
    assert best_ask == Decimal("1")
