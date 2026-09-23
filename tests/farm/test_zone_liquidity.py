"""Zone-liquidity math (book_zone_liquidity / zone_liquidity_usd) and the batched
fetch_books fetcher. Zone liquidity = USD resting inside rewards_max_spread of the
midpoint, summed across both outcome books — the crowding measure the farm filters on."""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.bot.schemas import BookLevel, OrderBook
from app.farm import discovery as discovery_mod
from app.farm.discovery import fetch_books
from app.farm.schemas import Market
from app.farm.zone_liquidity import (
    book_bid_depth,
    book_zone_liquidity,
    zone_distribution,
    zone_liquidity_usd,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _book(bids: list[tuple[str, str]], asks: list[tuple[str, str]], asset_id="tok") -> OrderBook:
    return OrderBook(
        market="m",
        asset_id=asset_id,
        timestamp=NOW,
        bids=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in bids],
        asks=[BookLevel(price=Decimal(p), size=Decimal(s)) for p, s in asks],
        min_order_size=Decimal("1"),
        tick_size=Decimal("0.01"),
        neg_risk=False,
        hash="h",
    )


def _market(**overrides) -> Market:
    base = dict(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="y",
        no_token_id="n",
        rewards_max_spread_cents=Decimal("3"),
        rewards_min_size=Decimal("100"),
        rewards_rate_per_day=Decimal("5"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        end_date=datetime(2030, 1, 1, tzinfo=timezone.utc),
        created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
        volume_24h=Decimal("100"),
        liquidity=Decimal("100"),
        spread_cents=Decimal("1"),
        price_change_24h=Decimal("0"),
    )
    base.update(overrides)
    return Market(**base)


# ── book_zone_liquidity ──────────────────────────────────────────────────────


def test_counts_both_sides_within_band():
    # max_spread_price 0.03 around mid 0.50: in-band = [0.47, 0.53].
    book = _book(bids=[("0.49", "100")], asks=[("0.51", "100")])
    # 0.49*100 + 0.51*100 = 49 + 51 = 100
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("100")


def test_excludes_levels_outside_band():
    book = _book(bids=[("0.49", "100"), ("0.40", "999")], asks=[("0.51", "100"), ("0.60", "999")])
    # Only 0.49 and 0.51 are within 0.03 of 0.50; 0.40 and 0.60 are out.
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("100")


def test_boundary_is_inclusive():
    # |0.47 - 0.50| == 0.03 exactly → counted (<=, not <).
    book = _book(bids=[("0.47", "200")], asks=[])
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("94")


def test_empty_book_is_zero():
    assert book_zone_liquidity(_book([], []), Decimal("0.50"), Decimal("0.03")) == Decimal("0")


def test_order_exactly_at_midpoint_counts():
    # spread s = 0 → in-band.
    book = _book(bids=[("0.50", "100")], asks=[])
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("50")


def test_zero_size_level_contributes_nothing():
    book = _book(bids=[("0.49", "0")], asks=[])
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("0")


def test_crossed_levels_counted_by_distance_not_side():
    # An ask priced below the midpoint (crossed/odd book) is still in-band by |price-mid|.
    book = _book(bids=[], asks=[("0.49", "100")])
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("49")


def test_fractional_sizes_keep_decimal_precision():
    book = _book(bids=[("0.49", "12.5")], asks=[])
    assert book_zone_liquidity(book, Decimal("0.50"), Decimal("0.03")) == Decimal("6.125")


def test_extreme_low_midpoint_does_not_break_band():
    # mid 0.02, band ±0.03 → [−0.01, 0.05]. 0.01 and 0.04 in, 0.06 out.
    book = _book(bids=[("0.01", "100")], asks=[("0.04", "100"), ("0.06", "100")])
    # 0.01*100 + 0.04*100 = 1 + 4 = 5
    assert book_zone_liquidity(book, Decimal("0.02"), Decimal("0.03")) == Decimal("5")


# ── zone_liquidity_usd ───────────────────────────────────────────────────────


def test_sums_both_outcome_books_using_market_max_spread():
    market = _market(rewards_max_spread_cents=Decimal("3"))
    yes_book = _book(
        bids=[("0.49", "100"), ("0.47", "200")], asks=[("0.51", "100"), ("0.55", "50")]
    )
    no_book = _book(bids=[("0.48", "100")], asks=[("0.52", "100")])
    # yes: 0.49*100 + 0.47*200 + 0.51*100 = 49 + 94 + 51 = 194  (0.55 is out of band)
    # no:  0.48*100 + 0.52*100 = 48 + 52 = 100
    result = zone_liquidity_usd(market, yes_book, Decimal("0.50"), no_book, Decimal("0.50"))
    assert result == Decimal("294")


def test_negrisk_books_use_their_own_midpoints():
    # negRisk: YES and NO midpoints both 0.60 (sum 1.20, not 1.0). Each book bands on its own.
    market = _market(rewards_max_spread_cents=Decimal("3"))
    yes_book = _book(bids=[("0.59", "100")], asks=[])
    no_book = _book(bids=[("0.58", "100")], asks=[])
    # yes: 0.59*100 = 59 ; no: 0.58*100 = 58
    result = zone_liquidity_usd(market, yes_book, Decimal("0.60"), no_book, Decimal("0.60"))
    assert result == Decimal("117")


def test_fractional_max_spread_cents():
    # 2.5 cent max spread → band ±0.025. 0.48 in (0.02), 0.47 out (0.03).
    market = _market(rewards_max_spread_cents=Decimal("2.5"))
    yes_book = _book(bids=[("0.48", "100"), ("0.47", "100")], asks=[])
    no_book = _book(bids=[], asks=[])
    assert zone_liquidity_usd(market, yes_book, Decimal("0.50"), no_book, Decimal("0.50")) == (
        Decimal("48")
    )


def test_wider_max_spread_includes_more_depth():
    # Same books, but a 10-cent max spread now pulls 0.55 and everything else into the band.
    market = _market(rewards_max_spread_cents=Decimal("10"))
    yes_book = _book(
        bids=[("0.49", "100"), ("0.47", "200")], asks=[("0.51", "100"), ("0.55", "50")]
    )
    no_book = _book(bids=[], asks=[])
    # yes: 49 + 94 + 51 + 0.55*50(=27.5) = 221.5
    result = zone_liquidity_usd(market, yes_book, Decimal("0.50"), no_book, Decimal("0.50"))
    assert result == Decimal("221.5")


# ── invariants ───────────────────────────────────────────────────────────────


def test_total_equals_sum_of_both_books():
    # zone_liquidity_usd is exactly the two per-book figures added.
    market = _market(rewards_max_spread_cents=Decimal("3"))
    yb = _book(bids=[("0.49", "100")], asks=[("0.51", "30")])
    nb = _book(bids=[("0.48", "70")], asks=[])
    v = Decimal("0.03")
    expected = book_zone_liquidity(yb, Decimal("0.50"), v) + book_zone_liquidity(
        nb, Decimal("0.50"), v
    )
    assert zone_liquidity_usd(market, yb, Decimal("0.50"), nb, Decimal("0.50")) == expected


def test_adding_in_band_depth_strictly_increases():
    base = _book(bids=[("0.49", "100")], asks=[])
    more = _book(bids=[("0.49", "100"), ("0.48", "100")], asks=[])
    mid, v = Decimal("0.50"), Decimal("0.03")
    assert book_zone_liquidity(more, mid, v) > book_zone_liquidity(base, mid, v)


def test_adding_out_of_band_depth_does_not_change_total():
    base = _book(bids=[("0.49", "100")], asks=[])
    plus_far = _book(bids=[("0.49", "100"), ("0.30", "9999")], asks=[("0.80", "9999")])
    mid, v = Decimal("0.50"), Decimal("0.03")
    assert book_zone_liquidity(plus_far, mid, v) == book_zone_liquidity(base, mid, v)


def test_fuzz_book_zone_liquidity_invariants():
    # Randomised books: result must always be non-negative and exactly equal an independent
    # in-band recomputation; out-of-band orders must never contribute. Seeded for determinism.
    import random

    rng = random.Random(1234)

    def d(value: float) -> Decimal:
        return Decimal(str(round(value, 2)))

    for _ in range(500):
        mid = d(rng.uniform(0.02, 0.98))
        v = d(rng.uniform(0.01, 0.10))
        levels = [
            (d(rng.uniform(0.0, 1.0)), d(rng.uniform(0, 500))) for _ in range(rng.randint(0, 12))
        ]
        half = len(levels) // 2
        book = _book(
            bids=[(str(p), str(s)) for p, s in levels[:half]],
            asks=[(str(p), str(s)) for p, s in levels[half:]],
        )
        result = book_zone_liquidity(book, mid, v)
        expected = sum((p * s for p, s in levels if abs(p - mid) <= v), Decimal(0))
        assert result == expected
        assert result >= 0


# ── zone_distribution ────────────────────────────────────────────────────────


def test_distribution_returns_low_median_high():
    markets = [
        _market(zone_liquidity=Decimal("100")),
        _market(zone_liquidity=Decimal("540")),
        _market(zone_liquidity=Decimal("9800")),
    ]
    assert zone_distribution(markets) == (Decimal("100"), Decimal("540"), Decimal("9800"))


def test_distribution_ignores_uncomputed_markets():
    markets = [_market(zone_liquidity=Decimal("200")), _market(zone_liquidity=None)]
    assert zone_distribution(markets) == (Decimal("200"), Decimal("200"), Decimal("200"))


def test_distribution_none_when_nothing_computed():
    assert zone_distribution([_market(zone_liquidity=None)]) is None
    assert zone_distribution([]) is None


def test_distribution_single_market():
    assert zone_distribution([_market(zone_liquidity=Decimal("42"))]) == (
        Decimal("42"),
        Decimal("42"),
        Decimal("42"),
    )


def test_distribution_even_count_typical_is_upper_middle():
    # Even count → median takes values[len//2] (upper-middle); low/high are exact bounds.
    markets = [_market(zone_liquidity=Decimal(v)) for v in ("100", "200", "300", "400")]
    assert zone_distribution(markets) == (Decimal("100"), Decimal("300"), Decimal("400"))


def test_distribution_handles_duplicates_and_unsorted_input():
    markets = [_market(zone_liquidity=Decimal(v)) for v in ("500", "100", "500")]
    assert zone_distribution(markets) == (Decimal("100"), Decimal("500"), Decimal("500"))


# ── fetch_books ──────────────────────────────────────────────────────────────


def books_payload(asset_id: str) -> dict:
    # Mirrors the REAL POST /books response shape (captured live): epoch-MILLISECONDS string
    # timestamp, plus the extra last_trade_price the OrderBook model drops.
    return {
        "market": "0xcond",
        "asset_id": asset_id,
        "timestamp": "1781489243814",
        "hash": "abc",
        "bids": [{"price": "0.49", "size": "100"}],
        "asks": [{"price": "0.51", "size": "100"}],
        "min_order_size": "1",
        "tick_size": "0.01",
        "neg_risk": False,
        "last_trade_price": "0.50",
    }


async def test_fetch_books_parses_real_epoch_ms_timestamp():
    # Regression: real /books sends timestamp as epoch-ms ("1781489243814"), NOT ISO. The
    # OrderBook model must read it as milliseconds (→ year 2026), not seconds (→ year ~58000).
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=[books_payload("tokA")])
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    result = await fetch_books(http, ["tokA"])

    assert result["tokA"].timestamp.year == 2026


async def test_fetch_books_keys_by_asset_id():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=[books_payload("tokA"), books_payload("tokB")])
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    result = await fetch_books(http, ["tokA", "tokB"])

    assert set(result) == {"tokA", "tokB"}
    assert result["tokA"].bids[0].price == Decimal("0.49")
    assert http.post.await_count == 1


async def test_fetch_books_batches_over_limit(monkeypatch):
    monkeypatch.setattr(discovery_mod, "SPREADS_BATCH_LIMIT", 2)
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=[books_payload("x")])
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    await fetch_books(http, ["a", "b", "c"])  # 3 tokens, batch size 2 → 2 POSTs

    assert http.post.await_count == 2


async def test_fetch_books_exact_batch_boundaries(monkeypatch):
    monkeypatch.setattr(discovery_mod, "SPREADS_BATCH_LIMIT", 2)
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=[])
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    await fetch_books(http, ["a", "b"])  # exactly one batch
    assert http.post.await_count == 1

    http.post.reset_mock()
    await fetch_books(http, ["a", "b", "c", "d"])  # exactly two batches
    assert http.post.await_count == 2


async def test_fetch_books_empty_skips_http():
    http = MagicMock()
    http.post = AsyncMock()

    assert await fetch_books(http, []) == {}
    assert http.post.await_count == 0


async def test_fetch_books_returns_subset_when_response_partial():
    # Requested two tokens but the API only returns one book → caller gets a partial dict
    # and treats the missing one as zone_unknown.
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=[books_payload("tokA")])
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    result = await fetch_books(http, ["tokA", "tokB"])

    assert set(result) == {"tokA"}


async def test_fetch_books_parses_empty_levels():
    payload = books_payload("tokA")
    payload["bids"] = []
    payload["asks"] = []
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=[payload])
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    result = await fetch_books(http, ["tokA"])

    assert result["tokA"].bids == []
    assert result["tokA"].asks == []


async def test_fetch_books_propagates_http_error():
    request = httpx.Request("POST", "https://clob.polymarket.com/books")
    response = httpx.Response(500, request=request)
    resp = MagicMock()
    resp.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError("500", request=request, response=response)
    )
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)

    with pytest.raises(httpx.HTTPStatusError):
        await fetch_books(http, ["tokA"])


def test_book_bid_depth_counts_only_in_band_bids():
    # mid 0.50, band 0.04: bids at 0.49 (in) + 0.47 (in) count; 0.40 (out) and all asks excluded.
    book = _book(
        bids=[("0.49", "100"), ("0.47", "50"), ("0.40", "999")],
        asks=[("0.51", "888")],
    )
    assert book_bid_depth(book, Decimal("0.50"), Decimal("0.04")) == Decimal("150")


def test_book_bid_depth_zero_on_one_sided_book():
    # an empty bid side (all depth on asks) is the strand case -> 0 shares to sell into.
    book = _book(bids=[], asks=[("0.06", "10000")])
    assert book_bid_depth(book, Decimal("0.03"), Decimal("0.04")) == Decimal("0")
