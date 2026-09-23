from decimal import Decimal

import httpx
import pytest

from app.bot import market as market_mod
from app.bot.market import (
    get_best_bid_ask,
    get_order_book,
    resolve_event_markets,
    resolve_market,
)


def make_book(bids, asks):
    return {
        "market": "0xmarket",
        "asset_id": "tok-1",
        "timestamp": "2025-01-01T00:00:00Z",
        "bids": bids,
        "asks": asks,
        "min_order_size": "5",
        "tick_size": "0.01",
        "neg_risk": False,
        "hash": "abc",
    }


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def install_fake_client(monkeypatch, payload, captured=None):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            if captured is not None:
                captured.append((url, params))
            return FakeResponse(payload)

    monkeypatch.setattr(market_mod.httpx, "AsyncClient", FakeClient)


async def test_resolve_market_parses_slug_and_tokens(monkeypatch):
    captured = []
    events = [
        {
            "markets": [
                {
                    "clobTokenIds": '["tok-yes", "tok-no"]',
                    "question": "Will it rain?",
                }
            ]
        }
    ]
    install_fake_client(monkeypatch, events, captured)

    yes, no, question = await resolve_market(
        "https://polymarket.com/event/some-event/will-it-rain/"
    )

    assert (yes, no, question) == ("tok-yes", "tok-no", "Will it rain?")
    # Slug parsed from the trailing path segment (trailing slash stripped).
    url, params = captured[0]
    assert url.endswith("/events")
    assert params == {"slug": "will-it-rain"}


async def test_resolve_market_no_events_raises(monkeypatch):
    install_fake_client(monkeypatch, [])
    with pytest.raises(ValueError, match="No event found for slug: will-it-rain"):
        await resolve_market("https://polymarket.com/event/will-it-rain")


async def test_get_order_book_validates(monkeypatch):
    payload = make_book(
        bids=[{"price": "0.40", "size": "10"}],
        asks=[{"price": "0.60", "size": "5"}],
    )
    install_fake_client(monkeypatch, payload)

    book = await get_order_book("tok-1")

    assert book.asset_id == "tok-1"
    assert book.bids[0].price == Decimal("0.40")
    assert book.asks[0].size == Decimal("5")


async def test_get_best_bid_ask_picks_extremes(monkeypatch):
    payload = make_book(
        bids=[
            {"price": "0.40", "size": "10"},
            {"price": "0.45", "size": "3"},
            {"price": "0.30", "size": "2"},
        ],
        asks=[
            {"price": "0.60", "size": "5"},
            {"price": "0.55", "size": "1"},
            {"price": "0.70", "size": "9"},
        ],
    )
    install_fake_client(monkeypatch, payload)

    best_bid, best_ask = await get_best_bid_ask("tok-1")

    assert best_bid == Decimal("0.45")  # max of bids
    assert best_ask == Decimal("0.55")  # min of asks


async def test_get_best_bid_ask_empty_book_defaults(monkeypatch):
    payload = make_book(bids=[], asks=[])
    install_fake_client(monkeypatch, payload)

    best_bid, best_ask = await get_best_bid_ask("tok-1")

    assert best_bid == Decimal("0")
    assert best_ask == Decimal("1")


# ── resolve_event_markets ─────────────────────────────────────────────────────


def install_fake_client_raising(monkeypatch, status_code=404):
    """A client whose response.raise_for_status() raises HTTPStatusError, like a Gamma non-200."""

    class RaisingResponse:
        def raise_for_status(self):
            request = httpx.Request("GET", "https://gamma.example/events")
            response = httpx.Response(status_code, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

        def json(self):  # pragma: no cover - never reached after raise_for_status
            return []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, params=None):
            return RaisingResponse()

    monkeypatch.setattr(market_mod.httpx, "AsyncClient", FakeClient)


async def test_resolve_event_markets_returns_one_dict_per_market(monkeypatch):
    captured = []
    events = [
        {
            "markets": [
                {"conditionId": "0xc1", "slug": "team-a-wins", "question": "Team A wins?"},
                {"conditionId": "0xc2", "slug": "team-b-wins", "question": "Team B wins?"},
            ]
        }
    ]
    install_fake_client(monkeypatch, events, captured)

    result = await resolve_event_markets("https://polymarket.com/event/world-cup-final/")

    assert result == [
        {"condition_id": "0xc1", "slug": "team-a-wins", "question": "Team A wins?"},
        {"condition_id": "0xc2", "slug": "team-b-wins", "question": "Team B wins?"},
    ]
    # Slug parsed from the trailing path segment (trailing slash stripped).
    url, params = captured[0]
    assert url.endswith("/events")
    assert params == {"slug": "world-cup-final"}


async def test_resolve_event_markets_skips_markets_without_condition_id(monkeypatch):
    events = [
        {
            "markets": [
                {"slug": "no-cid", "question": "no conditionId here"},
                {"conditionId": "0xc2", "slug": "has-cid", "question": "kept?"},
            ]
        }
    ]
    install_fake_client(monkeypatch, events)

    result = await resolve_event_markets("https://polymarket.com/event/some-event")

    assert result == [{"condition_id": "0xc2", "slug": "has-cid", "question": "kept?"}]


async def test_resolve_event_markets_no_events_raises(monkeypatch):
    install_fake_client(monkeypatch, [])
    with pytest.raises(ValueError, match="No event found for slug: ghost"):
        await resolve_event_markets("https://polymarket.com/event/ghost")


async def test_resolve_event_markets_no_market_with_condition_id_raises(monkeypatch):
    events = [{"markets": [{"slug": "m", "question": "?"}, {"question": "no cid"}]}]
    install_fake_client(monkeypatch, events)
    with pytest.raises(ValueError, match="no markets with a conditionId: dead-event"):
        await resolve_event_markets("https://polymarket.com/event/dead-event")


async def test_resolve_event_markets_empty_markets_list_raises(monkeypatch):
    install_fake_client(monkeypatch, [{"markets": []}])
    with pytest.raises(ValueError, match="no markets with a conditionId: empty-event"):
        await resolve_event_markets("https://polymarket.com/event/empty-event")


async def test_resolve_event_markets_missing_markets_key_raises(monkeypatch):
    # events[0] has no "markets" key at all → .get("markets") or [] → empty → ValueError.
    install_fake_client(monkeypatch, [{}])
    with pytest.raises(ValueError, match="no markets with a conditionId: keyless"):
        await resolve_event_markets("https://polymarket.com/event/keyless")


async def test_resolve_event_markets_propagates_http_status_error(monkeypatch):
    install_fake_client_raising(monkeypatch, status_code=502)
    with pytest.raises(httpx.HTTPStatusError):
        await resolve_event_markets("https://polymarket.com/event/whatever")
