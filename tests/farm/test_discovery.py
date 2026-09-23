from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.farm import discovery as discovery_mod
from app.farm.discovery import (
    build_market,
    extract_rate_per_day,
    fetch_eligible_markets,
    fetch_midpoints,
    fetch_sampling_simplified,
    fetch_spreads,
    parse_datetime,
    parse_optional_datetime,
    split_yes_no,
)
from app.farm.schemas import Market


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(discovery_mod.asyncio, "sleep", AsyncMock())


def ok_response(payload):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)
    return resp


def raw_market(condition_id="0xcid"):
    return {
        "condition_id": condition_id,
        "tokens": [
            {"token_id": "tok-yes", "outcome": "Yes"},
            {"token_id": "tok-no", "outcome": "No"},
        ],
        "rewards": {
            "max_spread": "3",
            "min_size": "100",
            "rates": [{"rewards_daily_rate": "5"}],
        },
        "minimum_tick_size": "0.01",
        "minimum_order_size": "5",
        "question": "fallback question",
    }


def gamma_for(condition_id="0xcid"):
    return {
        condition_id: {
            "slug": "the-slug",
            "question": "Gamma question?",
            "endDate": "2030-01-01T00:00:00Z",
            "createdAt": "2025-01-01T00:00:00Z",
            "volume24hr": 1234,
            "liquidity": 5678,
            "oneDayPriceChange": "0.02",
            "gameStartTime": "2030-06-01T12:00:00Z",
        }
    }


# ── build_market happy path ─────────────────────────────────────────────────
def test_build_market_happy_path():
    spreads = {"tok-yes": Decimal("4")}
    market = build_market(raw_market(), gamma_for(), spreads)

    assert isinstance(market, Market)
    assert market.condition_id == "0xcid"
    assert market.slug == "the-slug"
    assert market.question == "Gamma question?"
    assert market.yes_token_id == "tok-yes"
    assert market.no_token_id == "tok-no"
    assert market.rewards_max_spread_cents == Decimal("3")
    assert market.rewards_min_size == Decimal("100")
    assert market.rewards_rate_per_day == Decimal("5")
    assert market.tick_size == Decimal("0.01")
    assert market.min_order_size == Decimal("5")
    assert market.volume_24h == Decimal("1234")
    assert market.liquidity == Decimal("5678")
    assert market.spread_cents == Decimal("4")
    assert market.price_change_24h == Decimal("0.02")
    assert market.end_date == datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert market.game_start_time is not None


def test_build_market_question_falls_back_to_raw():
    gamma = gamma_for()
    del gamma["0xcid"]["question"]
    market = build_market(raw_market(), gamma, {})
    assert market.question == "fallback question"


# ── build_market None guards ────────────────────────────────────────────────
def test_build_market_missing_condition_id():
    raw = raw_market()
    del raw["condition_id"]
    assert build_market(raw, gamma_for(), {}) is None


def test_build_market_missing_gamma():
    assert build_market(raw_market(), {}, {}) is None


def test_build_market_fewer_than_two_tokens():
    raw = raw_market()
    raw["tokens"] = [{"token_id": "only", "outcome": "Yes"}]
    assert build_market(raw, gamma_for(), {}) is None


def test_build_market_max_spread_non_positive():
    raw = raw_market()
    raw["rewards"]["max_spread"] = "0"
    assert build_market(raw, gamma_for(), {}) is None


def test_build_market_min_size_non_positive():
    raw = raw_market()
    raw["rewards"]["min_size"] = "0"
    assert build_market(raw, gamma_for(), {}) is None


def test_build_market_numeric_parse_except():
    # Non-numeric max_spread -> Decimal raises -> caught -> None.
    raw = raw_market()
    raw["rewards"]["max_spread"] = "not-a-number"
    assert build_market(raw, gamma_for(), {}) is None


def test_build_market_outer_except_returns_none():
    # max_spread/min_size pass the > 0 gate, but Market construction fails because
    # a required positive field (tick_size) is non-positive -> outer except -> None.
    raw = raw_market()
    raw["minimum_tick_size"] = "0"  # PositiveDecimal -> ValidationError
    assert build_market(raw, gamma_for(), {}) is None


# ── split_yes_no ────────────────────────────────────────────────────────────
def test_split_yes_no_by_label():
    tokens = [
        {"token_id": "n", "outcome": "No"},
        {"token_id": "y", "outcome": "Yes"},
    ]
    assert split_yes_no(tokens) == ("y", "n")


def test_split_yes_no_fallback_array_order():
    tokens = [
        {"token_id": "first", "outcome": "Maybe"},
        {"token_id": "second", "outcome": "Perhaps"},
    ]
    assert split_yes_no(tokens) == ("first", "second")


# ── extract_rate_per_day ────────────────────────────────────────────────────
def test_extract_rate_list_of_dicts():
    assert extract_rate_per_day({"rates": [{"rewards_daily_rate": "12.5"}]}) == Decimal("12.5")


def test_extract_rate_dict():
    assert extract_rate_per_day({"rates": {"rewards_daily_rate": "7"}}) == Decimal("7")


def test_extract_rate_none_or_other():
    assert extract_rate_per_day({}) == Decimal("0")
    assert extract_rate_per_day({"rates": None}) == Decimal("0")
    assert extract_rate_per_day({"rates": "weird"}) == Decimal("0")
    assert extract_rate_per_day({"rates": []}) == Decimal("0")


# ── parse_datetime ──────────────────────────────────────────────────────────
def test_parse_datetime_none_is_epoch():
    assert parse_datetime(None) == datetime(1970, 1, 1, tzinfo=timezone.utc)


def test_parse_datetime_iso_with_z():
    assert parse_datetime("2025-03-04T05:06:07Z") == datetime(
        2025, 3, 4, 5, 6, 7, tzinfo=timezone.utc
    )


# ── parse_optional_datetime ─────────────────────────────────────────────────
def test_parse_optional_datetime_none():
    assert parse_optional_datetime(None) is None


def test_parse_optional_datetime_valid():
    result = parse_optional_datetime("2025-03-04T05:06:07Z")
    assert result is not None
    assert result.year == 2025 and result.month == 3 and result.day == 4


def test_parse_optional_datetime_unparseable():
    assert parse_optional_datetime("definitely not a date") is None


# ── fetch_sampling_simplified ───────────────────────────────────────────────
async def test_fetch_sampling_paginates_then_stops():
    http = MagicMock()
    http.get = AsyncMock(
        side_effect=[
            ok_response({"data": [{"condition_id": "a"}], "next_cursor": "CUR2"}),
            ok_response({"data": [{"condition_id": "b"}], "next_cursor": "LTE="}),
        ]
    )

    result = await fetch_sampling_simplified(http)

    assert result == [{"condition_id": "a"}, {"condition_id": "b"}]
    assert http.get.await_count == 2
    # First call has empty params, second carries the cursor.
    first_params = http.get.await_args_list[0].kwargs["params"]
    second_params = http.get.await_args_list[1].kwargs["params"]
    assert first_params == {}
    assert second_params == {"next_cursor": "CUR2"}


async def test_fetch_sampling_empty_breaks_immediately():
    http = MagicMock()
    http.get = AsyncMock(return_value=ok_response({"data": [], "next_cursor": ""}))

    result = await fetch_sampling_simplified(http)

    assert result == []
    assert http.get.await_count == 1


# ── fetch_spreads ───────────────────────────────────────────────────────────
async def test_fetch_spreads_converts_to_cents():
    http = MagicMock()
    http.post = AsyncMock(return_value=ok_response({"tok-1": "0.04", "tok-2": "0.10"}))

    result = await fetch_spreads(http, ["tok-1", "tok-2"])

    assert result == {"tok-1": Decimal("4.00"), "tok-2": Decimal("10.00")}
    assert http.post.await_count == 1


async def test_fetch_spreads_empty_skips_http():
    http = MagicMock()
    http.post = AsyncMock()

    result = await fetch_spreads(http, [])

    assert result == {}
    assert http.post.await_count == 0


# ── fetch_midpoints ─────────────────────────────────────────────────────────
async def test_fetch_midpoints_converts_decimal():
    http = MagicMock()
    http.post = AsyncMock(return_value=ok_response({"tok-1": "0.52"}))

    result = await fetch_midpoints(http, ["tok-1"])

    assert result == {"tok-1": Decimal("0.52")}
    assert http.post.await_count == 1


async def test_fetch_midpoints_empty_skips_http():
    http = MagicMock()
    http.post = AsyncMock()

    result = await fetch_midpoints(http, [])

    assert result == {}
    assert http.post.await_count == 0


# ── fetch_eligible_markets orchestrator ─────────────────────────────────────
async def test_fetch_eligible_markets_builds_list(monkeypatch):
    sampling_rows = [raw_market("0xcid")]

    monkeypatch.setattr(
        discovery_mod,
        "fetch_sampling_simplified",
        AsyncMock(return_value=sampling_rows),
    )
    monkeypatch.setattr(
        discovery_mod, "fetch_gamma_markets", AsyncMock(return_value=gamma_for("0xcid"))
    )
    monkeypatch.setattr(
        discovery_mod,
        "fetch_spreads",
        AsyncMock(return_value={"tok-yes": Decimal("4")}),
    )

    result = await fetch_eligible_markets(MagicMock())

    assert len(result) == 1
    assert result[0].condition_id == "0xcid"
    assert result[0].spread_cents == Decimal("4")


async def test_fetch_eligible_markets_empty_sampling_returns_early(monkeypatch):
    gamma_mock = AsyncMock()
    spreads_mock = AsyncMock()
    monkeypatch.setattr(discovery_mod, "fetch_sampling_simplified", AsyncMock(return_value=[]))
    monkeypatch.setattr(discovery_mod, "fetch_gamma_markets", gamma_mock)
    monkeypatch.setattr(discovery_mod, "fetch_spreads", spreads_mock)

    result = await fetch_eligible_markets(MagicMock())

    assert result == []
    # Early return -> enrichment never invoked.
    assert gamma_mock.await_count == 0
    assert spreads_mock.await_count == 0


async def test_fetch_eligible_markets_drops_unbuildable(monkeypatch):
    # A sampling row with no matching gamma data -> build_market returns None -> dropped.
    sampling_rows = [raw_market("0xcid"), raw_market("0xmissing")]

    monkeypatch.setattr(
        discovery_mod,
        "fetch_sampling_simplified",
        AsyncMock(return_value=sampling_rows),
    )
    monkeypatch.setattr(
        discovery_mod, "fetch_gamma_markets", AsyncMock(return_value=gamma_for("0xcid"))
    )
    monkeypatch.setattr(discovery_mod, "fetch_spreads", AsyncMock(return_value={}))

    result = await fetch_eligible_markets(MagicMock())

    assert [m.condition_id for m in result] == ["0xcid"]
