from datetime import datetime, timedelta, timezone

from app.farm.filters import effective, passes_created_date, passes_time_remaining
from app.farm.schemas import Market

NOW = datetime(2026, 5, 24, 12, 0, tzinfo=timezone.utc)


def market_ending_in(market: Market, delta: timedelta) -> Market:
    return market.model_copy(update={"end_date": NOW + delta})


def market_created_ago(market: Market, delta: timedelta) -> Market:
    return market.model_copy(update={"created_at": NOW - delta})


def test_time_remaining_excludes_market_resolving_within_window(market: Market):
    m = market_ending_in(market, timedelta(hours=12))
    assert passes_time_remaining(m, "1d", NOW) is False


def test_time_remaining_includes_market_resolving_beyond_window(market: Market):
    m = market_ending_in(market, timedelta(hours=36))
    assert passes_time_remaining(m, "1d", NOW) is True


def test_time_remaining_all_window_passes_everything(market: Market):
    m = market_ending_in(market, timedelta(minutes=5))
    assert passes_time_remaining(m, "all", NOW) is True


def test_time_remaining_12h_excludes_market_within_window(market: Market):
    m = market_ending_in(market, timedelta(hours=6))
    assert passes_time_remaining(m, "12h", NOW) is False


def test_time_remaining_12h_includes_market_beyond_window(market: Market):
    m = market_ending_in(market, timedelta(hours=18))
    assert passes_time_remaining(m, "12h", NOW) is True


def test_time_remaining_12h_boundary_is_strict(market: Market):
    m = market_ending_in(market, timedelta(hours=12))
    assert passes_time_remaining(m, "12h", NOW) is False


def test_time_remaining_7d_excludes_market_within_window(market: Market):
    m = market_ending_in(market, timedelta(days=5))
    assert passes_time_remaining(m, "7d", NOW) is False


def test_time_remaining_7d_includes_market_beyond_window(market: Market):
    m = market_ending_in(market, timedelta(days=10))
    assert passes_time_remaining(m, "7d", NOW) is True


def test_time_remaining_30d_excludes_market_within_window(market: Market):
    # 30d backward-compat: guards the TIME_WINDOW_DAYS retype to dict[str, float].
    m = market_ending_in(market, timedelta(days=20))
    assert passes_time_remaining(m, "30d", NOW) is False


def test_time_remaining_30d_includes_market_beyond_window(market: Market):
    m = market_ending_in(market, timedelta(days=40))
    assert passes_time_remaining(m, "30d", NOW) is True


def test_created_date_excludes_market_younger_than_window(market: Market):
    m = market_created_ago(market, timedelta(hours=12))
    assert passes_created_date(m, "1d", NOW) is False


def test_created_date_includes_market_older_than_window(market: Market):
    m = market_created_ago(market, timedelta(hours=36))
    assert passes_created_date(m, "1d", NOW) is True


def test_created_date_all_window_passes_everything(market: Market):
    m = market_created_ago(market, timedelta(minutes=5))
    assert passes_created_date(m, "all", NOW) is True


def test_created_date_7d_excludes_market_younger_than_window(market: Market):
    m = market_created_ago(market, timedelta(days=5))
    assert passes_created_date(m, "7d", NOW) is False


def test_created_date_7d_includes_market_older_than_window(market: Market):
    m = market_created_ago(market, timedelta(days=10))
    assert passes_created_date(m, "7d", NOW) is True


def test_created_date_30d_excludes_market_younger_than_window(market: Market):
    # created_date shares TIME_WINDOW_DAYS; the dict retype must not shift its thresholds.
    m = market_created_ago(market, timedelta(days=20))
    assert passes_created_date(m, "30d", NOW) is False


def test_created_date_30d_includes_market_older_than_window(market: Market):
    m = market_created_ago(market, timedelta(days=40))
    assert passes_created_date(m, "30d", NOW) is True


# effective() is Decimal-annotated but reused verbatim for TimeRemainingFilter strings in the
# FILTER_CHAIN. These guard that the None-vs-present branch works for string values too — in
# particular that the truthy-but-loosening literal "all" is never mistaken for "unset".


def test_effective_returns_string_value_when_present():
    assert effective("12h", "all") == "12h"


def test_effective_falls_back_to_string_when_value_is_none():
    assert effective(None, "7d") == "7d"


def test_effective_all_override_is_not_treated_as_unset():
    assert effective("all", "7d") == "all"
