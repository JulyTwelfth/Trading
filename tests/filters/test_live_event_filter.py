from datetime import datetime, timedelta, timezone

from app.farm.filters import passes_live_event_filter
from app.farm.schemas import Market

NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


def test_live_event_filter_excludes_within_2h(market: Market):
    m = market.model_copy(update={"game_start_time": NOW + timedelta(hours=1)})
    assert passes_live_event_filter(m, NOW) is False


def test_live_event_filter_allows_outside_2h(market: Market):
    m = market.model_copy(update={"game_start_time": NOW + timedelta(hours=4)})
    assert passes_live_event_filter(m, NOW) is True


def test_live_event_filter_excludes_during_live_period(market: Market):
    m = market.model_copy(update={"game_start_time": NOW - timedelta(hours=2)})
    assert passes_live_event_filter(m, NOW) is False


def test_live_event_filter_allows_pre_window(market: Market):
    m = market.model_copy(update={"game_start_time": NOW + timedelta(hours=8)})
    assert passes_live_event_filter(m, NOW) is True


def test_live_event_filter_allows_non_sports(market: Market):
    assert market.game_start_time is None
    assert passes_live_event_filter(market, NOW) is True
