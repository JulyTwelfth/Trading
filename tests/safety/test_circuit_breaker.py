from datetime import datetime, timedelta, timezone

from app.constants import (
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    CIRCUIT_BREAKER_MAX_FAILURES,
    CIRCUIT_BREAKER_WINDOW_SECONDS,
)
from app.farm.health import is_paused, mark_paused, record_market_failure
from app.farm.schemas import FarmState, MarketHealth


def test_circuit_breaker_does_not_trip_before_threshold(farm_state: FarmState):
    market_id = "market-A"
    tripped = False
    for _ in range(CIRCUIT_BREAKER_MAX_FAILURES - 1):
        tripped = record_market_failure(farm_state, market_id)
    assert tripped is False


def test_circuit_breaker_trips_at_threshold(farm_state: FarmState):
    market_id = "market-A"
    tripped = False
    for _ in range(CIRCUIT_BREAKER_MAX_FAILURES):
        tripped = record_market_failure(farm_state, market_id)
    assert tripped is True


def test_circuit_breaker_prunes_stale_failures(farm_state: FarmState):
    market_id = "market-A"
    now = 1_000_000.0
    stale = now - CIRCUIT_BREAKER_WINDOW_SECONDS - 10
    # Inject failures outside the window — they should be pruned before counting.
    for _ in range(CIRCUIT_BREAKER_MAX_FAILURES):
        record_market_failure(farm_state, market_id, now=stale)
    # A single fresh failure must not trip even though 5 stale ones existed.
    tripped = record_market_failure(farm_state, market_id, now=now)
    assert tripped is False


def test_mark_paused_sets_future_paused_until(farm_state: FarmState):
    market_id = "market-A"
    before = datetime.now(timezone.utc)
    mark_paused(farm_state, market_id)
    after = datetime.now(timezone.utc)
    paused_until = farm_state.health[market_id].paused_until
    assert paused_until is not None
    assert before + timedelta(seconds=CIRCUIT_BREAKER_COOLDOWN_SECONDS - 1) <= paused_until
    assert paused_until <= after + timedelta(seconds=CIRCUIT_BREAKER_COOLDOWN_SECONDS + 1)


def test_is_paused_true_when_paused_until_is_future(farm_state: FarmState):
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) + timedelta(minutes=1)
    )
    assert is_paused(farm_state, "market-A") is True


def test_is_paused_false_after_cooldown(farm_state: FarmState):
    farm_state.health["market-A"] = MarketHealth(
        paused_until=datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    assert is_paused(farm_state, "market-A") is False


def test_is_paused_false_for_unknown_market(farm_state: FarmState):
    assert is_paused(farm_state, "never-seen") is False
