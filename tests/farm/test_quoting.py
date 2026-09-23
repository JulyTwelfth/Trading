from decimal import Decimal

from app.farm.quoting import (
    ceil_to_tick,
    compute_quote,
    floor_to_tick,
    size_per_market,
    two_leg_cost,
)


def test_compute_quote_happy_path(market):
    # max_spread 3c -> 0.03 price; tick 0.01; distance 0.02
    assert compute_quote(market, Decimal("0.5")) == (Decimal("0.48"), Decimal("0.52"))


def test_compute_quote_midpoint_at_or_below_zero(market):
    assert compute_quote(market, Decimal("0")) is None
    assert compute_quote(market, Decimal("-0.1")) is None


def test_compute_quote_midpoint_at_or_above_one(market):
    assert compute_quote(market, Decimal("1")) is None
    assert compute_quote(market, Decimal("1.5")) is None


def test_compute_quote_distance_non_positive(market):
    # tick == max_spread -> distance 0 -> no room
    m = market.model_copy(update={"tick_size": Decimal("0.03")})
    assert compute_quote(m, Decimal("0.5")) is None


def test_compute_quote_bid_rounds_to_or_below_zero(market):
    # midpoint so close to 0 that bid rounds to <= 0
    assert compute_quote(market, Decimal("0.01")) is None


def test_compute_quote_ask_rounds_to_or_above_one(market):
    # midpoint so close to 1 that ask rounds to >= 1
    assert compute_quote(market, Decimal("0.99")) is None


def test_compute_quote_ordering_guard(market):
    # coarse tick: bid rounds up onto the midpoint -> bid >= midpoint -> rejected
    m = market.model_copy(update={"tick_size": Decimal("0.02")})
    assert compute_quote(m, Decimal("0.5")) is None


def test_two_leg_cost_both_legs_quoteable(market):
    # size = max(min_order_size 5, rewards_min_size 100) = 100; both bids 0.48
    cost = two_leg_cost(market, Decimal("0.5"), Decimal("0.5"))
    assert cost == Decimal("100") * (Decimal("0.48") + Decimal("0.48"))


def test_two_leg_cost_unquoteable_leg_returns_none(market):
    # second leg midpoint at the boundary -> compute_quote None -> whole cost None
    assert two_leg_cost(market, Decimal("0.5"), Decimal("0")) is None


def test_size_per_market_picks_larger_minimum(market):
    assert size_per_market(market) == Decimal("100")
    smaller = market.model_copy(update={"rewards_min_size": Decimal("2")})
    assert size_per_market(smaller) == Decimal("5")  # falls back to min_order_size


def test_ceil_and_floor_to_tick():
    assert ceil_to_tick(Decimal("0.471"), Decimal("0.01")) == Decimal("0.48")
    assert floor_to_tick(Decimal("0.479"), Decimal("0.01")) == Decimal("0.47")
