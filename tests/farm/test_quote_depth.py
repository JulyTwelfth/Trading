"""quote_depth places each leg closer to / farther from the midpoint, trading reward
score against fill risk: safe = zone edge (original), aggressive = 1 tick from mid,
normal = ~half the max spread. compute_quote / two_leg_cost honour the chosen depth."""

from datetime import datetime, timezone
from decimal import Decimal

from app.farm.quoting import compute_quote, quote_distance, two_leg_cost
from app.farm.schemas import Market


def market(**overrides) -> Market:
    base = dict(
        condition_id="c",
        slug="s",
        question="?",
        yes_token_id="y",
        no_token_id="n",
        rewards_max_spread_cents=Decimal("4.5"),
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


# ── quote_distance ───────────────────────────────────────────────────────────


def test_distance_safe_is_zone_edge():
    # max_spread 4.5c, tick 1c → edge = 3.5c.
    assert quote_distance("safe", Decimal("0.045"), Decimal("0.01")) == Decimal("0.035")


def test_distance_aggressive_is_one_tick():
    assert quote_distance("aggressive", Decimal("0.045"), Decimal("0.01")) == Decimal("0.01")


def test_distance_normal_is_half_rounded_to_tick():
    # half of 4.5c = 2.25c, rounded to the 1c tick grid = 2c.
    assert quote_distance("normal", Decimal("0.045"), Decimal("0.01")) == Decimal("0.02")


def test_distance_normal_distinct_from_aggressive_on_common_spread():
    # 3.5c spread / 1c tick (the most common market): half=1.75c must ROUND to 2c, not
    # floor to 1c — otherwise normal collapses into aggressive (the live-test wart).
    v, tick = Decimal("0.035"), Decimal("0.01")
    assert quote_distance("aggressive", v, tick) == Decimal("0.01")
    assert quote_distance("normal", v, tick) == Decimal("0.02")
    assert quote_distance("safe", v, tick) == Decimal("0.025")


def test_distance_ordering_aggressive_lt_normal_lt_safe():
    v, tick = Decimal("0.045"), Decimal("0.01")
    agg = quote_distance("aggressive", v, tick)
    nor = quote_distance("normal", v, tick)
    safe = quote_distance("safe", v, tick)
    assert agg < nor < safe  # closer to mid → smaller distance


def test_distance_aggressive_clamps_when_spread_too_narrow():
    # max_spread 1.5c, tick 1c → edge = 0.5c; 1 tick (1c) would overshoot, so clamp to edge.
    assert quote_distance("aggressive", Decimal("0.015"), Decimal("0.01")) == Decimal("0.005")


def test_distance_degenerate_spread_returns_nonpositive():
    # max_spread == tick → edge 0; caller turns this into None.
    assert quote_distance("safe", Decimal("0.01"), Decimal("0.01")) == Decimal("0")


# ── compute_quote per depth ──────────────────────────────────────────────────


def test_compute_quote_bid_moves_toward_mid_with_depth():
    m = market()  # max_spread 4.5c, tick 1c
    mid = Decimal("0.50")
    assert compute_quote(m, mid, "safe")[0] == Decimal("0.47")  # 3c from mid
    assert compute_quote(m, mid, "normal")[0] == Decimal("0.48")  # 2c from mid
    assert compute_quote(m, mid, "aggressive")[0] == Decimal("0.49")  # 1c from mid


def test_compute_quote_default_is_safe():
    m = market()
    assert compute_quote(m, Decimal("0.50")) == compute_quote(m, Decimal("0.50"), "safe")


def test_compute_quote_ask_symmetric_to_depth():
    m = market()
    mid = Decimal("0.50")
    assert compute_quote(m, mid, "aggressive")[1] == Decimal("0.51")
    assert compute_quote(m, mid, "safe")[1] == Decimal("0.53")


# ── two_leg_cost per depth ───────────────────────────────────────────────────


def test_two_leg_cost_higher_when_aggressive():
    m = market()  # size = max(5, 100) = 100
    mid = Decimal("0.50")
    safe = two_leg_cost(m, mid, mid, "safe")
    agg = two_leg_cost(m, mid, mid, "aggressive")
    # safe: 100*(0.47+0.47)=94 ; aggressive: 100*(0.49+0.49)=98
    assert safe == Decimal("94")
    assert agg == Decimal("98")
    assert agg > safe  # closer to mid → higher bids → more capital committed
