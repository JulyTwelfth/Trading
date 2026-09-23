"""Geopolitics/war/crisis markets are always blocked (whole-word match, so common words
like 'Warriors' are NOT false-positives). Sports futures and ordinary politics pass."""

from datetime import datetime, timezone

from app.farm.filters import first_failing_filter, passes_crisis_filter
from app.farm.schemas import Market

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def mkt(market: Market, question: str, slug: str = "s") -> Market:
    return market.model_copy(update={"question": question, "slug": slug})


def test_blocks_war_market(market: Market):
    assert passes_crisis_filter(mkt(market, "Will there be a nuclear war in 2026?")) is False


def test_blocks_by_slug(market: Market):
    assert passes_crisis_filter(mkt(market, "?", "russia-ukraine-ceasefire-by-july")) is False


def test_blocks_country_crisis(market: Market):
    for q in ("Will Iran enrich uranium?", "Israel-Hamas hostage deal?", "Russia invades Moldova?"):
        assert passes_crisis_filter(mkt(market, q)) is False, q


def test_blocks_adjective_forms(market: Market):
    # The plain stem ("israel") with word boundaries misses the adjective ("israeli") — the
    # israeli-forces-enter-choukine market slipped through and got quoted+filled. Block both.
    for slug in ("israeli-forces-enter-choukine-by-june-30", "will-ukrainian-troops-hold-the-line"):
        assert passes_crisis_filter(mkt(market, "?", slug)) is False, slug


def test_allows_sports_future_with_warriors(market: Market):
    # 'Warriors' must NOT trip the whole-word 'war' keyword.
    q = "Will the Golden State Warriors win the title?"
    assert passes_crisis_filter(mkt(market, q)) is True


def test_allows_ordinary_politics_and_awards(market: Market):
    for q in ("Will Elizabeth Warren run in 2028?", "Best Picture award?", "Move forward bill?"):
        assert passes_crisis_filter(mkt(market, q)) is True, q


def test_crisis_is_named_in_funnel_after_live_event(market: Market, farm_state):
    f = farm_state.config.filters
    m = mkt(market, "Will Russia invade?")
    assert first_failing_filter(m, f, NOW) == "crisis"


def test_crisis_checked_before_user_filters(market: Market, farm_state):
    from decimal import Decimal

    f = farm_state.config.filters.model_copy(update={"vol_min": Decimal("999999999")})
    m = mkt(market, "Iran ceasefire?")
    # Fails both crisis and volume; crisis is earlier in the chain.
    assert first_failing_filter(m, f, NOW) == "crisis"
