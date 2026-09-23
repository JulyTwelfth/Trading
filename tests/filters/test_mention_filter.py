"""'Mention markets' (Will X say/tweet/insult/mention Y) are always blocked — a single post
reprices them ~40 points, pure adverse selection for a passive LP. Whole-word match, so words
like 'essay' are NOT false-positives. Ordinary markets pass. Ported from PR #29."""

from datetime import datetime, timezone
from decimal import Decimal

from app.farm.filters import first_failing_filter, passes_mention_filter
from app.farm.schemas import Market

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)


def mkt(market: Market, question: str, slug: str = "s") -> Market:
    return market.model_copy(update={"question": question, "slug": slug})


def test_blocks_say_tweet_mention_markets(market: Market):
    for q in (
        "Will Trump say 'tariff' in the speech?",
        "Will Elon tweet about Mars this week?",
        "Will the Fed mention inflation?",
        "Will the chair use the word recession?",
    ):
        assert passes_mention_filter(mkt(market, q)) is False, q


def test_blocks_insult_markets_by_slug(market: Market):
    # The exact markets the bot actually got filled on (run history Runs 8-9).
    for slug in (
        "will-donald-trump-publicly-insult-nicols-maduro-by-june-30-2026",
        "will-donald-trump-publicly-insult-emmanuel-macron-by-june-30-2026",
    ):
        assert passes_mention_filter(mkt(market, "?", slug)) is False, slug


def test_case_insensitive(market: Market):
    assert passes_mention_filter(mkt(market, "Will X TWEET tonight?")) is False
    assert passes_mention_filter(mkt(market, "Will he SAY it?")) is False


def test_whole_word_no_false_positive(market: Market):
    # 'essay' contains 'say' but must NOT be blocked (whole-word match).
    q = "Will she win the national essay contest in 2026?"
    assert passes_mention_filter(mkt(market, q)) is True


def test_allows_ordinary_markets(market: Market):
    for q in (
        "Will BTC hit 100k by 2026?",
        "Will Trump win the 2028 election?",
        "Will the Fed cut rates in July?",
    ):
        assert passes_mention_filter(mkt(market, q)) is True, q


def test_mention_named_in_funnel_after_crisis(market: Market, farm_state):
    f = farm_state.config.filters
    m = mkt(market, "Will Trump tweet about it?")
    assert first_failing_filter(m, f, NOW) == "mention"


def test_crisis_takes_precedence_over_mention(market: Market, farm_state):
    f = farm_state.config.filters
    # Contains both a crisis term (war) and a mention verb (say) → crisis is earlier in chain.
    m = mkt(market, "Will Putin say there will be war?")
    assert first_failing_filter(m, f, NOW) == "crisis"


def test_mention_checked_before_user_filters(market: Market, farm_state):
    f = farm_state.config.filters.model_copy(update={"vol_min": Decimal("999999999")})
    m = mkt(market, "Will he tweet today?")
    # Fails both mention and volume; mention is earlier in the chain.
    assert first_failing_filter(m, f, NOW) == "mention"
