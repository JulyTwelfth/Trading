"""AI-benchmark 'arena score' threshold longshots (e.g. "reach 1520 overall arena score") are
always blocked — a new SOTA-model release reprices the at-the-money threshold and runs over the
passive quote (net -$38.59 across logs). Tight two-phrase match ("arena score" / "debut at a
score of"), grep-verified cross-bucket clean. Ordinary AI markets (best-model rankings, price
moves) still pass."""

from datetime import datetime, timezone
from decimal import Decimal

from app.farm.filters import first_failing_filter, passes_ai_arena_filter
from app.farm.schemas import Market

NOW = datetime(2026, 6, 15, tzinfo=timezone.utc)

LOSER_SLUG = "will-any-ai-model-reach-1520-overall-arena-score-by-september-30-2026"


def mkt(market: Market, question: str, slug: str = "s") -> Market:
    return market.model_copy(update={"question": question, "slug": slug})


def test_blocks_real_loser_slugs(market: Market):
    # Regression on the 3 actual fills that drove the -$38.59 net-of-reward loss.
    for slug in (
        "will-any-ai-model-reach-1520-overall-arena-score-by-september-30-2026",
        "will-the-next-model-released-by-openai-debut-at-a-score-of-at-least-1450",
        "will-any-ai-model-reach-1580-coding-arena-score-by-december-31-2026",
    ):
        assert passes_ai_arena_filter(mkt(market, "?", slug)) is False, slug


def test_accuracy_guard_net_positive_siblings_pass(market: Market):
    # These are net-POSITIVE AI reward-farm siblings. A broad regex (e.g. matching on
    # "best ai model" or a general AI/model catch-all) would have wrongly banned them too —
    # the tight two-phrase pattern must NOT regress into that rejected broad form.
    for slug in (
        "will-anthropic-have-the-third-best-ai-model-at-the-end-of-june-2026",  # +$1.98
        "will-openai-have-a-1-ai-model",  # +$1.52
        "will-claude-go-down-6-8-july",  # +$2.44
    ):
        assert passes_ai_arena_filter(mkt(market, "?", slug)) is True, slug


def test_accepted_casualty_gemini_leaderboard_debut(market: Market):
    # Disclosed collateral: this sibling is net +$1.81, but its slug is lexically identical to
    # the losers ("debut at a score of") — winners and losers can't be told apart by wording, so
    # no tighter regex separates them and this market is knowingly sacrificed.
    slug = (
        "will-the-next-google-gemini-pro-model-added-to-the-arena-leaderboard"
        "-debut-at-a-score-of-at-least-1510"
    )
    assert passes_ai_arena_filter(mkt(market, "?", slug)) is False


def test_no_false_positive_on_arena_venue(market: Market):
    # 'arena' without 'score' (e.g. a literal sports/concert venue) must NOT be blocked.
    slug = "will-the-new-arena-sell-out-in-2026"
    assert passes_ai_arena_filter(mkt(market, "?", slug)) is True


def test_case_insensitive(market: Market):
    q = "Will any AI model REACH 1520 overall ARENA SCORE by September 30 2026?"
    assert passes_ai_arena_filter(mkt(market, q)) is False


def test_ai_arena_named_in_funnel(market: Market, farm_state):
    f = farm_state.config.filters
    m = mkt(market, "?", LOSER_SLUG)
    assert first_failing_filter(m, f, NOW) == "ai_arena_score"


def test_ai_arena_checked_before_user_filters(market: Market, farm_state):
    f = farm_state.config.filters.model_copy(update={"vol_min": Decimal("999999999")})
    m = mkt(market, "?", LOSER_SLUG)
    # Fails both ai_arena_score and volume; ai_arena_score is earlier in the chain.
    assert first_failing_filter(m, f, NOW) == "ai_arena_score"
