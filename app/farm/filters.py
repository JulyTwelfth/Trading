import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.constants import (
    CRISIS_KEYWORDS,
    ELECTION_RESULT_WINDOW_HOURS,
    LIVE_EVENT_PREGAME_HOURS,
    TIME_WINDOW_DAYS,
)
from app.farm.schemas import FarmFilters, Market
from app.types import Change24hFilter, CreatedDateFilter, Range24hFilter, TimeRemainingFilter

RANGE_24H_THRESHOLDS: dict[str, Decimal] = {
    "lt5": Decimal("0.05"),
    "lt10": Decimal("0.10"),
    "lt20": Decimal("0.20"),
}

CRISIS_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in CRISIS_KEYWORDS) + r")\b", re.IGNORECASE
)

MENTION_RE = re.compile(
    r"\b(?:say|says|said|tweet|tweets|tweeted|mention|mentions|mentioned"
    r"|insult|insults|insulted|use the word)\b",
    re.IGNORECASE,
)

TRANSFER_RE = re.compile(
    r"\bstays?\s+at\b"
    r"|\bjoins?\b(?!\s+the\b)"
    r"|\bsigns?\s+(?:for|with)\b"
    r"|\btransfers?\b",
    re.IGNORECASE,
)

WC_CONTEXT_RE = re.compile(r"\b(?:world cup|fifa)\b", re.IGNORECASE)
WC_TEAM_OUTCOME_RE = re.compile(
    r"\b(?:"
    r"be eliminated"
    r"|stage of elimination"
    r"|advanc(?:e|es|ed|ing|ement)"
    r"|reach(?:es)? the (?:round of \d+|last \d+|(?:quarter|semi)[ -]?finals?|finals?)"
    r"|round of \d+"
    r"|go(?:es)? unbeaten"
    r"|win(?:s)? (?:their|the) group"
    r"|group winner"
    r"|win(?:s)? the (?:20\d\d )?(?:fifa )?world cup"
    r"|finish(?:es)? (?:top|first|second|third|fourth|bottom|last)"
    r"|be an advancing group stage"
    r")\b",
    re.IGNORECASE,
)
WC_AGGREGATE_STAT_RE = re.compile(
    r"\b(?:"
    r"go to extra time|matches?\s+to\s+extra\s+time"
    r"|penalty shootout|decided by (?:a )?shootout"
    r"|missed penalt(?:y|ies)"
    r"|suspended by weather|weather protocol"
    r"|var stoppages?"
    r"|total (?:number of )?goals"
    r"|red cards?|yellow cards?|own goals?|hat[- ]?tricks?"
    r"|highest scoring team|most goals"
    r")\b",
    re.IGNORECASE,
)

DRAFT_RE = re.compile(
    r"\b(?:"
    r"\d+(?:st|nd|rd|th)\s+overall\s+pick"
    r"|overall\s+(?:draft\s+)?pick"
    r"|lottery\s+pick"
    r"|(?:1st|first|2nd|second|3rd|third)\s+round\s+pick"
    r"|number\s+(?:one|1)\s+(?:overall\s+)?pick"
    r"|be\s+drafted"
    r"|(?:nba|wnba|nfl|mlb|nhl)\s+draft"
    r")\b",
    re.IGNORECASE,
)

IPO_MA_RE = re.compile(
    r"\b(?:"
    r"ipo|initial public offering|going?\s+public|goes public|underwrit(?:er|ing)"
    r"|merger|merge\s+with|mergeracquisition|takeover|buyout"
    r"|be acquired|acquir(?:e|es|ed|ing|ition)"
    r"|q[1-4]\s+(?:earnings|investment banking)"
    r"|earnings\s+(?:report|call|beat|miss)"
    r")\b",
    re.IGNORECASE,
)

# AI benchmark score-threshold markets (LMArena / coding-arena "reach 15xx" longshots).
# Net-of-reward -$38.59 across logs; a new SOTA-model release reprices the at-the-money
# threshold and runs over the passive quote. "arena score" and "debut at a score of" are
# benchmark-specific (grep-verified: zero non-AI collisions). Deliberately tight — the broad
# "best ai model"/"gdp|cpi" forms were rejected for banning net-POSITIVE reward-farm siblings.
AI_ARENA_SCORE_RE = re.compile(
    r"\barena score\b"
    r"|\bdebut(?:s)?\s+at\s+a\s+score\s+of\b",
    re.IGNORECASE,
)

TEMPERATURE_RE = re.compile(r"\b(?:highest|lowest)\s+temperature\b", re.IGNORECASE)

VALUATION_RE = re.compile(r"\bvaluation\b", re.IGNORECASE)

ELECTION_RE = re.compile(
    r"\b(?:"
    r"election|primar(?:y|ies)|by[- ]election|caucus|runoff"
    r"|governor|gubernatorial|senate|senator|mayoral|parliamentary"
    r"|prime minister|premier of|presidency member|majority leader"
    r"|presidential election|nominee|electoral"
    r")\b",
    re.IGNORECASE,
)


def passes_volume(market: Market, vol_min: Decimal, vol_max: Decimal) -> bool:
    return vol_min <= market.volume_24h <= vol_max


def passes_liquidity(market: Market, liq_min: Decimal, liq_max: Decimal) -> bool:
    return liq_min <= market.liquidity <= liq_max


def passes_spread(market: Market, spread_min: Decimal, spread_max: Decimal) -> bool:
    return spread_min <= market.spread_cents <= spread_max


def passes_reward(market: Market, reward_min: Decimal) -> bool:
    return market.rewards_rate_per_day >= reward_min


def passes_time_remaining(market: Market, window: TimeRemainingFilter, now: datetime) -> bool:
    if window == "all":
        return True
    return market.end_date - now > timedelta(days=TIME_WINDOW_DAYS[window])


def passes_created_date(market: Market, window: CreatedDateFilter, now: datetime) -> bool:
    if window == "all":
        return True
    return now - market.created_at > timedelta(days=TIME_WINDOW_DAYS[window])


def passes_change_24h(market: Market, bucket: Change24hFilter) -> bool:
    abs_change = abs(market.price_change_24h)
    match bucket:
        case "all":
            return True
        case "lt10":
            return abs_change < Decimal("0.10")
        case "gt10":
            return abs_change >= Decimal("0.10")
        case "gt20":
            return abs_change >= Decimal("0.20")


def passes_live_event_filter(market: Market, now: datetime) -> bool:
    if market.game_start_time is None:
        return True
    return now < market.game_start_time - timedelta(hours=LIVE_EVENT_PREGAME_HOURS)


def passes_range_24h(price_range: Decimal | None, bucket: Range24hFilter) -> bool:
    if bucket == "all":
        return True
    if price_range is None:
        return False
    return price_range < RANGE_24H_THRESHOLDS[bucket]


def passes_crisis_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    return CRISIS_RE.search(text) is None


def passes_mention_filter(market: Market) -> bool:
    return MENTION_RE.search(f"{market.question} {market.slug}") is None


def passes_transfer_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    return TRANSFER_RE.search(text) is None


def passes_wc_aggregate_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    if WC_CONTEXT_RE.search(text) is None:
        return True
    return WC_AGGREGATE_STAT_RE.search(text) is None


def passes_wc_team_outcome_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    if WC_CONTEXT_RE.search(text) is None:
        return True
    return WC_TEAM_OUTCOME_RE.search(text) is None


def passes_draft_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    return DRAFT_RE.search(text) is None


def passes_ipo_ma_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    return IPO_MA_RE.search(text) is None


def passes_ai_arena_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    return AI_ARENA_SCORE_RE.search(text) is None


def passes_weather_filter(market: Market) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    return TEMPERATURE_RE.search(text) is None


def passes_valuation_filter(market: Market) -> bool:
    """Block private-company valuation-milestone markets ("<company>'s valuation hit $YB by <date>"
    — Stripe, Lambda). Like weather/IPO they gap on a discrete leak/report for little-to-no reward;
    repeat single-fill losers (Stripe -$7, Lambda -$13). FDV token-launch markets don't match
    "valuation", so those (net-positive) stay farmable. Always on."""
    text = f"{market.question} {market.slug}".replace("-", " ")
    return VALUATION_RE.search(text) is None


def passes_election_window(market: Market, now: datetime) -> bool:
    text = f"{market.question} {market.slug}".replace("-", " ")
    if ELECTION_RE.search(text) is None:
        return True
    return market.end_date - now > timedelta(hours=ELECTION_RESULT_WINDOW_HOURS)


def passes_sports_schedule(market: Market) -> bool:
    return not market.sports_event_active


def passes_price(market: Market, price_min: Decimal | None, price_max: Decimal | None) -> bool:
    if price_min is None and price_max is None:
        return True
    if market.midpoint is None:
        return False
    if price_min is not None and market.midpoint < price_min:
        return False
    if price_max is not None and market.midpoint > price_max:
        return False
    return True


def passes_zone_liquidity(market: Market, zone_liq_max: Decimal | None) -> bool:
    if zone_liq_max is None:
        return True
    if market.zone_liquidity is None:
        return False
    return market.zone_liquidity <= zone_liq_max


def passes_max_fill_loss(market: Market, max_fill_loss: Decimal | None) -> bool:
    if max_fill_loss is None:
        return True
    if market.exit_loss is None:
        return False
    return market.exit_loss <= max_fill_loss


def passes_bid_depth(market: Market, mult: Decimal | None, order_size: Decimal) -> bool:
    if mult is None:
        return True
    if market.yes_bid_depth is None or market.no_bid_depth is None:
        return False
    return min(market.yes_bid_depth, market.no_bid_depth) >= mult * order_size


def effective[T](value: T | None, fallback: T) -> T:
    """Per-market override `value` if set, else the global `fallback`. The generic types both
    cases correctly on its own: for a required global (liq_min/reward_min/time_remaining) T binds
    to the concrete type and the result is never None, while for an optional one (zone_liq_max) T
    binds to `Decimal | None` and the result is optional too — which is why callers of the latter
    still have to None-check."""
    return value if value is not None else fallback


FILTER_CHAIN: list[tuple[Callable[[Market, FarmFilters, datetime], bool], str]] = [
    (lambda m, f, t: passes_live_event_filter(m, t), "live_event"),
    (lambda m, f, t: passes_crisis_filter(m), "crisis"),
    (lambda m, f, t: passes_mention_filter(m), "mention"),
    (lambda m, f, t: passes_transfer_filter(m), "transfer"),
    (lambda m, f, t: passes_wc_aggregate_filter(m), "wc_aggregate_stat"),
    (lambda m, f, t: passes_wc_team_outcome_filter(m), "wc_team_outcome"),
    (lambda m, f, t: passes_draft_filter(m), "draft"),
    (lambda m, f, t: passes_ipo_ma_filter(m), "ipo_ma_earnings"),
    (lambda m, f, t: passes_ai_arena_filter(m), "ai_arena_score"),
    (lambda m, f, t: passes_weather_filter(m), "weather_temp"),
    (lambda m, f, t: passes_valuation_filter(m), "valuation"),
    (lambda m, f, t: passes_sports_schedule(m), "sports_event_active"),
    (lambda m, f, t: passes_election_window(m, t), "election_window"),
    (lambda m, f, t: passes_volume(m, f.vol_min, f.vol_max), "volume"),
    (
        lambda m, f, t: passes_liquidity(m, effective(m.effective_liq_min, f.liq_min), f.liq_max),
        "liquidity",
    ),
    (lambda m, f, t: passes_spread(m, f.spread_min, f.spread_max), "spread"),
    (lambda m, f, t: passes_reward(m, effective(m.effective_reward_min, f.reward_min)), "reward"),
    (
        lambda m, f, t: passes_time_remaining(
            m, effective(m.effective_time_remaining, f.time_remaining), t
        ),
        "time_remaining",
    ),
    (lambda m, f, t: passes_created_date(m, f.created_date, t), "created_date"),
    (lambda m, f, t: passes_change_24h(m, f.change_24h), "change_24h"),
]


def first_failing_filter(
    market: Market, filters: FarmFilters, now: datetime | None = None
) -> str | None:
    moment = now if now is not None else datetime.now(timezone.utc)
    for check, label in FILTER_CHAIN:
        if not check(market, filters, moment):
            return label
    if filters.price_min is not None or filters.price_max is not None:
        if market.midpoint is None:
            return "price_unknown"
        if not passes_price(market, filters.price_min, filters.price_max):
            return "price"
    zone_liq_max = effective(market.effective_zone_liq_max, filters.zone_liq_max)
    if zone_liq_max is not None:
        if market.zone_liquidity is None:
            return "zone_unknown"
        if market.zone_liquidity > zone_liq_max:
            return "zone_liquidity"
    if filters.max_fill_loss is not None:
        if market.exit_loss is None:
            return "exit_loss_unknown"
        if market.exit_loss > filters.max_fill_loss:
            return "exit_loss"
    if filters.min_bid_depth_mult is not None:
        if market.yes_bid_depth is None or market.no_bid_depth is None:
            return "bid_depth_unknown"
        order_size = max(market.min_order_size, market.rewards_min_size)
        if not passes_bid_depth(market, filters.min_bid_depth_mult, order_size):
            return "bid_depth"
    return None


def passes_all(market: Market, filters: FarmFilters, now: datetime | None = None) -> bool:
    return first_failing_filter(market, filters, now) is None
