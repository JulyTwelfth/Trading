from app.farm.health import in_guard_pull_cooldown, is_event_excluded, is_paused
from app.farm.schemas import FarmState, MarketPosition
from app.farm.volatility import is_blacklisted


def quote_block_reason(state: FarmState, condition_id: str, event_slug: str) -> str | None:
    """The reason we must NOT have a resting quote on this market, or None if it's clear."""
    if state.killed:
        return "killed"
    if is_paused(state, condition_id):
        return "paused"
    if is_blacklisted(state, condition_id):
        return "blacklisted"
    if condition_id in state.excluded_markets:
        return "excluded_market"
    if event_slug and is_event_excluded(state, event_slug):
        return "excluded_event"
    if in_guard_pull_cooldown(state, condition_id):
        return "guard_cooldown"
    return None


def should_quote(state: FarmState, pos: MarketPosition) -> bool:
    """True when none of the protective flags block quoting this position's market."""
    return quote_block_reason(state, pos.market.condition_id, pos.market.event_slug) is None
