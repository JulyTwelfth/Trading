import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.constants import (
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    CIRCUIT_BREAKER_MAX_FAILURES,
    CIRCUIT_BREAKER_WINDOW_SECONDS,
    DEPTH_GUARD_PULL_COOLDOWN_SECONDS,
    GUARD_TRIP_COOLOFF_TIERS,
    GUARD_TRIP_SEVERE_LOSS,
)
from app.farm.schemas import FarmState, MarketHealth
from app.infra.strategy_log import strat

logger = logging.getLogger(__name__)


def record_market_failure(state: FarmState, market_id: str, now: float | None = None) -> bool:
    moment = now if now is not None else datetime.now(timezone.utc).timestamp()
    health = state.health.setdefault(market_id, MarketHealth())
    cutoff = moment - CIRCUIT_BREAKER_WINDOW_SECONDS
    health.recent_failures = [t for t in health.recent_failures if t >= cutoff]
    health.recent_failures.append(moment)
    if len(health.recent_failures) >= CIRCUIT_BREAKER_MAX_FAILURES:
        logger.warning(
            "circuit breaker threshold reached market=%s failures=%d window=%ds",
            market_id,
            len(health.recent_failures),
            CIRCUIT_BREAKER_WINDOW_SECONDS,
        )
        strat("breaker_trip", market=market_id, failures=len(health.recent_failures))
        return True
    logger.debug(
        "market failure recorded market=%s count=%d", market_id, len(health.recent_failures)
    )
    return False


def mark_paused(state: FarmState, market_id: str) -> None:
    health = state.health.setdefault(market_id, MarketHealth())
    health.paused_until = datetime.now(timezone.utc) + timedelta(
        seconds=CIRCUIT_BREAKER_COOLDOWN_SECONDS
    )
    logger.warning(
        "market paused (circuit breaker) market=%s until=%s cooldown=%ds",
        market_id,
        health.paused_until.isoformat(),
        CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    )
    strat("pause", market=market_id, cooldown_s=CIRCUIT_BREAKER_COOLDOWN_SECONDS)


def is_paused(state: FarmState, market_id: str, now: datetime | None = None) -> bool:
    health = state.health.get(market_id)
    if health is None or health.paused_until is None:
        return False
    moment = now if now is not None else datetime.now(timezone.utc)
    return health.paused_until > moment


def mark_guard_pulled(
    state: FarmState,
    market_id: str,
    now: datetime | None = None,
    cooldown_s: int = DEPTH_GUARD_PULL_COOLDOWN_SECONDS,
) -> None:
    health = state.health.setdefault(market_id, MarketHealth())
    health.guard_pull_until = (now or datetime.now(timezone.utc)) + timedelta(seconds=cooldown_s)
    logger.info(
        "market guard-pulled market=%s until=%s cooldown=%ds",
        market_id,
        health.guard_pull_until.isoformat(),
        cooldown_s,
    )
    strat("guard_cooldown", market=market_id, cooldown_s=cooldown_s)


def in_guard_pull_cooldown(state: FarmState, market_id: str, now: datetime | None = None) -> bool:
    health = state.health.get(market_id)
    if health is None or health.guard_pull_until is None:
        return False
    moment = now if now is not None else datetime.now(timezone.utc)
    return health.guard_pull_until > moment


def record_guard_trip(
    state: FarmState,
    market_id: str,
    est_loss: Decimal,
    now: datetime | None = None,
) -> None:
    """Escalating cool-off for a market whose depth/exit-loss guard trips with a SEVERE estimated
    loss. Noise trips (est_loss < GUARD_TRIP_SEVERE_LOSS) are ignored; each severe trip bumps the
    tier (30 min -> session-long) and extends (never shortens) blacklist_until, so a market that
    keeps flashing big danger stops being re-entered every fixed guard cooldown."""
    if est_loss < GUARD_TRIP_SEVERE_LOSS:
        return
    health = state.health.setdefault(market_id, MarketHealth())
    if health.blacklist_permanent:
        return
    moment = now if now is not None else datetime.now(timezone.utc)
    health.severe_guard_trips += 1
    tier = GUARD_TRIP_COOLOFF_TIERS[
        min(health.severe_guard_trips - 1, len(GUARD_TRIP_COOLOFF_TIERS) - 1)
    ]
    new_until = moment + timedelta(seconds=tier)
    if health.blacklist_until is None or new_until > health.blacklist_until:
        health.blacklist_until = new_until
    logger.warning(
        "severe guard-trip cool-off market=%s trips=%d %dmin est_loss=%s",
        market_id,
        health.severe_guard_trips,
        tier // 60,
        est_loss,
    )
    strat(
        "blacklist",
        market=market_id,
        kind="guard_trip",
        severe_trips=health.severe_guard_trips,
        cooloff_min=tier // 60,
        permanent=0,
    )


def is_event_excluded(state: FarmState, event_slug: str, now: datetime | None = None) -> bool:
    """True while an event family is quarantined after a fill. Timed (auto-expires), so the
    family re-enters once the cooldown passes — mirrors the volatility blacklist semantics."""
    until = state.excluded_events.get(event_slug)
    if until is None:
        return False
    moment = now if now is not None else datetime.now(timezone.utc)
    return until > moment
