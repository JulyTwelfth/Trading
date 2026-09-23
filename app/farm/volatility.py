import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from app.constants import (
    CATASTROPHIC_SINGLE_LOSS,
    DEEP_NET_LOSS,
    FILL_COOLOFF_TIERS,
    LOSS_NOISE_FLOOR,
    MAX_TEMP_BLACKLIST_SECONDS,
    MIN_LOSS_ROUNDTRIPS,
    NET_LOSS_TEMP_MULTIPLIER,
    PERSISTENT_LOSS,
    VOLATILITY_SAMPLE_MAX_AGE_SECONDS,
    VOLATILITY_WINDOWS,
)
from app.farm.schemas import FarmState, MarketHealth
from app.infra.strategy_log import strat

logger = logging.getLogger(__name__)


def blacklist_for_fill(state: FarmState, market_id: str, now: datetime | None = None) -> None:
    moment = now if now is not None else datetime.now(timezone.utc)
    health = state.health.setdefault(market_id, MarketHealth())
    if health.blacklist_permanent:
        return
    health.fill_strikes += 1
    tier = FILL_COOLOFF_TIERS[min(health.loss_roundtrips, len(FILL_COOLOFF_TIERS) - 1)]
    if health.net_fill_pnl < 0:
        tier = min(tier * NET_LOSS_TEMP_MULTIPLIER, MAX_TEMP_BLACKLIST_SECONDS)
    new_until = moment + timedelta(seconds=tier)
    if health.blacklist_until is None or new_until > health.blacklist_until:
        health.blacklist_until = new_until
    logger.info(
        "fill cool-off market=%s %dmin (loss_roundtrips=%d net=%s)",
        market_id,
        tier // 60,
        health.loss_roundtrips,
        health.net_fill_pnl,
    )
    strat(
        "blacklist",
        market=market_id,
        kind="fill",
        cooloff_min=tier // 60,
        loss_roundtrips=health.loss_roundtrips,
        permanent=0,
    )


def market_net(state: FarmState, market_id: str, health: MarketHealth | None = None) -> Decimal:
    health = health if health is not None else state.health.get(market_id)
    if health is None:
        return Decimal(0)
    earned = state.market_rewards.get(market_id, Decimal(0))
    base = state.market_rewards_baseline.get(market_id, earned)
    live_reward = earned - base
    return health.net_fill_pnl + max(live_reward, health.cum_reward_credit)


def record_roundtrip_pnl(
    state: FarmState,
    market_id: str,
    outcome: str,
    net_rt: Decimal,
    closed: bool,
    now: datetime | None = None,
) -> None:
    health = state.health.setdefault(market_id, MarketHealth())
    if health.blacklist_permanent:
        return
    health.net_fill_pnl += net_rt
    health.open_rt_net[outcome] = health.open_rt_net.get(outcome, Decimal(0)) + net_rt
    earned = state.market_rewards.get(market_id, Decimal(0))
    base = state.market_rewards_baseline.get(market_id, earned)
    live_reward = earned - base
    if live_reward > health.cum_reward_credit:
        health.cum_reward_credit = live_reward
    if not closed:
        return

    rt_net = health.open_rt_net.pop(outcome, Decimal(0))
    if rt_net <= -LOSS_NOISE_FLOOR:
        health.loss_roundtrips += 1
    elif rt_net > 0:
        health.loss_roundtrips = max(0, health.loss_roundtrips - 1)

    net_of_reward = health.net_fill_pnl + max(live_reward, health.cum_reward_credit)
    underwater = net_of_reward < 0
    catastrophic = underwater and rt_net <= -CATASTROPHIC_SINGLE_LOSS
    persistent = (
        underwater
        and health.net_fill_pnl <= -PERSISTENT_LOSS
        and health.loss_roundtrips >= MIN_LOSS_ROUNDTRIPS
    )
    deep = underwater and health.net_fill_pnl <= -DEEP_NET_LOSS
    door = (
        "catastrophic" if catastrophic else "persistent" if persistent else "deep" if deep else None
    )
    strat(
        "roundtrip_close",
        market=market_id,
        outcome=outcome,
        rt_net=rt_net,
        net_fill=health.net_fill_pnl,
        loss_rts=health.loss_roundtrips,
        net_of_reward=net_of_reward,
        verdict=door or ("reward_covered" if not underwater else "ok"),
    )
    if door is None:
        return
    health.blacklist_permanent = True
    health.blacklist_until = None
    logger.warning(
        "economic blacklist PERMANENT (%s) market=%s net_fill=%s net_of_reward=%s loss_rts=%d",
        door,
        market_id,
        health.net_fill_pnl,
        net_of_reward,
        health.loss_roundtrips,
    )
    strat(
        "blacklist",
        market=market_id,
        kind="econ",
        door=door,
        permanent=1,
        net_fill=health.net_fill_pnl,
        net_of_reward=net_of_reward,
    )


def clear_ban_if_recovered(state: FarmState, market_id: str) -> bool:
    health = state.health.get(market_id)
    if health is None or (not health.blacklist_permanent and health.blacklist_until is None):
        return False
    if not health.blacklist_permanent and health.net_fill_pnl >= 0:
        return False
    if market_net(state, market_id, health) < 0:
        return False
    health.blacklist_permanent = False
    health.blacklist_until = None
    health.net_fill_pnl = Decimal(0)
    health.loss_roundtrips = 0
    health.severe_guard_trips = 0
    health.open_rt_net.clear()
    logger.info("economic blacklist lifted (net recovered) for market %s", market_id)
    strat("blacklist_lift", market=market_id, kind="recovered")
    return True


def blacklist_for_gap(state: FarmState, market_id: str, now: datetime | None = None) -> None:
    tier_seconds = VOLATILITY_WINDOWS[0][2]
    moment = now if now is not None else datetime.now(timezone.utc)
    health = state.health.setdefault(market_id, MarketHealth())
    if health.blacklist_permanent:
        logger.debug("gap crash on already-permanently-blacklisted market %s", market_id)
        return
    logger.warning("gap crash detected market=%s tier=%dmin", market_id, tier_seconds // 60)
    strat("blacklist", market=market_id, kind="gap", tier_min=tier_seconds // 60)
    new_until = moment + timedelta(seconds=tier_seconds)
    if health.blacklist_until is None or new_until > health.blacklist_until:
        health.blacklist_until = new_until
        logger.warning(
            "gap blacklist extended market=%s until=%s", market_id, new_until.isoformat()
        )


def window_move(
    samples: list[tuple[float, Decimal]], window_seconds: int, now_ts: float
) -> Decimal:
    cutoff = now_ts - window_seconds
    prices = [p for ts, p in samples if ts >= cutoff]
    if len(prices) < 2:
        return Decimal(0)
    return max(prices) - min(prices)


def apply_most_severe(
    health: MarketHealth, moment: datetime, now_ts: float, market_id: str
) -> str | None:
    for window_seconds, threshold, blacklist_seconds in reversed(VOLATILITY_WINDOWS):
        if window_move(health.price_samples, window_seconds, now_ts) < threshold:
            continue
        new_until = moment + timedelta(seconds=blacklist_seconds)
        if health.blacklist_until is not None and health.blacklist_until >= new_until:
            return None
        health.blacklist_until = new_until
        label = f"{blacklist_seconds // 60}min"
        logger.warning(
            "volatility blacklist (%s) market=%s until=%s", label, market_id, new_until.isoformat()
        )
        strat("blacklist", market=market_id, kind="vol", permanent=0, window=window_seconds)
        return label
    return None


def record_price_sample(
    state: FarmState, market_id: str, price: Decimal, now: datetime | None = None
) -> str | None:
    moment = now if now is not None else datetime.now(timezone.utc)
    now_ts = moment.timestamp()
    health = state.health.setdefault(market_id, MarketHealth())

    cutoff = now_ts - VOLATILITY_SAMPLE_MAX_AGE_SECONDS
    health.price_samples = [(ts, p) for ts, p in health.price_samples if ts >= cutoff]
    health.price_samples.append((now_ts, price))

    if health.blacklist_permanent:
        return None
    return apply_most_severe(health, moment, now_ts, market_id)


def is_blacklisted(state: FarmState, market_id: str, now: datetime | None = None) -> bool:
    health = state.health.get(market_id)
    if health is None:
        return False
    if health.blacklist_permanent:
        return True
    if health.blacklist_until is None:
        return False
    moment = now if now is not None else datetime.now(timezone.utc)
    return health.blacklist_until > moment


def reevaluate_blacklist(state: FarmState, market_id: str, now: datetime | None = None) -> None:
    health = state.health.get(market_id)
    if health is None or health.blacklist_permanent or health.blacklist_until is None:
        return
    moment = now if now is not None else datetime.now(timezone.utc)
    if health.blacklist_until > moment:
        return

    now_ts = moment.timestamp()
    still_volatile = any(
        window_move(health.price_samples, window_seconds, now_ts) >= threshold
        for window_seconds, threshold, _ in VOLATILITY_WINDOWS
    )
    if still_volatile:
        health.blacklist_until = moment + timedelta(seconds=VOLATILITY_WINDOWS[0][2])
        logger.info("volatility blacklist re-applied (still volatile) for market %s", market_id)
        strat("blacklist", market=market_id, kind="vol_extend", permanent=0)
    else:
        health.blacklist_until = None
        logger.info("volatility blacklist lifted for market %s", market_id)
        strat("blacklist_lift", market=market_id, kind="vol")
