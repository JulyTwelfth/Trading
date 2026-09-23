import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from app.constants import REWARDS_POLL_INTERVAL_SECONDS
from app.farm.schemas import FarmState, MarketPosition
from app.farm.volatility import clear_ban_if_recovered
from app.infra.strategy_log import strat

logger = logging.getLogger(__name__)


async def fetch_total_earnings(client: Any) -> Decimal:
    return await client.total_earnings_today()


async def fetch_market_earnings(client: Any) -> dict[str, Decimal]:
    return await client.market_earnings_today()


async def fetch_reward_percentages(client: Any) -> dict[str, Decimal]:
    return await client.reward_percentages()


def record_market_earnings(state: FarmState, market_earned: dict[str, Decimal]) -> None:
    for cid, earned in market_earned.items():
        baseline = state.market_rewards_baseline.get(cid)
        if baseline is None or earned < baseline:
            state.market_rewards_baseline[cid] = earned
            baseline = earned
        state.market_rewards[cid] = earned
        clear_ban_if_recovered(state, cid)
        session_earned = earned - baseline
        if earned > 0:
            pos = state.positions.get(cid)
            slug = pos.market.slug if pos is not None else cid
            strat(
                "reward_earned",
                market=cid,
                slug=slug,
                earned_today=earned,
                session=session_earned,
            )


def expected_rewards_per_day(
    percentages: dict[str, Decimal], positions: dict[str, MarketPosition]
) -> Decimal:
    total = Decimal(0)
    for cid, pct in percentages.items():
        pos = positions.get(cid)
        if pos is None:
            logger.debug("reward percentage for unheld market %s; skipping", cid)
            continue
        est = (pct / Decimal(100)) * pos.market.rewards_rate_per_day
        total += est
        if pct > 0:
            strat(
                "reward_market",
                slug=pos.market.slug,
                pct=pct,
                pool=pos.market.rewards_rate_per_day,
                est_day=est,
            )
    return total


async def rewards_poll_loop(client: Any, state: FarmState) -> None:
    while True:
        try:
            state.rewards_earned = await fetch_total_earnings(client)
            record_market_earnings(state, await fetch_market_earnings(client))
            percentages = await fetch_reward_percentages(client)
            state.expected_rewards_per_day = expected_rewards_per_day(percentages, state.positions)
            now = datetime.now(timezone.utc)
            strat(
                "farm_perf",
                earned_today=state.rewards_earned,
                expected_per_day=state.expected_rewards_per_day,
                per_hour=state.rewards_per_hour(),
                elapsed_s=state.elapsed_seconds(now),
                markets=len(state.positions),
                volume=state.total_volume,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("rewards_poll: fetch failed; will retry next tick")
        await asyncio.sleep(REWARDS_POLL_INTERVAL_SECONDS)
